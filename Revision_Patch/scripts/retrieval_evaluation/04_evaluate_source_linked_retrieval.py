# -*- coding: utf-8 -*-
"""Stage 4: Evaluate all 49,176 retrieval queries from Stage 3.

IMPORTANT: the five conventional measures in this script use **SOURCE-LINKED**
relevance (sentence.source_qid == query.question_id), not physician-verified
clinical relevance. All reported measure names explicitly reflect that fact.

Evaluates RAG / RAG+MMR / MedSAGE, all Top-10, on the exact 49,176 matched
queries. Does not retrieve, re-encode, change data, call an API, or manufacture
clinical relevance/facet scores. Works with standard-library Python + NumPy
(NumPy is optional for bootstrap CIs; fallback uses a paired normal CI).

Right-click Run in PyCharm. To test with a sandbox/fixed fixture:
  MEDSAGE_STAGE4_BASE=<directory> python this_script.py
and set MEDSAGE_STAGE4_TARGET to the fixture query count.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import os
import random
import sqlite3
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(os.environ.get('MEDSAGE_STAGE4_BASE', r'E:\Desktop\10.14\MedSAGE_FINAL_RETRIEVAL_V1'))
KB = BASE / '01_UNIFIED_EXPERT_KB'
STAGE3 = BASE / '03_FULL49176_RETRIEVAL'
OUT = BASE / '04_FULL49176_RETRIEVAL_EVALUATION'
EXPECTED = int(os.environ.get('MEDSAGE_STAGE4_TARGET', '49176'))
K = 10
METHODS = ('RAG', 'RAG+MMR', 'MedSAGE')
FILENAMES = ('RAG.jsonl', 'RAG_MMR.jsonl', 'MedSAGE.jsonl')
BOOTSTRAP_REPLICATES = int(os.environ.get('MEDSAGE_STAGE4_BOOTSTRAPS', '1000'))
RANDOM_SEED = 20261103
SAMPLE_TO_ANNOTATE = 300

# Explicitly source-linked, NOT medical relevance metrics.
MEASURES = [
    'source_nDCG@10', 'source_Precision@10', 'source_Recall@10',
    'source_MAP@10', 'source_MRR@10',
    'EAC@10', 'same_source_sentence_fraction@10',
    'distinct_source_questions@10', 'distinct_answer_records@10'
]


def log(*args):
    print(*args, flush=True)


def write_json(path, obj):
    tmp = Path(str(path) + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(tmp, path)


def csv_open(path, cols):
    f = path.open('w', encoding='utf-8-sig', newline='')
    w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
    w.writeheader()
    return f, w


def checked_jsonl(path, hasher):
    with path.open('rb') as f:
        for line_number, line in enumerate(f, 1):
            hasher.update(line)
            if not line.strip():
                raise ValueError('%s: blank line %d' % (path, line_number))
            try:
                record = json.loads(line.decode('utf-8-sig'))
            except (ValueError, UnicodeError) as exc:
                raise ValueError('%s: malformed JSON line %d' % (path, line_number)) from exc
            if not isinstance(record, dict):
                raise ValueError('%s: JSON object expected at line %d' % (path, line_number))
            yield record


def load_sqlite_metadata(dbfile):
    if not dbfile.is_file():
        raise FileNotFoundError(str(dbfile))
    con = sqlite3.connect('file:%s?mode=ro' % dbfile.as_posix(), uri=True)
    try:
        q_answers = {str(q): int(n) for q, n in con.execute(
            'SELECT question_id,answer_record_count FROM questions')}
        q_sentences = {str(q): int(n) for q, n in con.execute(
            'SELECT source_qid,COUNT(*) FROM sentence_documents GROUP BY source_qid')}
        sid_lookup = {str(sid): (str(doc), str(qid)) for sid, doc, qid in con.execute(
            'SELECT sid,doc_id,source_qid FROM sentence_documents')}
        return q_answers, q_sentences, sid_lookup
    finally:
        con.close()


def compute_metrics(qid, evidence, answers_total, sentences_total, lookup):
    if len(evidence) != K:
        raise ValueError('%s: Top-%d expected, got %d' % (qid, K, len(evidence)))
    seen = set()
    relevant = []
    source_questions = set()
    answer_documents = set()
    same_docs = set()
    for pos, ev in enumerate(evidence, 1):
        if not isinstance(ev, dict):
            raise ValueError('%s: evidence is not an object' % qid)
        if ev.get('rank') != pos:
            raise ValueError('%s: incorrect rank at position %d' % (qid, pos))
        sid = str(ev.get('sid', ''))
        doc = str(ev.get('doc_id', ''))
        sqid = str(ev.get('source_qid', ''))
        if not sid or sid in seen:
            raise ValueError('%s: missing/repeated sid %s' % (qid, sid))
        seen.add(sid)
        if sid not in lookup:
            raise ValueError('%s: sid %s absent from new KB' % (qid, sid))
        if lookup[sid] != (doc, sqid):
            raise ValueError('%s: mismatched provenance for sid %s' % (qid, sid))
        source_questions.add(sqid)
        answer_documents.add(doc)
        hit = (sqid == qid)
        relevant.append(int(hit))
        if hit:
            same_docs.add(doc)
    R = int(sentences_total)
    if R <= 0 or answers_total <= 0:
        raise ValueError('%s: missing complete source qrels' % qid)
    gains = [1.0 / math.log2(rank + 1) for rank in range(1, K + 1)]
    dcg = sum(g for g, r in zip(gains, relevant) if r)
    idcg = sum(gains[:min(K, R)])
    hits = sum(relevant)
    ap_numer = sum(sum(relevant[:i]) / i for i in range(1, K + 1) if relevant[i - 1])
    # AP@K convention: normalize by min(R,K), explicitly documented.
    ap = ap_numer / min(R, K)
    first = next((i for i, r in enumerate(relevant, 1) if r), None)
    return {
        'source_nDCG@10': dcg / idcg,
        'source_Precision@10': hits / K,
        'source_Recall@10': hits / R,
        'source_MAP@10': ap,
        'source_MRR@10': 1.0 / first if first else 0.0,
        'EAC@10': len(same_docs) / answers_total,
        'same_source_sentence_fraction@10': hits / K,
        'distinct_source_questions@10': float(len(source_questions)),
        'distinct_answer_records@10': float(len(answer_documents)),
    }


def paired_stats(a, b, b_reps=1000, seed=20261103):
    diffs = [float(x - y) for x, y in zip(a, b)]
    if len(diffs) != len(a) or len(diffs) != len(b) or len(diffs) < 2:
        raise ValueError('paired samples mismatch')
    n = len(diffs)
    mean = statistics.fmean(diffs)
    wins = sum(v > 1e-12 for v in diffs)
    ties = sum(abs(v) <= 1e-12 for v in diffs)
    losses = n - wins - ties
    base = {'n': n, 'mean_difference': mean, 'wins': wins, 'ties': ties, 'losses': losses}
    if b_reps > 0:
        try:
            import numpy as np
            rng = np.random.default_rng(seed)
            arr = np.asarray(diffs, dtype=np.float64)
            boots = np.empty(b_reps, dtype=np.float64)
            batch = 16
            for start in range(0, b_reps, batch):
                count = min(batch, b_reps - start)
                ix = rng.integers(0, n, size=(count, n))
                boots[start:start+count] = arr[ix].mean(axis=1)
            ci = np.percentile(boots, [2.5, 97.5]).tolist()
            base.update({'ci_lower': float(ci[0]), 'ci_upper': float(ci[1]),
                         'ci_method': 'paired_percentile_bootstrap', 'bootstrap_replicates': b_reps})
            return base
        except ImportError:
            pass
    sd = statistics.stdev(diffs)
    half = 1.96 * sd / math.sqrt(n)
    base.update({'ci_lower': mean - half, 'ci_upper': mean + half,
                 'ci_method': 'paired_normal_approximation', 'bootstrap_replicates': 0})
    return base


def make_annotation_pool(rows_by_qid, q_answer_counts, path, n=SAMPLE_TO_ANNOTATE):
    # Reproducible, stratified pilot (not representative validation by default).
    groups = {'2': [], '3-4': [], '5+': []}
    for qid in rows_by_qid:
        cnt = q_answer_counts[qid]
        bucket = '2' if cnt == 2 else '3-4' if cnt <= 4 else '5+'
        groups[bucket].append(qid)
    rng = random.Random(RANDOM_SEED)
    selected = []
    for label, ids in groups.items():
        take = min(n // 3, len(ids))
        selected.extend(rng.sample(ids, take))
    if len(selected) < n:
        leftover = sorted(set(rows_by_qid) - set(selected))
        selected.extend(rng.sample(leftover, min(n-len(selected), len(leftover))))
    selected.sort()
    with path.open('w', encoding='utf-8') as outf:
        for qid in selected:
            x = rows_by_qid[qid]
            union = {}
            for method in METHODS:
                for ev in x['evidence'][method]:
                    sid = ev['sid']
                    if sid not in union:
                        union[sid] = {'sid':sid,'doc_id':ev['doc_id'], 'source_qid':ev['source_qid'],
                                      'text':ev['text'], 'present_in_methods':[]}
                    union[sid]['present_in_methods'].append(method)
            line = {'question_id': qid, 'question':x['question'],
                    'answer_record_count':q_answer_counts[qid], 'evidence_to_review':list(union.values())}
            outf.write(json.dumps(line, ensure_ascii=False)+'\n')
    return len(selected)


def main():
    log('=' * 80)
    log('MedSAGE Stage4 | 49,176-query source-linked retrieval evaluation | no API')
    log('IMPORTANT: source-linked ground truth != verified medical relevance')
    log('=' * 80)
    manifest_path = STAGE3 / '03_build_manifest.json'
    if not manifest_path.is_file():
        raise FileNotFoundError('Stage3 manifest missing: '+str(manifest_path))
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    if manifest.get('status') != 'PASS' or manifest.get('completed_queries') != EXPECTED:
        raise RuntimeError('Stage3 PASS/complete queries verification failed')
    if manifest.get('config', {}).get('same_question_id_excluded') is not False:
        raise RuntimeError('This analysis requires the same-question-allowed protocol')
    if manifest.get('top_k') != K:
        raise RuntimeError('Stage3 top_k mismatch')
    q_answer_counts, q_sentence_counts, sid_lookup = load_sqlite_metadata(KB/'expert_kb.sqlite')
    log('[知识库] queries=%s sentences=%s' % (format(len(q_answer_counts), ','), format(len(sid_lookup), ',')))
    if len(q_answer_counts) != EXPECTED:
        raise RuntimeError('Knowledge base query count differs from Stage3')
    paths = [STAGE3 / name for name in FILENAMES]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError('Missing Stage3 result: '+str(path))
    if OUT.exists() and (OUT / '04_eval_manifest.json').exists():
        log('[提示] 已有Stage4报告，本次将重新计算并覆盖Stage4输出，绝不修改Stage1-3文件。')
    OUT.mkdir(parents=True, exist_ok=True)
    detail_cols = ['question_id','method']+MEASURES
    detail_f, detail_writer = csv_open(OUT/'01_per_query_method_metrics.csv',detail_cols)
    results = {method: {key: [] for key in MEASURES} for method in METHODS}
    totals = Counter()
    hashes = [hashlib.sha256() for _ in paths]
    annotation_rows = {}
    seen_qids = set()
    started = time.time()
    try:
        streams = [checked_jsonl(path,h) for path,h in zip(paths,hashes)]
        for n, batch in enumerate(itertools.zip_longest(*streams), 1):
            if any(x is None for x in batch):
                raise RuntimeError('RAG, MMR, MedSAGE output lengths differ')
            qids = [str(x.get('question_id','')) for x in batch]
            if len(set(qids)) != 1:
                raise RuntimeError('Query alignment mismatch at row %d: %s' % (n, qids))
            qid = qids[0]
            if not qid or qid in seen_qids or qid not in q_answer_counts:
                raise RuntimeError('Missing/repeated/unrecognized question id at row %d' % n)
            seen_qids.add(qid)
            if n > EXPECTED:
                raise RuntimeError('More rows than %d' % EXPECTED)
            for method, record in zip(METHODS, batch):
                if record.get('method') != method or record.get('same_question_id_excluded') is not False:
                    raise RuntimeError('Method/protocol mismatch %s %s' % (qid, method))
                metrics = compute_metrics(qid, record.get('evidence',[]), q_answer_counts[qid],
                                          q_sentence_counts.get(qid,0), sid_lookup)
                results[method] = results[method]
                for metric in MEASURES:
                    results[method][metric].append(metrics[metric])
                detail_writer.writerow({'question_id': qid, 'method':method, **metrics})
                totals['evidence_verified'] += len(record['evidence'])
            # Only save short audit sample if requested; avoids retaining entire corpus.
            # Sampling based on deterministic qid hash, then reduce to balanced 300.
            if int(hashlib.sha256(qid.encode('utf-8')).hexdigest()[:8], 16) % 30 == 0:
                annotation_rows[qid] = {'question':batch[0].get('question',''),
                                        'evidence':{method:rec['evidence'] for method,rec in zip(METHODS,batch)}}
            if n % 10000 == 0 or n == EXPECTED:
                log('[已核验]',format(n,','),'/',format(EXPECTED,','))
    finally:
        detail_f.close()
    if len(seen_qids) != EXPECTED:
        raise RuntimeError('Final unique question count mismatch')
    for method, path, sha in zip(METHODS, paths, hashes):
        announced = manifest.get('outputs',{}).get(method,{}).get('sha256')
        if sha.hexdigest() != announced:
            raise RuntimeError('Stage3 file checksum mismatch (%s): file may have changed' % method)
    f, w = csv_open(OUT/'02_source_metric_summary.csv',['method','query_count']+MEASURES)
    try:
        for method in METHODS:
            row={'method':method, 'query_count':EXPECTED}
            for name in MEASURES:
                row[name]=statistics.fmean(results[method][name])
            w.writerow(row)
            log('[结果]', method,'source_nDCG@10=%.6f EAC@10=%.6f' %
                (row['source_nDCG@10'],row['EAC@10']))
    finally:
        f.close()
    f,w=csv_open(OUT/'03_paired_bootstrap.csv',
        ['comparison','metric','n','mean_difference','ci_lower','ci_upper','ci_method',
         'bootstrap_replicates','wins','ties','losses'])
    try:
        for j, comparator in enumerate(('RAG','RAG+MMR')):
            log('[配对Bootstrap] MedSAGE vs '+comparator)
            for k, name in enumerate(MEASURES):
                stat=paired_stats(results['MedSAGE'][name],results[comparator][name],
                                  BOOTSTRAP_REPLICATES, RANDOM_SEED + j*100+k)
                w.writerow({'comparison':'MedSAGE - '+comparator,'metric':name, **stat})
    finally:
        f.close()
    sample_n = make_annotation_pool(annotation_rows,q_answer_counts,
                                    OUT/'04_clinical_annotation_pilot_pool.jsonl',SAMPLE_TO_ANNOTATE)
    f, w = csv_open(OUT/'05_clinical_relevance_annotation_template.csv',
                    ['question_id','sid','clinical_relevance_grade_0_3','reviewer_id','notes'])
    f.close()
    f, w = csv_open(OUT/'06_clinical_facet_annotation_template.csv',
                    ['question_id','facet_id','facet_label','facet_type','sid','stance','reviewer_id','notes'])
    f.close()
    m = {
        'status':'PASS', 'scope':'FULL49176_SOURCE_LINKED_RETRIEVAL_EVALUATION',
        'queries':EXPECTED,'methods':list(METHODS),'top_k':K,'evidence_verified':totals['evidence_verified'],
        'same_question_id_excluded':False,
        'source_link_relevance_definition':'evidence.source_qid == query.question_id',
        'source_Recall@10_denominator':'all sentence records in the KB with the same question_id',
        'source_MAP@10_normalizer':'min(total same-question sentence records,10)',
        'EAC_definition':'unique same-question doc_id among Top10 divided by all answer records for that question',
        'clinical_relevance_judged':False,'clinical_facet_metrics_judged':False,
        'NOT_COMPUTED':['clinical_nDCG@10','clinical_P@10','clinical_Recall@10','clinical_MAP@10',
                        'clinical_MRR@10','CFC@10','CSC@10','CVC@10','alpha_nDCG@10'],
        'experimental_limitations':[
            'Source-linked metrics measure return of historical records, NOT independent clinical usefulness.',
            'MedSAGE uses question/answer retrieval channels, while baselines search global sentence vectors; candidate pools differ.',
            'Historical answer rows have no verified independent doctor_id.',
            'The Stage3 implementation was not proven identical to the historical frozen V4.',
            'Same-question historical answers are retained: not an unseen-query generalization test.'
        ],
        'statistical_method': 'Paired percentile bootstrap with %d resamples (or normal approximation if NumPy unavailable)' % BOOTSTRAP_REPLICATES,
        'clinical_annotation_pilot_sample_questions':sample_n,
        'stage3_manifest_config':manifest.get('config',{}),
        'created_at':time.strftime('%Y-%m-%d %H:%M:%S')
    }
    write_json(OUT/'04_eval_manifest.json',m)
    log('\n[完成] Stage4来源关联全量评价已完成：'+str(OUT))
    log('[请上传] 02_source_metric_summary.csv / 03_paired_bootstrap.csv / 04_eval_manifest.json')
    log('[重要] 临床医学相关性和CFC/CSC/CVC/α-nDCG仍待独立标注；脚本未生成这些分数。')

if __name__ == '__main__':
    main()
