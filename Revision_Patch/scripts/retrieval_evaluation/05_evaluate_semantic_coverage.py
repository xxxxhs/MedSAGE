# -*- coding: utf-8 -*-
"""MedSAGE Stage 5 — full-corpus SEMANTIC RETRIEVAL macro evaluation.

This is NOT Table 9, NOT entity precision/recall, and NOT a physician-labelled
clinical relevance benchmark. It evaluates *automatically identified semantic
facets* from the historical physician answer sentences, using a fixed, frozen
embedding and reference rule applied equally to RAG / RAG+MMR / MedSAGE.

Right-click Run in PyCharm using the existing pythonProject interpreter.
No FAISS, no model loading, no API, no modification of Stage 1–4 assets.

For an isolated test fixture, set:
  MEDSAGE_STAGE5_BASE=/path/to/test_root
  MEDSAGE_STAGE5_TARGET=3
  MEDSAGE_STAGE5_BOOTSTRAPS=100

WARNING: similarity decisions use the SAME M3E-Small vectors as Stage 3 retrieval.
The score is an automatic retrieval-semantic proxy, not independently judged
clinical correctness, entailment, or verified clinical consensus. CVC omitted.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import sqlite3
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

BASE = Path(os.environ.get('MEDSAGE_STAGE5_BASE', r'E:\Desktop\10.14\MedSAGE_FINAL_RETRIEVAL_V1'))
KB = BASE / '01_UNIFIED_EXPERT_KB'
IDX = BASE / '02_SHARED_EMBEDDINGS_HNSW'
RETR = BASE / '03_FULL49176_RETRIEVAL'
OUT = BASE / '05_FULL49176_MACRO_SEMANTIC_EVALUATION'
EXPECTED = int(os.environ.get('MEDSAGE_STAGE5_TARGET', '49176'))
BOOT = int(os.environ.get('MEDSAGE_STAGE5_BOOTSTRAPS', '1000'))
METHODS = ('RAG', 'RAG+MMR', 'MedSAGE')
FILES = ('RAG.jsonl', 'RAG_MMR.jsonl', 'MedSAGE.jsonl')
TOP_K = 10
CLUSTER_COS = 0.82   # fixed before comparison, no results-driven tuning
MATCH_COS = 0.82     # independently fixed reference facet match threshold
DUP_COS = 0.85       # within-Top10 near-duplicate pair threshold
ALPHA = 0.5         # alpha-nDCG novelty penalty
RANDOM_SEED = 20261103
COMMIT_BATCH = 200
VERSION = 'stage5_20261103_v2_reference_plus_pool_facets'
METRIC_NAMES = (
    'auto_SFC@10',                 # reference-derived semantic facet coverage
    'auto_cross_source_SFC@10',    # fraction reference facets retrieved via OTHER source QIDs
    'auto_CSC_proxy@10',           # consensus-like facet gets >=2 distinct retrieved answer records
    'auto_consensus_any@10',       # consensus-like facet seen >=1 time
    'pooled_auto_alpha_nDCG@10',   # ideal candidates = reference sentences + pooled 3xTop10, not global gold
    'near_duplicate_pair_fraction@10',  # lower is better; 45 unordered sentence pairs
)


def log(*args):
    print(*args, flush=True)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump_json(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
    os.replace(tmp, path)


def load_json(path):
    with Path(path).open('r', encoding='utf-8-sig') as f:
        return json.load(f)


def checked_pair_manifests():
    p1, p2, p3 = KB/'00_build_manifest.json', IDX/'02_build_manifest.json', RETR/'03_build_manifest.json'
    for p in (p1, p2, p3):
        if not p.is_file():
            raise FileNotFoundError('Missing previous-stage manifest: ' + str(p))
    m1, m2, m3 = [load_json(p) for p in (p1, p2, p3)]
    if any(m.get('status') != 'PASS' for m in (m1, m2, m3)):
        raise ValueError('Stage 1/2/3 must all have PASS manifests')
    if m1.get('selected_question_count') != EXPECTED or m2.get('questions') != EXPECTED or m3.get('completed_queries') != EXPECTED:
        raise ValueError('Previous-stage query counts do not agree with target=%s' % EXPECTED)
    if m1.get('excludes_same_question_id') is not False or m2.get('same_question_id_excluded') is not False or m3.get('config',{}).get('same_question_id_excluded') is not False:
        raise ValueError('Same-question answer protocol unexpectedly changed')
    if m3.get('top_k') != TOP_K or m3.get('config',{}).get('scheme_version') is None:
        raise ValueError('Unexpected Stage3 protocol')
    if m3.get('stage2_manifest_sha256') != sha256_file(p2):
        raise ValueError('Stage2 manifest changed since Stage3, cannot verify provenance')
    expected_sha = {meth: m3['outputs'][meth]['sha256'] for meth in METHODS}
    cfg = dict(version=VERSION, expected_queries=EXPECTED, source_methods=METHODS,
               stage3_scheme=m3['config']['scheme_version'], cluster_cos=CLUSTER_COS,
               match_cos=MATCH_COS, dup_cos=DUP_COS, alpha=ALPHA,
               top_k=TOP_K, consensus_min_different_answer_records=2,
               use_retrieval_encoder_for_reference_matching=True,
               use_reference_sentences_plus_pooled_top10_for_alpha_ndcg_normalizer=True,
               no_entity_dictionary=True, no_llm_api=True, no_human_annotation=True,
               stage3_output_sha256=expected_sha,
               stage1_manifest_sha256=sha256_file(p1),
               stage2_manifest_sha256=sha256_file(p2),
               stage3_manifest_sha256=sha256_file(p3))
    return cfg


def load_maps_and_counts(np):
    qfile = KB / 'questions_49176.jsonl'
    sidmap = IDX / 'sentences_position_map.jsonl'
    vecfile = IDX / 'sentences_embeddings.npy'
    for x in (qfile,sidmap,vecfile):
        if not x.is_file():
            raise FileNotFoundError(str(x))
    svec = np.load(vecfile, mmap_mode='r', allow_pickle=False)
    if svec.dtype != np.float32 or svec.ndim != 2 or svec.shape[1] != 512:
        raise ValueError('Expected float32 (n,512) normalized embeddings')
    qids = []
    answer_counts = {}
    questions = {}
    with qfile.open('r', encoding='utf-8-sig') as f:
        for ln, line in enumerate(f,1):
            o=json.loads(line)
            qid=str(o['question_id'])
            if qid in answer_counts: raise ValueError('Repeated QID in original reference: '+qid)
            qids.append(qid)
            answer_counts[qid] = int(o['answer_record_count'])
            questions[qid] = o['question']
    if len(qids)!=EXPECTED: raise ValueError('Wrong query count in reference file')
    by_q = defaultdict(list)
    sid_to_pos = {}
    sid_to_doc = {}
    sid_to_qid = {}
    with sidmap.open('r', encoding='utf-8-sig') as f:
        for pos, line in enumerate(f):
            o = json.loads(line)
            sid=str(o['sid']); qid=str(o['source_qid']); doc=str(o['doc_id'])
            if o['row']!=pos or sid in sid_to_pos:
                raise ValueError('Unstable sid position mapping at row %d' % pos)
            if qid not in answer_counts:
                raise ValueError('Source QID not found in original query reference: '+qid)
            sid_to_pos[sid]=pos; sid_to_doc[sid]=doc; sid_to_qid[sid]=qid
            by_q[qid].append(pos)
    if len(sid_to_pos)!=svec.shape[0]:
        raise ValueError('Sentence count mismatch: position mapping / vector matrix')
    for qid in qids:
        if not by_q.get(qid): raise ValueError('Reference has no source sentences: '+qid)
    log('[参考数据] queries=%s, sentences=%s' % (format(len(qids),','),format(len(sid_to_pos),',')))
    return qids,answer_counts,questions,by_q,sid_to_pos,sid_to_doc,sid_to_qid,svec


def cluster_reference(sim, doc_ids):
    """Deterministic union-find sentence clusters under fixed cosine threshold.

    This is a proxy for semantic aspects, not a validated clinical facet classifier.
    Connected components may merge semantically adjacent claims transitively.
    """
    n=len(doc_ids)
    parent=list(range(n))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]
            i=parent[i]
        return i
    for i in range(n):
        for j in range(i+1,n):
            if float(sim[i,j])>=CLUSTER_COS:
                a,b=find(i),find(j)
                if a!=b: parent[b]=a
    clusters={}
    for i in range(n): clusters.setdefault(find(i),[]).append(i)
    return sorted(clusters.values(),key=lambda xs:xs[0])


def aspect_match_matrix(np, retrieved, refvec, clusters):
    """Boolean evidence-by-facet matrix. A match must pass fixed cosine rule.
    Clusters are reference-only; method's retrieval order does not form facets.
    """
    if len(retrieved)==0: return np.zeros((0,len(clusters)),dtype=bool)
    sims=np.asarray(retrieved @ refvec.T, dtype=np.float32)
    out=np.zeros((len(retrieved),len(clusters)),dtype=bool)
    for j,members in enumerate(clusters):
        out[:,j]=np.max(sims[:,members],axis=1)>=MATCH_COS
    return out


def alpha_dcg(np, matches, alpha=ALPHA, k=TOP_K):
    seen=np.zeros(matches.shape[1],dtype=np.int32)
    total=0.0
    for rank,row in enumerate(matches[:k],1):
        gain=float((row.astype(np.float64)*(1-alpha)**seen).sum())
        total+=gain/math.log2(rank+1)
        seen+=row.astype(np.int32)
    return total


def pooled_ideal_dcg(np, pooled_matches, alpha=ALPHA, k=TOP_K):
    """Greedy ideal rank from original reference sentences PLUS pooled top10.

    Not an exhaustive gold-document ideal. This is an automatic alpha-nDCG proxy.
    """
    n, facets=pooled_matches.shape
    if not n or not facets: return 0.0
    used=np.zeros(n,dtype=bool)
    seen=np.zeros(facets,dtype=np.int32)
    total=0.0
    for rank in range(1,min(k,n)+1):
        gain=pooled_matches @ np.power(1.0-alpha,seen)
        gain[used]=-1.0
        ix=int(np.argmax(gain))
        if gain[ix]<=0: break
        total+=float(gain[ix])/math.log2(rank+1)
        used[ix]=True
        seen+=pooled_matches[ix].astype(np.int32)
    return total


def calc_method_metrics(np,qid,ev,mat,clusters,support_counts,alpha_idcg,vecs):
    if len(ev)!=TOP_K or mat.shape[0]!=TOP_K:
        raise ValueError('Top10 required: '+qid)
    nfacets=len(clusters)
    hits=np.any(mat,axis=0)
    allcov=float(np.count_nonzero(hits)/nfacets)
    crossmask=np.array([str(o['source_qid'])!=qid for o in ev],dtype=bool)
    crosscov=float(np.count_nonzero(np.any(mat[crossmask,:],axis=0))/nfacets) if np.any(crossmask) else 0.0
    consensus=[f for f,x in enumerate(support_counts) if x>=2]
    csc=None; many_any=None
    if consensus:
        any_cov=0; two_docs=0
        for j in consensus:
            matching_docs={str(ev[i]['doc_id']) for i in range(TOP_K) if mat[i,j]}
            if matching_docs: any_cov+=1
            if len(matching_docs)>=2: two_docs+=1
        csc=float(two_docs/len(consensus))
        many_any=float(any_cov/len(consensus))
    alpha_score=(float(alpha_dcg(np,mat)/alpha_idcg) if alpha_idcg>1e-12 else None)
    if alpha_score is not None:
        if alpha_score < -1e-7 or alpha_score > 1+1e-7:
            raise ValueError('Invalid normalized pooled alpha-nDCG: %r for query %s' % (alpha_score,qid))
        alpha_score = max(0.0,min(1.0,alpha_score))
    similarity=vecs@vecs.T
    upper=np.triu_indices(TOP_K,1)
    near_dup=float(np.mean(similarity[upper]>=DUP_COS))
    return {
        'auto_SFC@10':allcov, 'auto_cross_source_SFC@10':crosscov,
        'auto_CSC_proxy@10':csc, 'auto_consensus_any@10':many_any,
        'pooled_auto_alpha_nDCG@10':alpha_score,
        'near_duplicate_pair_fraction@10':near_dup,
        'num_matched_facets':int(np.count_nonzero(hits)),
        'num_consensus_facets':len(consensus), 'num_reference_facets':nfacets,
        'n_external_evidence':int(crossmask.sum()),
        'n_same_question_evidence':int(TOP_K-crossmask.sum()),
    }


def new_cache(db_path, cfg):
    con=sqlite3.connect(str(db_path))
    con.execute('PRAGMA journal_mode=WAL')
    con.execute('PRAGMA synchronous=FULL')
    con.executescript('''
    CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS processed (query_index INTEGER PRIMARY KEY, question_id TEXT NOT NULL UNIQUE);
    CREATE TABLE IF NOT EXISTS facets (
      query_index INTEGER NOT NULL, question_id TEXT NOT NULL, facet_id TEXT NOT NULL,
      representative_sid TEXT NOT NULL, reference_sentence_count INTEGER NOT NULL,
      supporting_answer_record_count INTEGER NOT NULL, member_sids_json TEXT NOT NULL,
      member_doc_ids_json TEXT NOT NULL, PRIMARY KEY(question_id,facet_id));
    CREATE TABLE IF NOT EXISTS metrics (
      query_index INTEGER NOT NULL, question_id TEXT NOT NULL, method TEXT NOT NULL,
      answer_record_count INTEGER NOT NULL, facet_count INTEGER NOT NULL,
      consensus_facet_count INTEGER NOT NULL,
      auto_sfc REAL, auto_external_sfc REAL, auto_csc REAL, auto_consensus_any REAL,
      pooled_alpha_ndcg REAL, near_duplicate_frac REAL,
      matched_facets INTEGER NOT NULL, n_external_evidence INTEGER NOT NULL,
      n_same_question_evidence INTEGER NOT NULL,
      matched_facet_ids_json TEXT NOT NULL,
      PRIMARY KEY(question_id,method));
    ''')
    encoded=json.dumps(cfg,ensure_ascii=False,sort_keys=True)
    row=con.execute('SELECT value FROM meta WHERE key=?',('protocol_config',)).fetchone()
    if row and row[0]!=encoded:
        con.close()
        raise ValueError('Stage5 analysis cache is from a DIFFERENT protocol/input. Keep it and use a new output directory; do not mix runs.')
    con.execute('INSERT OR IGNORE INTO meta(key,value) VALUES (?,?)',('protocol_config',encoded))
    con.commit()
    return con


def do_one_query(np,idx,qid,answer_counts,by_q,sid_to_pos,sid_to_doc,sid_to_qid,svec,records):
    refs=by_q[qid]
    ref_vec=np.asarray(svec[refs],dtype=np.float32)
    # Numerically defensible normalization guard (sample all reference vectors for this query).
    if np.any(np.abs(np.linalg.norm(ref_vec,axis=1)-1)>5e-3):
        raise ValueError('Reference embeddings appear not normalized: '+qid)
    ref_sids=[];ref_doc_ids=[]
    for pos in refs:
        # Reverse lookup is avoided by passing inverse global row maps in outer setup.
        ref_sids.append(_POS_TO_SID[pos])
        ref_doc_ids.append(sid_to_doc[_POS_TO_SID[pos]])
    if len(set(ref_doc_ids)) != int(answer_counts[qid]):
        raise ValueError('Historical answer record count inconsistent with sentence mapping at '+qid)
    sim=ref_vec@ref_vec.T
    clusters=cluster_reference(sim,ref_doc_ids)
    support_counts=[len({ref_doc_ids[j] for j in c}) for c in clusters]
    facets_to_write=[]
    for n,members in enumerate(clusters):
        sids=[ref_sids[j] for j in members]
        docs=sorted({ref_doc_ids[j] for j in members})
        facets_to_write.append((idx,qid,'%s_F%04d'%(qid,n+1),sids[0],len(members),len(docs),
                                json.dumps(sids,ensure_ascii=False,separators=(',',':')),
                                json.dumps(docs,ensure_ascii=False,separators=(',',':'))))
    # Union (up to 30 unique sentences) is fixed for all compared methods.
    union=[];seen=set();local_rows=[]
    for meth, rec in zip(METHODS,records):
        if str(rec['question_id'])!=qid or str(rec['method'])!=meth:
            raise ValueError('Retrieved query/method mismatch %s %s' % (qid,meth))
        evid=rec['evidence']
        if len(evid)!=TOP_K or len({str(x['sid']) for x in evid})!=TOP_K:
            raise ValueError('Missing/duplicate sid in Top10: '+qid+' '+meth)
        p=[]
        for rank,e in enumerate(evid,1):
            if int(e.get('rank',-1))!=rank:
                raise ValueError('Wrong evidence rank at %s method %s' % (qid,meth))
            sid=str(e['sid'])
            if sid not in sid_to_pos or str(e['doc_id'])!=sid_to_doc[sid] or str(e['source_qid'])!=sid_to_qid[sid]:
                raise ValueError('Incorrect evidence SID provenance at '+qid+' '+meth+' '+sid)
            if sid not in seen:
                seen.add(sid);union.append(sid)
            p.append(sid)
        local_rows.append(p)
    retrieved_vecs=np.asarray(svec[[sid_to_pos[sid] for sid in union]],dtype=np.float32)
    pooled_matches=aspect_match_matrix(np,retrieved_vecs,ref_vec,clusters)
    pool_index={sid:i for i,sid in enumerate(union)}
    # The fixed ideal-ranking candidate set includes ALL historical reference
    # sentences (not just the retrieved ones), plus pooled method outputs.
    extra_ref_sids=[sid for sid in ref_sids if sid not in pool_index]
    ideal_vecs = (np.concatenate([retrieved_vecs, np.asarray(svec[[sid_to_pos[s] for s in extra_ref_sids]],dtype=np.float32)],axis=0)
                  if extra_ref_sids else retrieved_vecs)
    ideal_matches=aspect_match_matrix(np,ideal_vecs,ref_vec,clusters)
    idcg=pooled_ideal_dcg(np,ideal_matches)
    # Greedy maximization is approximate: the shared ideal must at least be as
    # large as the score of each observed method, to keep normalized values <=1.
    observed_dcg=[alpha_dcg(np,pooled_matches[[pool_index[str(e['sid'])] for e in rec['evidence']]])
                  for rec in records]
    idcg=max([idcg]+observed_dcg)
    metrics_to_write=[]
    for method, rec, sids in zip(METHODS,records,local_rows):
        indices=[pool_index[s] for s in sids]
        match=pooled_matches[indices,:]
        vecs=retrieved_vecs[indices,:]
        metrics=calc_method_metrics(np,qid,rec['evidence'],match,clusters,support_counts,idcg,vecs)
        match_ids=['%s_F%04d'%(qid,j+1) for j,covered in enumerate(np.any(match,axis=0)) if covered]
        metrics_to_write.append((idx,qid,method,answer_counts[qid],len(clusters),metrics['num_consensus_facets'],
          metrics['auto_SFC@10'],metrics['auto_cross_source_SFC@10'],metrics['auto_CSC_proxy@10'],
          metrics['auto_consensus_any@10'],metrics['pooled_auto_alpha_nDCG@10'],
          metrics['near_duplicate_pair_fraction@10'],metrics['num_matched_facets'],
          metrics['n_external_evidence'],metrics['n_same_question_evidence'],
          json.dumps(match_ids,ensure_ascii=False,separators=(',',':'))))
    return facets_to_write,metrics_to_write


_POS_TO_SID=[]


def process_all(np,con,cfg,qids,answer_counts,by_q,sid_to_pos,sid_to_doc,sid_to_qid,svec):
    processed={int(row[0]):str(row[1]) for row in con.execute('SELECT query_index,question_id FROM processed')}
    log('[断点复用] %s/%s题已写入缓存' %(format(len(processed),','),format(EXPECTED,',')))
    paths=[RETR/f for f in FILES]
    for path in paths:
        if not path.is_file(): raise FileNotFoundError(str(path))
    hashers=[hashlib.sha256() for _ in METHODS]
    last_time=time.monotonic(); done_new=0
    handles=[p.open('rb') for p in paths]
    try:
        for idx in range(EXPECTED):
            recs=[]
            for h,hs in zip(handles,hashers):
                line=h.readline()
                if not line: raise ValueError('Stage3 retrieval file is shorter than %s queries' %EXPECTED)
                hs.update(line)
                recs.append(json.loads(line.decode('utf-8-sig')))
            qid=qids[idx]
            if any(str(r['question_id'])!=qid for r in recs):
                raise ValueError('Stage3 queries not aligned with reference question order at row '+str(idx))
            if idx in processed:
                if processed[idx]!=qid: raise ValueError('Checkpoint query mismatch')
                continue
            facets,metrics=do_one_query(np,idx,qid,answer_counts,by_q,sid_to_pos,sid_to_doc,sid_to_qid,svec,recs)
            con.executemany('INSERT INTO facets VALUES (?,?,?,?,?,?,?,?)',facets)
            con.executemany('INSERT INTO metrics VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',metrics)
            con.execute('INSERT INTO processed VALUES (?,?)',(idx,qid))
            done_new+=1
            if done_new%COMMIT_BATCH==0:
                con.commit()
                if (idx+1)%1000<COMMIT_BATCH:
                    log('[已计算] %s/%s题，新增%s题，间隔%.1f秒' % (format(idx+1,','),format(EXPECTED,','),format(done_new,','),time.monotonic()-last_time))
                    last_time=time.monotonic()
        con.commit()
        for p,h in zip(paths,handles):
            extra=h.readline()
            if extra: raise ValueError('Stage3 retrieval file has more rows than expected: '+str(p))
    finally:
        for f in handles:f.close()
    for method,hs in zip(METHODS,hashers):
        if hs.hexdigest()!=cfg['stage3_output_sha256'][method]:
            raise ValueError('Stage3 retrieval SHA256 CHANGED for method '+method)
    if con.execute('SELECT COUNT(*) FROM processed').fetchone()[0]!=EXPECTED:
        raise ValueError('Stage5 incomplete; no final result table will be published')
    if con.execute('SELECT COUNT(*) FROM metrics').fetchone()[0]!=EXPECTED*len(METHODS):
        raise ValueError('Stage5 method output counts incomplete')
    log('[全部计算完成] %s题 × 3种方法；开始生成正式CSV与配对统计' %format(EXPECTED,','))


def csv_write(path,head,rows):
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(head)
        for r in rows:
            writer.writerow(['' if x is None else x for x in r])
    os.replace(tmp,path)


def export_csvs(con):
    facet_csv=OUT/'05_Retrieval_Semantic_Facets_49176.csv'
    metric_csv=OUT/'05_Full_Semantic_Metrics_49176.csv'
    csv_write(facet_csv,
      ['query_index','question_id','facet_id','representative_sid','reference_sentence_count',
       'supporting_answer_record_count','member_sids_json','member_doc_ids_json'],
      con.execute('SELECT query_index,question_id,facet_id,representative_sid,reference_sentence_count,supporting_answer_record_count,member_sids_json,member_doc_ids_json FROM facets ORDER BY query_index,facet_id'))
    cols=['query_index','question_id','method','answer_record_count','facet_count','consensus_facet_count',
          'auto_SFC@10','auto_cross_source_SFC@10','auto_CSC_proxy@10','auto_consensus_any@10',
          'pooled_auto_alpha_nDCG@10','near_duplicate_pair_fraction@10',
          'num_matched_facets','n_external_evidence','n_same_question_evidence','matched_facet_ids_json']
    csv_write(metric_csv,cols,con.execute('SELECT * FROM metrics ORDER BY query_index,CASE method WHEN "RAG" THEN 0 WHEN "RAG+MMR" THEN 1 ELSE 2 END'))
    # Separate summary from all raw scores; denominator and missingness documented.
    dbcols=['auto_sfc','auto_external_sfc','auto_csc','auto_consensus_any','pooled_alpha_ndcg','near_duplicate_frac']
    summary=[]
    for meth in METHODS:
        for name,col in zip(METRIC_NAMES,dbcols):
            n,mean,stdev=con.execute('SELECT COUNT(%s),AVG(%s),AVG(%s*%s) FROM metrics WHERE method=?'%(col,col,col,col),(meth,)).fetchone()
            std=math.sqrt(max(0.0,stdev-mean*mean)) if n and mean is not None else None
            summary.append((meth,name,int(n),EXPECTED-int(n),mean,std))
    csv_write(OUT/'05_Semantic_Method_Summary.csv',
      ['method','metric','eligible_queries','not_applicable_queries','macro_mean','population_std'],summary)
    strata_sql='CASE WHEN answer_record_count=2 THEN "2" WHEN answer_record_count<=4 THEN "3-4" WHEN answer_record_count<=7 THEN "5-7" ELSE "8+" END'
    strata=[]
    for meth in METHODS:
        for stratum in ('2','3-4','5-7','8+'):
            n_all=con.execute('SELECT COUNT(*) FROM metrics WHERE method=? AND '+strata_sql+'=?',(meth,stratum)).fetchone()[0]
            for name,col in zip(METRIC_NAMES,dbcols):
                n,avg=con.execute('SELECT COUNT(%s),AVG(%s) FROM metrics WHERE method=? AND %s=?'%(col,col,strata_sql),
                                  (meth,stratum)).fetchone()
                strata.append((stratum,meth,name,int(n_all),int(n),n_all-int(n),avg))
    csv_write(OUT/'05_AnswerCount_Stratified_Summary.csv',
              ['answers_per_question','method','metric','total_queries','eligible_queries','not_applicable_queries','macro_mean'],strata)
    return {'facets':facet_csv,'metrics':metric_csv}


def paired_bootstrap(np,con):
    pairs=[]
    cols=['auto_sfc','auto_external_sfc','auto_csc','auto_consensus_any','pooled_alpha_ndcg','near_duplicate_frac']
    rng=np.random.default_rng(RANDOM_SEED)
    for baseline in ('RAG','RAG+MMR'):
        for metric,col in zip(METRIC_NAMES,cols):
            rows=list(con.execute('SELECT med.%s,base.%s FROM metrics med JOIN metrics base ON med.question_id=base.question_id WHERE med.method="MedSAGE" AND base.method=? AND med.%s IS NOT NULL AND base.%s IS NOT NULL ORDER BY med.query_index'% (col,col,col,col),(baseline,)))
            if not rows:
                pairs.append((baseline,metric,0,None,None,None,None,None,None,BOOT))
                continue
            vals=np.asarray(rows,dtype=np.float64)
            diffs=vals[:,0]-vals[:,1]
            mean=float(diffs.mean())
            win=int((diffs>1e-12).sum()); tie=int((np.abs(diffs)<=1e-12).sum());loss=len(diffs)-win-tie
            boots=np.empty(BOOT,dtype=np.float64)
            for j in range(0,BOOT,16):
                count=min(16,BOOT-j)
                ids=rng.integers(0,len(diffs),(count,len(diffs)))
                boots[j:j+count]=diffs[ids].mean(axis=1)
            ci=np.percentile(boots,[2.5,97.5])
            pairs.append((baseline,metric,len(diffs),mean,float(ci[0]),float(ci[1]),win,tie,loss,BOOT))
        log('[配对Bootstrap]', 'MedSAGE vs',baseline)
    csv_write(OUT/'05_Semantic_Paired_Bootstrap.csv',
              ['baseline_method','metric','paired_eligible_queries','mean_delta_MedSAGE_minus_baseline','ci95_lower','ci95_upper','wins','ties','losses','bootstrap_replicates'],pairs)


def build_manifest(cfg,con):
    nfacet=con.execute('SELECT COUNT(*) FROM facets').fetchone()[0]
    ncons=con.execute('SELECT COUNT(*) FROM facets WHERE supporting_answer_record_count>=2').fetchone()[0]
    eligible=con.execute('SELECT COUNT(*) FROM metrics WHERE method="MedSAGE" AND auto_csc IS NOT NULL').fetchone()[0]
    eligalpha=con.execute('SELECT COUNT(*) FROM metrics WHERE method="MedSAGE" AND pooled_alpha_ndcg IS NOT NULL').fetchone()[0]
    stats=con.execute('SELECT MIN(facet_count),AVG(facet_count),MAX(facet_count) FROM metrics WHERE method="MedSAGE"').fetchone()
    result={'status':'PASS','protocol':cfg,'queries_evaluated':EXPECTED,'methods':list(METHODS),'top_k':TOP_K,
       'source_sentences_as_facets':int(nfacet),'facets_supported_by_2plus_different_answer_records':int(ncons),
       'queries_with_any_consensus_like_facet':int(eligible),
       'queries_with_pooled_alpha_ndcg_eligible':int(eligalpha),
       'facets_per_question_min_avg_max':[stats[0],stats[1],stats[2]],
       'method_metric_rows':EXPECTED*3,
       'outputs':['05_Retrieval_Semantic_Facets_49176.csv','05_Full_Semantic_Metrics_49176.csv',
           '05_Semantic_Method_Summary.csv','05_AnswerCount_Stratified_Summary.csv',
           '05_Semantic_Paired_Bootstrap.csv','05_analysis_cache.sqlite'],
       'not_computed':['CVC: no reliable context-aware medical contradiction labels',
           'true clinical-facet CFC: automatic semantic clusters do not constitute medically validated clinical facets',
           'physician-verified medical relevance nDCG/MAP',
           'independently evaluated medical correctness'],
       'cautions':['Medical reference derives from original physician answers; same-QID historical answers are eligible, per protocol.',
           'Automatic semantic clusters use retrieval M3E-Small embeddings and thus are not independent medical judgments.',
           'Both facets and aspect relevance are proxy annotations; automatic semantic similarity may merge incompatible claims.',
           'Pooled alpha-nDCG ideal ranking is greedily selected from all original reference sentences plus union of the three methods Top10; not an exhaustive-corpus ideal.',
           'Distinct answer records do not establish distinct clinician identities.',
           'Stage3 method candidate pools differ; overall pipeline effects cannot be causally attributed solely to C-R-D.'],
       'finished_at':time.strftime('%Y-%m-%d %H:%M:%S')}
    dump_json(OUT/'05_Evaluation_Manifest.json',result)
    return result


def main():
    if BOOT<=0: raise ValueError('Bootstrap count must be > 0')
    import numpy as np
    log('='*80)
    log('MedSAGE Stage5 | 49,176-query macro semantic retrieval evaluation')
    log('New unsupervised semantic facets; NO Table9 entity metric; NO API')
    log('='*80)
    cfg=checked_pair_manifests()
    OUT.mkdir(parents=True, exist_ok=True)
    qids,acount,questions,by_q,sidpos,siddoc,sidqid,svec=load_maps_and_counts(np)
    global _POS_TO_SID
    _POS_TO_SID=[None]*len(sidpos)
    for sid,pos in sidpos.items(): _POS_TO_SID[pos]=sid
    db=OUT/'05_analysis_cache.sqlite'
    con=new_cache(db,cfg)
    try:
        process_all(np,con,cfg,qids,acount,by_q,sidpos,siddoc,sidqid,svec)
        export_csvs(con)
        paired_bootstrap(np,con)
        manifest=build_manifest(cfg,con)
    finally:
        con.close()
    log('\n[完成] Stage5宏观检索实验',manifest['status'])
    log('[目录]', OUT)
    log('[只需上传] 05_Evaluation_Manifest.json / 05_Semantic_Method_Summary.csv / 05_AnswerCount_Stratified_Summary.csv / 05_Semantic_Paired_Bootstrap.csv')
    log('[注意] 不得把自动语义方面分数写成已验证的临床事实准确率；CVC未计算。')


if __name__=='__main__':
    try:
        main()
    except Exception as e:
        import traceback
        log('[失败]',type(e).__name__,str(e))
        traceback.print_exc()
        sys.exit(1)
