from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pathlib import Path
from tqdm import tqdm
from medsage.config import load_config, ensure_dir
from medsage.embedding import SentenceEmbedder
from medsage.generation import LocalHFGenerator, GenerationConfig
from medsage.io import read_jsonl, write_jsonl
from medsage.prompting import build_base_prompt, build_evidence_prompt, messages_to_plain_text
from medsage.segmentation import split_many
from medsage.aggregation import select_rag_sentences, select_avg_sentences, select_mmr_sentences, select_medsage_sentences

def collect_candidate_sentences(item,cfg):
    answers=[]
    for rec in item.get('retrieved',[]): answers.extend(rec.get('answers',[]))
    return split_many(answers, cfg['sentence'].get('regex',r'[。！？!?；;]+'), int(cfg['sentence'].get('min_sentence_length',4)), int(cfg['sentence'].get('max_sentence_length',180)))

def build_evidence(scheme,item,cfg,embedder):
    top_k=int(cfg['medsage']['sentence_top_k']); candidates=collect_candidate_sentences(item,cfg)
    if scheme=='base' or not candidates: return []
    if scheme=='rag': return select_rag_sentences(candidates,top_k)['selected']
    sent_emb=embedder.encode(candidates,batch_size=int(cfg['runtime'].get('batch_size',64)),show_progress_bar=False); q_emb=embedder.encode([item['question']],batch_size=1,show_progress_bar=False)[0]
    if scheme=='rag_avg': return select_avg_sentences(candidates,sent_emb,q_emb,top_k)['selected']
    if scheme=='rag_mmr': return select_mmr_sentences(candidates,sent_emb,q_emb,top_k,float(cfg['rag_mmr'].get('lambda_mult',0.7)))['selected']
    if scheme=='medsage':
        w=cfg['medsage']['weights']; return select_medsage_sentences(candidates,sent_emb,q_emb,top_k,float(cfg['medsage']['similarity_threshold_tau']),(float(w['centrality']),float(w['relevance']),float(w['diversity'])))['selected']
    raise ValueError(scheme)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/default.yaml'); p.add_argument('--paths',default='configs/paths.yaml'); p.add_argument('--scheme',required=True,choices=['base','rag','rag_avg','rag_mmr','medsage']); p.add_argument('--model',default=None); p.add_argument('--dry-run',action='store_true'); args=p.parse_args(); cfg=load_config(args.config,args.paths)
    model_name=args.model or cfg['models']['default_generation_model']; rows=read_jsonl(cfg['outputs']['retrieval_results']); dry=args.dry_run or bool(cfg['generation'].get('dry_run',False))
    embedder=None if args.scheme=='base' else SentenceEmbedder(cfg['models']['embedding_model_path'],cfg['runtime'].get('device','auto'),bool(cfg['retrieval'].get('normalize_embeddings',True)))
    generator=None
    if not dry:
        mc=cfg['models'][model_name]; generator=LocalHFGenerator(mc['model_path'],mc.get('tokenizer_path',mc['model_path']),mc.get('device_map','auto'),mc.get('torch_dtype','auto'))
    gen_cfg=GenerationConfig(float(cfg['generation'].get('temperature',0.7)),float(cfg['generation'].get('top_p',0.9)),int(cfg['generation'].get('max_new_tokens',1024)),bool(cfg['generation'].get('do_sample',True)))
    outputs=[]
    for item in tqdm(rows, desc=f'Generating {args.scheme}/{model_name}'):
        evidence=build_evidence(args.scheme,item,cfg,embedder)
        messages=build_base_prompt(item['question']) if args.scheme=='base' else build_evidence_prompt(item['question'],evidence)
        answer='' if dry else generator.generate(messages,gen_cfg)
        outputs.append({'question_id':item['query_id'],'question':item['question'],'scheme':args.scheme,'model':model_name,'evidence':evidence,'prompt':messages_to_plain_text(messages),'generated_answer':answer,'references':item.get('references',[])})
    out_dir=Path(cfg['outputs']['generation_dir'])/model_name; ensure_dir(out_dir); write_jsonl(out_dir/f'{args.scheme}.jsonl',outputs); print(f'Saved -> {out_dir}/{args.scheme}.jsonl')
if __name__=='__main__': main()
