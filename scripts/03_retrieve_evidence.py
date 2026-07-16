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
from medsage.io import read_json, read_jsonl, write_jsonl
from medsage.retrieval import load_faiss_index, search_index, collect_retrieved_answers

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/default.yaml'); p.add_argument('--paths',default='configs/paths.yaml'); args=p.parse_args(); cfg=load_config(args.config,args.paths)
    rows=read_jsonl(cfg['outputs']['prepared_questions']); qid_to_answers=read_json(cfg['outputs']['qid_to_answers']); meta=read_json(str(Path(cfg['outputs']['faiss_index']).with_suffix('.meta.json'))); index=load_faiss_index(cfg['outputs']['faiss_index'])
    embedder=SentenceEmbedder(cfg['models']['embedding_model_path'],cfg['runtime'].get('device','auto'),bool(cfg['retrieval'].get('normalize_embeddings',True)))
    q_emb=embedder.encode([r['question'] for r in rows],batch_size=int(cfg['runtime'].get('batch_size',64)))
    scores,indices=search_index(index,q_emb,int(cfg['retrieval'].get('faiss_search_k',80)),bool(cfg['retrieval'].get('normalize_embeddings',True)))
    out=[]
    for i,row in enumerate(tqdm(rows, desc='Collecting retrieved evidence')):
        item=collect_retrieved_answers(row['question_id'],indices[i].tolist(),scores[i].tolist(),[str(x) for x in meta['qids']],qid_to_answers,bool(cfg['retrieval'].get('remove_same_question_id',True)),int(cfg['retrieval']['top_k_retrieval']))
        item['question']=row['question']; item['references']=row['answers']; out.append(item)
    ensure_dir(Path(cfg['outputs']['retrieval_results']).parent); write_jsonl(cfg['outputs']['retrieval_results'],out); print('Saved retrieval results')
if __name__=='__main__': main()
