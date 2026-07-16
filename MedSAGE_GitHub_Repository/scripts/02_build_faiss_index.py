from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pathlib import Path
from medsage.config import load_config, ensure_dir
from medsage.data import load_cmedqa2
from medsage.embedding import SentenceEmbedder, save_embeddings
from medsage.io import write_json
from medsage.retrieval import build_faiss_index, save_faiss_index

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/default.yaml'); p.add_argument('--paths',default='configs/paths.yaml'); args=p.parse_args(); cfg=load_config(args.config,args.paths)
    qdf,_=load_cmedqa2(cfg['data']['question_file'],cfg['data']['answer_file'],cfg['data'].get('question_id_col','question_id'),cfg['data'].get('question_text_col','content'),cfg['data'].get('answer_question_id_col','question_id'),cfg['data'].get('answer_text_col','content'))
    qids=qdf[cfg['data'].get('question_id_col','question_id')].astype(str).tolist(); questions=qdf[cfg['data'].get('question_text_col','content')].astype(str).tolist()
    emb=SentenceEmbedder(cfg['models']['embedding_model_path'],cfg['runtime'].get('device','auto'),bool(cfg['retrieval'].get('normalize_embeddings',True))).encode(questions,batch_size=int(cfg['runtime'].get('batch_size',64)))
    save_embeddings(cfg['outputs']['question_embeddings'], emb); index=build_faiss_index(emb,bool(cfg['retrieval'].get('normalize_embeddings',True))); save_faiss_index(index,cfg['outputs']['faiss_index']); write_json(str(Path(cfg['outputs']['faiss_index']).with_suffix('.meta.json')), {'qids':qids,'questions':questions})
    print('Saved FAISS index and metadata')
if __name__=='__main__': main()
