from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pathlib import Path
from medsage.config import load_config
from medsage.io import read_jsonl, save_table
from medsage.statistics import compare_methods

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/default.yaml'); p.add_argument('--paths',default='configs/paths.yaml'); p.add_argument('--model',default=None); p.add_argument('--target',default='medsage'); args=p.parse_args(); cfg=load_config(args.config,args.paths); model=args.model or cfg['models']['default_generation_model']; eval_dir=Path(cfg['outputs']['evaluation_dir'])/model
    method_rows={}
    for scheme in ['base','rag','rag_avg','rag_mmr',args.target]:
        path=eval_dir/f'{scheme}_scored.jsonl'
        if path.exists(): method_rows[scheme]={str(r['question_id']):r for r in read_jsonl(path)}
    if args.target in method_rows:
        recs=compare_methods(method_rows,args.target,[s for s in ['base','rag','rag_avg','rag_mmr'] if s in method_rows],['bleu2','rouge_l','meteor','cider'])
        save_table(Path(cfg['outputs']['evaluation_dir'])/f'{model}_{args.target}_wilcoxon_holm.csv', recs)
if __name__=='__main__': main()
