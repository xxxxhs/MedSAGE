from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pathlib import Path
from collections import defaultdict
from medsage.config import load_config
from medsage.io import read_jsonl, save_table
from medsage.category import load_lexicon, assign_categories
from medsage.metrics import evaluate_generation_rows, summarize_metrics

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/default.yaml'); p.add_argument('--paths',default='configs/paths.yaml'); p.add_argument('--lexicon',default='configs/medical_lexicon_zh.yaml'); args=p.parse_args(); cfg=load_config(args.config,args.paths); lex=load_lexicon(args.lexicon); out=[]
    for model_dir in Path(cfg['outputs']['generation_dir']).glob('*'):
        if not model_dir.is_dir(): continue
        for f in model_dir.glob('*.jsonl'):
            cats=defaultdict(list)
            for r in read_jsonl(f):
                for c in assign_categories(r.get('question',''),lex): cats[c].append(r)
            for c,rows in cats.items(): out.append({'model':model_dir.name,'scheme':f.stem,'category':c,**summarize_metrics(evaluate_generation_rows(rows),['bleu2','rouge_l','meteor','cider','distinct1_corpus','distinct2_corpus']),'n':len(rows)})
    save_table(Path(cfg['outputs']['evaluation_dir'])/'terminology_category_summary.csv', out)
if __name__=='__main__': main()
