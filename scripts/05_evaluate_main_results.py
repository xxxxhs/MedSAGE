from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pathlib import Path
from medsage.config import load_config, ensure_dir
from medsage.io import read_jsonl, write_jsonl, save_table
from medsage.metrics import evaluate_generation_rows, summarize_metrics
from medsage.bertscore_eval import add_bertscore_best_match

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/default.yaml'); p.add_argument('--paths',default='configs/paths.yaml'); p.add_argument('--model',default=None); p.add_argument('--add-bertscore',action='store_true'); args=p.parse_args(); cfg=load_config(args.config,args.paths)
    models=[args.model] if args.model else cfg['models']['generation_models']; schemes=['base','rag','rag_avg','rag_mmr','medsage']; rows_sum=[]; eval_dir=Path(cfg['outputs']['evaluation_dir']); ensure_dir(eval_dir)
    for model in models:
        for scheme in schemes:
            path=Path(cfg['outputs']['generation_dir'])/model/f'{scheme}.jsonl'
            if not path.exists(): print(f'[Skip] {path}'); continue
            rows=read_jsonl(path); scored=evaluate_generation_rows(rows)
            if args.add_bertscore: scored=add_bertscore_best_match(scored, cfg['models'].get('bertscore_model_path'))
            out=eval_dir/model/f'{scheme}_scored.jsonl'; ensure_dir(out.parent); write_jsonl(out,scored)
            keys=['bleu2','rouge_l','meteor','cider','distinct1_corpus','distinct2_corpus'] + (['bertscore_f1'] if args.add_bertscore else [])
            rows_sum.append({'model':model,'scheme':scheme,**summarize_metrics(scored,keys),'n':len(scored)})
    save_table(eval_dir/'main_summary.csv', rows_sum); print('Saved main summary')
if __name__=='__main__': main()
