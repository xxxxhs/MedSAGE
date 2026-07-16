from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pathlib import Path
from medsage.config import load_config
from medsage.io import read_jsonl, save_table
from medsage.metrics import distinct_n

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/default.yaml'); p.add_argument('--paths',default='configs/paths.yaml'); args=p.parse_args(); cfg=load_config(args.config,args.paths); out=[]
    for model_dir in Path(cfg['outputs']['generation_dir']).glob('*'):
        if not model_dir.is_dir(): continue
        for f in model_dir.glob('*.jsonl'):
            rows=read_jsonl(f); texts=[r.get('generated_answer','') for r in rows]; out.append({'model':model_dir.name,'scheme':f.stem,'distinct1':distinct_n(texts,1),'distinct2':distinct_n(texts,2),'n':len(texts)})
    save_table(Path(cfg['outputs']['evaluation_dir'])/'diversity_summary.csv',out)
if __name__=='__main__': main()
