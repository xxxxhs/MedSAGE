from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pathlib import Path
import pandas as pd
from medsage.config import load_config

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/default.yaml'); p.add_argument('--paths',default='configs/paths.yaml'); args=p.parse_args(); cfg=load_config(args.config,args.paths); tables=Path(cfg['outputs']['tables_dir']); tables.mkdir(parents=True,exist_ok=True)
    for csv_path in Path(cfg['outputs']['evaluation_dir']).rglob('*.csv'):
        try: pd.read_csv(csv_path).to_excel(tables/(csv_path.stem+'.xlsx'),index=False)
        except Exception as e: print(f'[Skip] {csv_path}: {e}')
if __name__=='__main__': main()
