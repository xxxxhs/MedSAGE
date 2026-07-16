from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pathlib import Path
from medsage.config import load_config
from medsage.io import read_jsonl, write_jsonl, save_table
from medsage.category import load_lexicon
from medsage.entity_consistency import entity_consistency_for_row, summarize_entity_consistency

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/default.yaml'); p.add_argument('--paths',default='configs/paths.yaml'); p.add_argument('--lexicon',default='configs/medical_lexicon_zh.yaml'); args=p.parse_args(); cfg=load_config(args.config,args.paths); lex=load_lexicon(args.lexicon); out=[]; out_dir=Path(cfg['outputs']['evaluation_dir'])/'entity_consistency'; out_dir.mkdir(parents=True,exist_ok=True)
    for model_dir in Path(cfg['outputs']['generation_dir']).glob('*'):
        if not model_dir.is_dir(): continue
        for f in model_dir.glob('*.jsonl'):
            analyzed=[entity_consistency_for_row(r,lex) for r in read_jsonl(f)]; write_jsonl(out_dir/f'{model_dir.name}_{f.stem}_entity.jsonl', analyzed); out.append({'model':model_dir.name,'scheme':f.stem,**summarize_entity_consistency(analyzed),'n':len(analyzed)})
    save_table(out_dir/'entity_consistency_summary.csv', out)
if __name__=='__main__': main()
