from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Creates temporary configs for K, tau, and C/R/D weight sensitivity runs.
from pathlib import Path
from copy import deepcopy
import subprocess, sys, yaml
from medsage.config import load_config, ensure_dir

def dump(cfg,path):
    with open(path,'w',encoding='utf-8') as f: yaml.safe_dump(cfg,f,allow_unicode=True)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/default.yaml'); p.add_argument('--paths',default='configs/paths.yaml'); p.add_argument('--model',default=None); p.add_argument('--dry-run',action='store_true'); args=p.parse_args(); base=load_config(args.config,args.paths); temp=Path(base['outputs']['root'])/'temp_configs'; ensure_dir(temp); runs=[]
    for k in base['sensitivity']['k_values']:
        cfg=deepcopy(base); cfg['medsage']['sentence_top_k']=int(k); cfg['outputs']['generation_dir']=str(Path(base['outputs']['generation_dir'])/'sensitivity'/f'k_{k}'); path=temp/f'k_{k}.yaml'; dump(cfg,path); runs.append(path)
    for tau in base['sensitivity']['tau_values']:
        cfg=deepcopy(base); cfg['medsage']['similarity_threshold_tau']=float(tau); cfg['outputs']['generation_dir']=str(Path(base['outputs']['generation_dir'])/'sensitivity'/f'tau_{tau}'); path=temp/f'tau_{tau}.yaml'; dump(cfg,path); runs.append(path)
    for name,w in base['sensitivity']['weight_settings'].items():
        cfg=deepcopy(base); cfg['medsage']['weights']={'centrality':w[0],'relevance':w[1],'diversity':w[2]}; cfg['outputs']['generation_dir']=str(Path(base['outputs']['generation_dir'])/'sensitivity'/f'w_{name}'); path=temp/f'w_{name}.yaml'; dump(cfg,path); runs.append(path)
    for cfg_path in runs:
        cmd=[sys.executable,'scripts/04_run_generation.py','--config',str(cfg_path),'--scheme','medsage']
        if args.model: cmd += ['--model',args.model]
        if args.dry_run: cmd.append('--dry-run')
        subprocess.run(cmd,check=True)
if __name__=='__main__': main()
