from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import subprocess, sys

def run(cmd): print('\n[RUN]',' '.join(cmd)); subprocess.run(cmd,check=True)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/default.yaml'); p.add_argument('--paths',default='configs/paths.yaml'); p.add_argument('--model',default=None); p.add_argument('--dry-run',action='store_true'); args=p.parse_args(); base=[sys.executable]
    run(base+['scripts/01_prepare_cmedqa2.py','--config',args.config,'--paths',args.paths]); run(base+['scripts/02_build_faiss_index.py','--config',args.config,'--paths',args.paths]); run(base+['scripts/03_retrieve_evidence.py','--config',args.config,'--paths',args.paths])
    for scheme in ['base','rag','rag_avg','rag_mmr','medsage']:
        cmd=base+['scripts/04_run_generation.py','--config',args.config,'--paths',args.paths,'--scheme',scheme]
        if args.model: cmd += ['--model',args.model]
        if args.dry_run: cmd.append('--dry-run')
        run(cmd)
    cmd=base+['scripts/05_evaluate_main_results.py','--config',args.config,'--paths',args.paths]
    if args.model: cmd += ['--model',args.model]
    run(cmd); run(base+['scripts/06_evaluate_diversity.py','--config',args.config,'--paths',args.paths]); run(base+['scripts/07_evaluate_terminology_categories.py','--config',args.config,'--paths',args.paths]); run(base+['scripts/10_entity_consistency_analysis.py','--config',args.config,'--paths',args.paths]); run(base+['scripts/11_statistical_tests.py','--config',args.config,'--paths',args.paths]); run(base+['scripts/12_make_tables.py','--config',args.config,'--paths',args.paths])
if __name__=='__main__': main()
