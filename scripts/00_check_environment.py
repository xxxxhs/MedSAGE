from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def main():
    for m in ['numpy','pandas','scipy','sklearn','yaml','tqdm','sentence_transformers','faiss','torch','transformers']:
        try:
            mod=__import__(m); print(f'[OK] {m}: {getattr(mod,"__version__","unknown")}')
        except Exception as e: print(f'[FAIL] {m}: {e}')
if __name__ == '__main__': main()
