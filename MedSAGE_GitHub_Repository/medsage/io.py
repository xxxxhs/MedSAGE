from pathlib import Path
import json
import pandas as pd

def read_csv_auto(path):
    last = None
    for enc in ['utf-8-sig','utf-8','gb18030','gbk']:
        try: return pd.read_csv(path, encoding=enc)
        except Exception as e: last=e
    raise RuntimeError(f'Failed to read CSV {path}: {last}')

def write_jsonl(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False)+'\n')

def read_jsonl(path):
    rows=[]
    with Path(path).open('r', encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if line: rows.append(json.loads(line))
    return rows

def write_json(path, obj, indent=2):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f: json.dump(obj, f, ensure_ascii=False, indent=indent)

def read_json(path):
    with Path(path).open('r', encoding='utf-8') as f: return json.load(f)

def save_table(path, rows):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    df=pd.DataFrame(rows)
    if path.suffix.lower()=='.xlsx': df.to_excel(path, index=False)
    else: df.to_csv(path, index=False, encoding='utf-8-sig')
