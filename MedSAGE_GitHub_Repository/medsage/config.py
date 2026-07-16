from pathlib import Path
from typing import Any, Dict
import yaml

def load_yaml(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'YAML file not found: {path}')
    with path.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}

def deep_update(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out

def load_config(config_path, paths_path=None):
    cfg = load_yaml(config_path)
    if paths_path:
        cfg = deep_update(cfg, load_yaml(paths_path))
    return cfg

def ensure_dir(path):
    p = Path(path); p.mkdir(parents=True, exist_ok=True); return p
