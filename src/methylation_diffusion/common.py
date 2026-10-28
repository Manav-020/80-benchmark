import hashlib
import json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]

def config(name):
    return yaml.safe_load((ROOT / 'configs' / f'{name}.yaml').read_text())

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str, allow_nan=False), encoding='utf-8')
    tmp.replace(path)

def data_root():
    p = ROOT / 'configs/paths.local.yaml'
    cfg = yaml.safe_load(p.read_text()) if p.exists() else config('paths.example')
    return (ROOT / cfg['data_root']).resolve()

def fingerprint():
    files = sorted((ROOT / 'src').rglob('*.py')) + sorted((ROOT / 'configs').glob('*.yaml'))
    files += [ROOT / 'outputs/audits/sources.json', ROOT / 'outputs/splits/splits.json']
    return hashlib.sha256(''.join(str(p.relative_to(ROOT)) + digest(p) for p in files if p.exists()).encode()).hexdigest()
