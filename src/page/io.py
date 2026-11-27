"""Small file helpers used by PAGE asset builders."""
from hashlib import sha256 as _sha256
from pathlib import Path
import json


def sha256(path):
    digest = _sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
