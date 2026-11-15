"""Load a standalone safetensors directory or the legacy local FLUTE pack."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from safetensors.numpy import load_file

FORMAT = 'doubao-mlx-safetensors'
VERSION = 1


def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


class SafeModel:
    def __init__(self, path):
        self.root = Path(path).resolve()
        self.config = json.loads((self.root / 'model.json').read_text())
        if (self.config.get('format'), self.config.get('version')) != (FORMAT, VERSION):
            raise ValueError('Unsupported exported model format/version')
        if self.config.get('quantization') != 'signed-int4-k8-low4-high4':
            raise ValueError('Unsupported quantization layout')
        self.checked = set()

    def checked_path(self, name):
        path = (self.root / name).resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            raise ValueError(f'Missing or invalid model file: {name}')
        if name not in self.checked:
            expected = self.config['files'][name]
            if path.stat().st_size != expected['bytes'] or sha256(path) != expected['sha256']:
                raise ValueError(f'Model file integrity check failed: {name}')
            self.checked.add(name)
        return path

    def model(self, name):
        entry = self.config['models'][name]
        tensors = load_file(str(self.checked_path(entry['weights'])))
        nodes = json.loads(self.checked_path(entry['graph']).read_text())
        return SimpleNamespace(tensors=tensors, nodes=nodes)

    def read(self, name):
        return self.checked_path(self.config['assets'][name]).read_bytes()


def open_model(path):
    if Path(path).is_dir():
        return SafeModel(path)
    # Lazy import: standalone model loading does not invoke the FLUTE parser.
    from weights import Pack
    return Pack(path)
