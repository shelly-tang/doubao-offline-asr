"""Small format/integrity tests, no proprietary weights or GPU required."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from safetensors.numpy import save_file
from model_source import SafeModel, FORMAT, VERSION, sha256


class SafeModelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.tensors = {'scalar': np.array(1e-5, dtype=np.float32),
                        'signed': np.array([[-128, -1, 0, 127]], dtype=np.int8)}
        save_file(self.tensors, str(self.root/'test.safetensors'))
        (self.root/'test.graph.json').write_text('[]')
        (self.root/'vocab.txt').write_text('example\n0\n')
        self.config = {'format': FORMAT, 'version': VERSION,
                       'quantization': 'signed-int4-k8-low4-high4',
                       'models': {'test': {'weights': 'test.safetensors', 'graph': 'test.graph.json'}},
                       'assets': {'vocab': 'vocab.txt'},
                       'files': {p.name: {'bytes': p.stat().st_size, 'sha256': sha256(p)} for p in self.root.iterdir()}}
        self.write_config()

    def write_config(self):
        (self.root/'model.json').write_text(json.dumps(self.config))

    def test_lossless_load_without_pack(self):
        with patch('weights.Pack', side_effect=AssertionError('FLUTE was used')):
            source = SafeModel(self.root)
            model = source.model('test')
            for key, expected in self.tensors.items():
                actual = model.tensors[key]
                self.assertEqual(actual.shape, expected.shape)
                self.assertEqual(actual.dtype, expected.dtype)
                self.assertEqual(actual.tobytes(), expected.tobytes())
            self.assertEqual(source.read('vocab'), b'example\n0\n')

    def test_corruption_rejected(self):
        (self.root/'vocab.txt').write_text('changed\n0\n')
        with self.assertRaisesRegex(ValueError, 'integrity'):
            SafeModel(self.root).read('vocab')

    def test_missing_asset_rejected(self):
        (self.root/'vocab.txt').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing'):
            SafeModel(self.root).read('vocab')

    def test_unknown_version_rejected(self):
        self.config['version'] = 99
        self.write_config()
        with self.assertRaisesRegex(ValueError, 'version'):
            SafeModel(self.root)

    def test_external_path_rejected(self):
        self.config['assets']['vocab'] = '../outside.txt'
        self.write_config()
        with self.assertRaisesRegex(ValueError, 'invalid'):
            SafeModel(self.root).read('vocab')


if __name__ == '__main__':
    unittest.main()
