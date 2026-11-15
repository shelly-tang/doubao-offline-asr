"""Compare real audio before/after export, denying FLUTE reads after export.
Run on macOS with a usable Metal GPU. Audio and embeddings stay local.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np

ROOT = Path(__file__).resolve().parent


def verify(pack, exported, wavs):
    results = []
    profile = (ROOT / 'offline.sb').read_text()
    with tempfile.TemporaryDirectory(prefix='verify-export-') as tmp:
        tmp = Path(tmp)
        before_profile = tmp / 'before.sb'
        after_profile = tmp / 'after.sb'
        before_profile.write_text(profile)
        # Deny all original/repacked .model files, not merely the chosen input.
        after_profile.write_text(profile + '\n(deny file-read* (regex #".*[.]model$"))\n')
        for wav in wavs:
            outputs, embeddings = [], []
            for i, (model, sb) in enumerate(((pack, before_profile), (exported, after_profile))):
                embedding = tmp / f'embeddings-{i}.npy'
                cmd = ['sandbox-exec', '-D', 'DOUBAO_DATA=' + str(Path.home() / 'Library/Application Support/DoubaoIme'),
                       '-D', 'PRIVATE_FRAMEWORKS=' + str(ROOT.parent / 'Frameworks'), '-f', str(sb),
                       sys.executable, str(ROOT / 'transcribe.py'), str(model), str(wav),
                       '--save-audio-embeddings', str(embedding)]
                run = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                if run.returncode:
                    raise RuntimeError(run.stderr)
                outputs.append(json.loads(run.stdout))
                embeddings.append(np.load(embedding))
            before, after = outputs
            for key in ('ids', 'raw_text', 'text', 'truncated'):
                if before[key] != after[key]:
                    raise AssertionError(f'{wav.name}: {key} differs')
            if before['truncated']:
                raise AssertionError(f'{wav.name}: output truncated; shorten audio')
            if not np.array_equal(*embeddings):
                raise AssertionError(f'{wav.name}: encoder outputs differ')
            result = {'sample': wav.name, 'text': after['text'], 'tokens': len(after['ids']),
                      'token_ids_equal': True, 'raw_text_equal': True,
                      'encoder_max_abs_error': float(np.max(np.abs(embeddings[0] - embeddings[1]))),
                      'flute_seconds': before['elapsed_seconds'], 'safetensors_seconds': after['elapsed_seconds'],
                      'safetensors_flute_reads_denied': True}
            print(json.dumps(result, ensure_ascii=False), flush=True)
            results.append(result)
    return results


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pack', type=Path, required=True)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('wav', type=Path, nargs='+')
    a = ap.parse_args()
    verify(a.pack.resolve(), a.model.resolve(), [p.resolve() for p in a.wav])
