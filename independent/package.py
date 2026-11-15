"""Build a local deployment folder. Generated weights are never uploaded."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile
from model_source import SafeModel

ROOT = Path(__file__).resolve().parent
RUNTIME = ['decoder.py', 'encoder.py', 'features.py', 'model_source.py',
           'quantization.py', 'text.py', 'transcribe.py', 'web_server.py',
           'offline.sb', 'requirements.txt']


def package(model_path, output):
    source = SafeModel(model_path)
    # Verify every exported file, including currently unused streaming weights.
    for name in source.config['files']:
        source.checked_path(name)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.deploy-', dir=output.parent) as tmp:
        target = Path(tmp)
        runtime = target / 'independent'
        runtime.mkdir()
        for name in RUNTIME:
            shutil.copy2(ROOT / name, runtime / name)
        (target / 'web').mkdir()
        shutil.copy2(ROOT.parent / 'web/index.html', target / 'web/index.html')
        shutil.copy2(ROOT.parent / 'LICENSE', target / 'LICENSE')
        shutil.copy2(ROOT / 'DEPLOYMENT.md', target / 'README.md')
        model = runtime / 'exported-model'
        model.mkdir()
        for name in source.config['files']:
            shutil.copy2(source.checked_path(name), model / name)
        shutil.copy2(source.root / 'model.json', model / 'model.json')
        (target / '.gitignore').write_text('*\n')
        # Deliberately no .venv, extractor, FLUTE reader, app binary or private library.
        target.rename(output)
    return {'directory': str(output), 'model_components': len(source.config['models']),
            'bytes': sum(p.stat().st_size for p in output.rglob('*') if p.is_file())}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', type=Path, default=ROOT / 'exported-model')
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    print(json.dumps(package(a.model, a.output), indent=2))
