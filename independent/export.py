"""Losslessly export local weights and required assets; never upload anything."""
import argparse
import json
import tempfile
from pathlib import Path
import numpy as np
from safetensors.numpy import save_file, load_file
from model_source import FORMAT, VERSION, sha256, SafeModel
from weights import Pack

MODELS = {
    'encoder': 'asr/models/non_s_encoder.opt.stream.pthlite',
    'decoder': 'asr/models/non_s_llm_decoder.pthlite',
    'embedding': 'asr/models/non_s_llm_embedding_logit.pthlite',
}
EXTRA_MODELS = {
    'stream_encoder': 'asr/models/encoder.opt.stream.pthlite',
    'stream_decoder': 'asr/models/llm_decoder.pthlite',
    'stream_embedding': 'asr/models/llm_decoder_embeding_logit.pthlite',
    'stream_position': 'asr/models/llm_decoder_embeding_wpe.pthlite',
    'draft': 'asr/models/non_s_draft.pthlite',
    'vad': 'vad/encoder/vad_v4.pth.pthlite',
}
ASSETS = {
    'config.yaml': 'config.yaml',
    'vocab.txt': 'asr/token/vocab.txt',
    'merges.txt': 'asr/token/merges.txt',
    'add_special_tokens.txt': 'asr/token/add_special_tokens.txt',
    'cmvn.txt': 'asr/feature_extractor/cmvn.txt',
    'fbank.conf': 'asr/feature_extractor/fbank.conf',
}


def export(pack_path, output, scope="all"):
    if scope not in ("all", "nonstream"):
        raise ValueError("scope must be all or nonstream")
    selected_models = {**MODELS, **EXTRA_MODELS} if scope == "all" else MODELS
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    # An interrupted export never leaves a partially usable final directory.
    with tempfile.TemporaryDirectory(prefix='.export-', dir=output.parent) as temporary:
        target = Path(temporary)
        (target / '.gitignore').write_text('*\n')
        source = Pack(pack_path)
        config = {'format': FORMAT, 'version': VERSION,
                  'quantization': 'signed-int4-k8-low4-high4',
                  'source_sha256': sha256(pack_path), 'scope': scope,
                  'models': {}, 'assets': {}, 'files': {}}
        tensor_count = 0
        try:
            for label, name in selected_models.items():
                model = source.model(name)
                weights, graph = label + '.safetensors', label + '.graph.json'
                # Copies retain original shapes (including scalars), dtypes and bits.
                tensors = {k: np.array(v, copy=True, order='C') for k, v in model.tensors.items()}
                save_file(tensors, str(target / weights), metadata={'format': FORMAT})
                recovered = load_file(str(target / weights))
                if tensors.keys() != recovered.keys():
                    raise ValueError('Exported tensor names differ')
                for k, a in tensors.items():
                    b = recovered[k]
                    if a.shape != b.shape or a.dtype != b.dtype or a.tobytes() != b.tobytes():
                        raise ValueError(f'Export verification failed: {name}/{k}')
                (target / graph).write_text(json.dumps(model.nodes, ensure_ascii=False, indent=2))
                config['models'][name] = {'weights': weights, 'graph': graph}
                tensor_count += len(tensors)
                print(f'Verified {label}: {len(tensors)} tensors', flush=True)
                del model, tensors, recovered
            for filename, name in ASSETS.items():
                (target / filename).write_bytes(source.read(name))
                config['assets'][name] = filename
        finally:
            source.data.close()
            source.file.close()
        for path in sorted(target.iterdir()):
            config['files'][path.name] = {'bytes': path.stat().st_size, 'sha256': sha256(path)}
        config['verified_tensor_count'] = tensor_count
        (target / 'model.json').write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n')
        standalone = SafeModel(target)
        for name in config['assets']:
            standalone.read(name)
        # All files and their manifest are complete before publishing locally.
        target.rename(output)
    return {'output': str(output), 'tensors': tensor_count,
            'bytes': sum(p.stat().st_size for p in output.iterdir())}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pack', type=Path, help='Existing local plaintext FLUTE v1 pack; omit to extract installed weights temporarily')
    ap.add_argument('--output', type=Path, default=Path(__file__).with_name('exported-model'))
    ap.add_argument('--scope', choices=['all', 'nonstream'], default='all', help='Export all nine components (default), or only the three currently used for nonstream inference')
    args = ap.parse_args()
    if args.output.exists():
        ap.error('Output exists; choose a new directory (no overwrite)')
    if args.pack:
        result = export(args.pack, args.output, args.scope)
    else:
        from extract import extract, APP, MODEL
        with tempfile.TemporaryDirectory(prefix='doubao-local-extract-') as tmp:
            pack = extract(APP, MODEL, Path(tmp) / 'plain.model')
            result = export(pack, args.output, args.scope)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
