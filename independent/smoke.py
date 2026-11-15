"""Checks finite outputs and full-prefill vs cached decoding consistency.
This is not an ASR quality test or an oracle comparison.
"""
import argparse, json, time
import numpy as np
import mlx.core as mx
from model_source import open_model
from decoder import Decoder

ap = argparse.ArgumentParser()
ap.add_argument('pack')
ap.add_argument('--float-activations', action='store_true')
a = ap.parse_args()
t = time.perf_counter()
model = Decoder(open_model(a.pack), activation_int8=not a.float_activations)
loaded = time.perf_counter()
ids = [4, 17, 18]
h = model(ids); logits = model.logits(h[-1:]); mx.eval(logits)
full = np.array(h)
model.reset()
steps = []
for token in ids: steps.append(np.array(model([token])))
cached = np.concatenate(steps)
error = float(np.max(np.abs(full - cached)))
assert np.isfinite(full).all() and np.isfinite(np.array(logits)).all()
print(json.dumps({'layers': len(model.blocks), 'hidden_shape': list(h.shape), 'logits_shape': list(logits.shape), 'load_seconds': loaded-t, 'test_seconds': time.perf_counter()-loaded, 'prefill_cached_max_abs_error': error, 'top5_ids': np.array(mx.argsort(logits[0])[-5:]).tolist(), 'status': 'experimental; no vendor numerical comparison; decoder-only smoke test'}, indent=2))
# Batched and cached float32 matmul may accumulate differently; report, don't hide.
if error > 0.02: raise SystemExit('Cache consistency check failed')
