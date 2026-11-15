"""Persistent, network-denied inference worker; one JSON line per PCM request."""
import base64
import json
import sys
import time
import numpy as np
from model_source import SafeModel
from stream_session import StreamingSession

start = time.perf_counter()
session = StreamingSession(SafeModel(sys.argv[1]))
print(json.dumps({'ready': True, 'load': time.perf_counter()-start}), flush=True)
for line in sys.stdin:
    try:
        request = json.loads(line)
        if request.get('reset'):
            session.reset()
            result = {'reset': True}
        else:
            pcm = np.frombuffer(base64.b64decode(request.get('pcm', ''), validate=True), dtype='<i2')
            result = session.feed(pcm, request.get('finish', False))
        print(json.dumps(result, ensure_ascii=False), flush=True)
    except Exception as exc:
        print(json.dumps({'error': str(exc)}), flush=True)
