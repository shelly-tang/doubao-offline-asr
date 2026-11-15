"""Replay a local PCM16/16k mono WAV through the recording HTTP endpoints."""
import argparse
import json
import time
import urllib.request
import wave


def benchmark(url, path, realtime=False, chunk_samples=2048):
    session = ''
    sequence = 0
    def post(route, body=b''):
        req = urllib.request.Request(url+route, data=body, headers={
            'Origin':url, 'Content-Type':'application/octet-stream',
            'X-Session':session, 'X-Sequence':str(sequence)})
        with urllib.request.urlopen(req, timeout=120) as response:
            return json.load(response)
    with wave.open(str(path)) as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1,2,16000):
            raise ValueError('Expected mono PCM16 16000 Hz WAV')
        data = wav.readframes(wav.getnframes())
    session = post('/start')['session']
    started = time.perf_counter()
    first = None
    events = []
    requests = []
    try:
        for offset in range(0,len(data),chunk_samples*2):
            chunk = data[offset:offset+chunk_samples*2]
            if realtime:
                due = started+(offset+len(chunk))/32000
                time.sleep(max(0,due-time.perf_counter()))
            before = time.perf_counter()
            result = post('/feed', chunk)
            requests.append(time.perf_counter()-before)
            sequence += 1
            events.extend(result['events'])
            if result['text'] and first is None:
                first = time.perf_counter()-started
        before = time.perf_counter()
        result = post('/finish')
        tail = time.perf_counter()-before
        events.extend(result['events'])
    except BaseException:
        post('/cancel')
        raise
    import statistics
    result.pop('events')
    result.update({'realtime_replay':realtime, 'first_text_wall_seconds':first,
                   'tail_seconds':tail, 'request_median_ms':statistics.median(requests)*1000,
                   'block_median_ms':statistics.median(e['compute_seconds'] for e in events)*1000,
                   'block_max_ms':max(e['compute_seconds'] for e in events)*1000,
                   'rtf':result['inference']/result['duration'], 'events':events})
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('wav')
    ap.add_argument('--url',default='http://localhost:8766')
    ap.add_argument('--realtime',action='store_true')
    ap.add_argument('--chunk-samples', type=int, default=2048, choices=range(1,32769))
    a = ap.parse_args()
    print(json.dumps(benchmark(a.url,a.wav,a.realtime,a.chunk_samples),ensure_ascii=False,indent=2))
