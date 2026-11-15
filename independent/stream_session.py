"""Incremental PCM16 -> fbank -> cached streaming MLX encoder/decoder."""
import time
import numpy as np
import mlx.core as mx
from features import fbank
from text import tokenizer
from streaming import StreamEncoder, StreamDecoder


class StreamingSession:
    def __init__(self, source):
        self.source = source
        self.encoder = StreamEncoder(source)
        self.decoder = StreamDecoder(source)
        self.tokenizer = tokenizer(source)
        self.reset()

    def reset(self):
        self.encoder.reset()
        self.decoder.reset()
        self.pcm = np.empty(0, np.float32)
        self.ids = []
        self.samples = self.frames = 0
        self.compute = 0.0
        self.first_text_audio = None
        self.finished = False

    def feed(self, pcm, finish=False):
        if self.finished:
            raise ValueError('Session already finished')
        pcm = np.asarray(pcm, dtype=np.float32)
        if pcm.ndim != 1 or not np.isfinite(pcm).all():
            raise ValueError('Expected finite mono PCM')
        if self.samples + len(pcm) > 18 * 16000:
            raise ValueError('Recording limit: 18 seconds')
        self.samples += len(pcm)
        self.pcm = np.concatenate((self.pcm, pcm))
        if finish:
            # Stream config: finish_pad_wav_ms=960. Complete the last feature chunk.
            self.pcm = np.pad(self.pcm, (0, 15360))
            frames = max(0, (len(self.pcm)-400)//160+1)
            if frames % 16:
                self.pcm = np.pad(self.pcm, (0, (16-frames%16)*160))
        events = []
        while len(self.pcm) >= 2800:  # sixteen 25 ms windows with 10 ms hops
            start = time.perf_counter()
            features = fbank(self.pcm[:2800], self.source, pad_samples=0)
            self.pcm = self.pcm[2560:]
            h = self.decoder(embeddings=self.encoder(features))
            previous = len(self.ids)
            for _ in range(32):
                token = int(mx.argmax(self.decoder.logits(h[-1:])[0]).item())
                # Control tokens also occupy positions and must enter the KV cache.
                h = self.decoder([token])
                mx.eval(h)
                if token in (2, 4, 64, 65, 66, 67):
                    break
                self.ids.append(token)
            else:
                raise RuntimeError('Streaming decoder did not yield within 32 tokens')
            elapsed = time.perf_counter()-start
            self.compute += elapsed
            self.frames += 16
            if len(self.ids) > previous and self.first_text_audio is None:
                self.first_text_audio = self.frames/100 + .015
            events.append({'audio_seconds': self.frames/100+.015,
                           'compute_seconds': elapsed, 'text': self.text})
        self.finished = finish
        return {'text': self.text, 'duration': self.samples/16000,
                'inference': self.compute, 'first_text_audio_seconds': self.first_text_audio,
                'events': events, 'finished': finish}

    @property
    def text(self):
        # Numeric tokens are marked special in this vocabulary; keep them.
        return self.tokenizer.decode(self.ids, skip_special_tokens=False)
