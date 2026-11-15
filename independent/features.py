"""NumPy implementation of the observed Kaldi-style 80-bin frontend.
Numerical parity with the original frontend remains to be measured.
"""
import io
import wave
import numpy as np


def read_wav(path):
    with wave.open(str(path)) as f:
        if (f.getnchannels(),f.getsampwidth(),f.getframerate()) != (1,2,16000):
            raise ValueError('Expected mono PCM16 WAV at 16000 Hz')
        audio = np.frombuffer(f.readframes(f.getnframes()),dtype='<i2').astype(np.float32)
        if not 1 <= audio.size <= 18*16000:
            raise ValueError('Experimental path requires nonempty audio of at most 18 seconds')
        return audio


def fbank(audio, pack, pad_samples=2560):
    audio = np.pad(audio,(0,pad_samples))  # observed finish_pad_wav_ms=160
    frames = np.lib.stride_tricks.sliding_window_view(audio,400)[::160].copy()
    frames -= frames.mean(axis=-1,keepdims=True)
    frames[:,1:] -= .97*frames[:,:-1].copy()
    frames[:,0] *= .03
    frames *= np.hanning(400).astype(np.float32)**.85
    power = np.abs(np.fft.rfft(frames,n=512))**2
    mel = lambda hz: 1127*np.log1p(hz/700)
    edges = np.linspace(mel(20),mel(8000),82)
    frequencies = mel(np.arange(257)*16000/512)
    filters = np.maximum(0,np.minimum((frequencies[None,:]-edges[:-2,None])/(edges[1:-1]-edges[:-2])[:,None],(edges[2:,None]-frequencies[None,:])/(edges[2:]-edges[1:-1])[:,None]))
    logmel = np.log(np.maximum(power@filters.T,np.finfo(np.float32).eps))
    cmvn = np.loadtxt(io.BytesIO(pack.read('asr/feature_extractor/cmvn.txt')))
    return ((logmel-cmvn[:,0])*cmvn[:,1]).astype(np.float32)
