"""Feature buffering must be independent of browser transport packet size."""
import unittest
import numpy as np
from features import fbank


class Source:
    def read(self, name):
        return ('0 1\n'*80).encode()


class FeatureTests(unittest.TestCase):
    def test_packet_boundaries(self):
        source = Source()
        audio = np.random.default_rng(42).normal(0,1000,16000).astype(np.float32)
        expected = fbank(audio,source,pad_samples=0)
        for size in (997,2048,4096):
            pending = np.empty(0,np.float32)
            chunks = []
            for i in range(0,len(audio),size):
                pending = np.concatenate((pending,audio[i:i+size]))
                while len(pending)>=2800:
                    chunks.append(fbank(pending[:2800],source,pad_samples=0))
                    pending = pending[2560:]
            actual = np.concatenate(chunks)
            np.testing.assert_allclose(actual,expected[:len(actual)],rtol=1e-6,atol=1e-6)

    def test_default_finish_padding_unchanged(self):
        audio = np.zeros(16000,np.float32)
        np.testing.assert_array_equal(fbank(audio,Source()),fbank(audio,Source(),pad_samples=2560))


if __name__=='__main__':unittest.main()
