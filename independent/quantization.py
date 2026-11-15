"""INT4 tensor layout shared by both local model formats; no container parser."""
import numpy as np


def unpack_q4(packed):
    """Signed K/2 x N storage: four low nibbles, then four high nibbles."""
    if packed.dtype != np.int8 or packed.ndim != 2 or packed.shape[0] % 4:
        raise ValueError('Expected int8 matrix with packed K divisible by four')
    p = packed.reshape(-1, 4, packed.shape[1]).astype(np.int16)
    low = ((p & 15) ^ 8) - 8
    high = p >> 4
    return np.concatenate((low, high), axis=1).reshape(-1, packed.shape[1]).astype(np.int8)
