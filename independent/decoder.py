"""Experimental MLX Qwen-shaped decoder. No vendor dynamic library imports.

This is a reconstruction, not yet numerically validated against Panther.
FP32 weights/activations intentionally isolate layout errors from FP16 rounding.
"""
import numpy as np
import mlx.core as mx
from weights import unpack_q4


def rms(x, weight, eps=1e-5):
    return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps) * weight


class Linear:
    def __init__(self, packed, scale, activation_int8=True):
        self.weight = mx.array(unpack_q4(packed).astype(np.float32))
        self.scale = mx.array(scale)
        self.activation_int8 = activation_int8
        mx.eval(self.weight, self.scale)

    def __call__(self, x):
        if self.activation_int8:
            scale = mx.maximum(mx.max(mx.abs(x), axis=-1, keepdims=True) / 127, 1e-9)
            x = mx.clip(mx.floor(x / scale + 0.5), -127, 127) * scale
        return (x * self.scale) @ self.weight


class Block:
    def __init__(self, tensors, node, activation_int8=True):
        t = [tensors.get(s) for s in node['inputs']]
        self.ln1, self.ln2 = mx.array(t[3]), mx.array(t[12])
        self.q, self.k, self.v, self.o, self.gate, self.up, self.down = [
            Linear(t[i], t[i + 1], activation_int8) for i in (4, 6, 8, 10, 13, 15, 17)]
        self.rope = mx.array(t[25])
        self.knorm, self.qnorm = mx.array(t[26]), mx.array(t[27])
        self.head_dim = t[26].size
        self.heads = node['attrs']['num_heads']
        self.kv_heads = t[6].shape[1] // self.head_dim
        self.eps = node['attrs']['ln_eps']
        self.cache = None

    def rotate(self, x, offset):
        # Serialized table is [cos/sin, position, head_dim], unlike GPU staging.
        count = x.shape[0]
        positions = mx.arange(offset, offset + count) % self.rope.shape[1]
        half = self.head_dim // 2
        c = self.rope[0, positions, :half][:, None, :]
        s = self.rope[1, positions, :half][:, None, :]
        a, b = x[..., :half], x[..., half:]
        return mx.concatenate((a * c - b * s, a * s + b * c), axis=-1)

    def __call__(self, x):
        n = x.shape[0]
        offset = 0 if self.cache is None else self.cache[0].shape[1]
        y = rms(x, self.ln1, self.eps)
        q = rms(self.q(y).reshape(n, self.heads, self.head_dim), self.qnorm, self.eps)
        k = rms(self.k(y).reshape(n, self.kv_heads, self.head_dim), self.knorm, self.eps)
        q = self.rotate(q, offset).transpose(1, 0, 2)
        k = self.rotate(k, offset).transpose(1, 0, 2)
        v = self.v(y).reshape(n, self.kv_heads, self.head_dim).transpose(1, 0, 2)
        if self.cache is not None:
            k = mx.concatenate((self.cache[0], k), axis=1)
            v = mx.concatenate((self.cache[1], v), axis=1)
        self.cache = k, v
        group = self.heads // self.kv_heads
        scores = (q @ mx.repeat(k, group, axis=0).transpose(0, 2, 1)) / self.head_dim ** 0.5
        mask = mx.arange(offset + n)[None, :] <= mx.arange(offset, offset + n)[:, None]
        probs = mx.softmax(mx.where(mask, scores, -float('inf')), axis=-1)
        attended = (probs @ mx.repeat(v, group, axis=0)).transpose(1, 0, 2).reshape(n, -1)
        x = x + self.o(attended)
        y = rms(x, self.ln2, self.eps)
        gate = self.gate(y)
        return x + self.down((gate * mx.sigmoid(gate)) * self.up(y))


class Decoder:
    def __init__(self, pack, activation_int8=True):
        m = pack.model('asr/models/non_s_llm_decoder.pthlite')
        self.blocks = [Block(m.tensors, n, activation_int8) for n in m.nodes
                       if n['op'] == 'W4A8ImeAsrQwen3TransformerBlock']
        self.norm = mx.array(m.tensors['ln_f.weight'])
        self.eps = float(m.tensors['/ln_f/Constant_2_output_0'])
        m = pack.model('asr/models/non_s_llm_embedding_logit.pthlite')
        a, s, z = [m.tensors[name] for name in m.nodes[0]['inputs'][:3]]
        if z.shape != (1,) or float(z[0]) != 1.0:
            raise ValueError('Unsupported embedding multiplier; only observed value 1 supported')
        self.embedding = mx.array(unpack_q4(a).astype(np.float32) * s[None, :])
        mx.eval(self.embedding)

    def reset(self):
        for block in self.blocks: block.cache = None

    def __call__(self, ids=None, embeddings=None):
        x = self.embedding[:, mx.array(ids)].T if embeddings is None else embeddings
        for block in self.blocks:
            x = block(x)
            mx.eval(x, block.cache)
        return rms(x, self.norm, self.eps)

    def logits(self, hidden):
        return hidden @ self.embedding
