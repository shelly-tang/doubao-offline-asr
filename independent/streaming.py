"""Experimental cached streaming reconstruction. Recognition parity is NOT established.
See STREAMING.md for measured errors and limitations."""
import numpy as np
import mlx.core as mx
from encoder import layernorm, gelu, Linear
from quantization import unpack_q4

def quant(x, s):
    return mx.clip(mx.round(x / s), -128, 127) * s

class QuantLinear(Linear):

    def __init__(self, w, scale, bias, act):
        super().__init__(w, scale, bias)
        self.act = float(act.item())

    def __call__(self, x):
        return super().__call__(quant(x, self.act))

def norm(x, w, b):
    return layernorm(x, mx.concatenate((w, b)))

class StreamBlock:

    def __init__(self, t, node):
        a = [t.get(k) for k in node['inputs']]
        self.ln1 = mx.array(a[3])
        self.b1 = mx.array(a[4])
        self.ln2 = mx.array(a[15])
        self.b2 = mx.array(a[16])
        self.qkv, self.out, self.fc, self.proj = [QuantLinear(a[i], a[i + 1], a[i + 3], a[i - 1]) for i in (6, 11, 18, 23)]
        self.scales = [float(a[i].item()) for i in (28, 29, 30, 31)]
        self.cache = None

    def __call__(self, x):
        n = x.shape[0]
        offset = 0 if self.cache is None else self.cache[0].shape[1]
        q, k, v = [y.reshape(n, 16, 64).transpose(1, 0, 2) for y in mx.split(self.qkv(norm(x, self.ln1, self.b1)), 3, axis=-1)]
        q = quant(q, self.scales[0])
        k = quant(k, self.scales[1])
        v = quant(v, self.scales[3])
        if self.cache is not None:
            k = mx.concatenate((self.cache[0], k), axis=1)
            v = mx.concatenate((self.cache[1], v), axis=1)
        self.cache = (k, v)
        mask = mx.arange(offset + n)[None, :] <= mx.arange(offset, offset + n)[:, None]
        prob = mx.softmax(mx.where(mask, q @ k.transpose(0, 2, 1) / 8, -float('inf')), axis=-1)
        prob = quant(prob, self.scales[2])
        x = x + self.out((prob @ v).transpose(1, 0, 2).reshape(n, 1024))
        return x + self.proj(gelu(self.fc(norm(x, self.ln2, self.b2))))

class StreamDecoder:

    def __init__(self, source):
        m = source.model('asr/models/llm_decoder.pthlite')
        t = m.tensors
        self.blocks = [StreamBlock(t, n) for n in m.nodes if n['op'] == 'W4A8GPT2TransformerBlock']
        self.ln = mx.array(t['model.transformer.ln_f.weight'])
        self.bias = mx.array(t['model.transformer.ln_f.bias'])
        m = source.model('asr/models/llm_decoder_embeding_logit.pthlite')
        a = [m.tensors[k] for k in m.nodes[0]['inputs'][:5]]
        self.embedding = mx.array(unpack_q4(a[0]).astype(np.float32) * a[1][None, :])
        mx.eval(self.embedding)
        m = source.model('asr/models/llm_decoder_embeding_wpe.pthlite')
        a = [m.tensors[k] for k in m.nodes[0]['inputs'][:2]]
        self.position = mx.array(a[0].astype(np.float32) * a[1][:, None])
        self.reset()

    def reset(self):
        self.pos = 0
        for b in self.blocks:
            b.cache = None

    def __call__(self, ids=None, embeddings=None):
        x = self.embedding[:, mx.array(ids)].T if embeddings is None else embeddings
        n = x.shape[0]
        if self.pos + n > 2048:
            raise ValueError('Decoder context limit')
        x = x + self.position[self.pos:self.pos + n]
        self.pos += n
        for b in self.blocks:
            x = b(x)
            mx.eval(x, b.cache)
        return norm(x, self.ln, self.bias)

    def logits(self, h):
        return quant(h, 0.12814985) @ self.embedding

class StreamConformer:

    def __init__(self, t, node):
        a = [t.get(k) for k in node['inputs']]
        self.norms = [mx.array(a[i]) for i in (2, 11, 20, 27, 32, 41)]
        self.ff1a, self.ff1b, self.qkv, self.out, self.pw1, self.pw2, self.ff2a, self.ff2b = [QuantLinear(a[i], a[i + 1], a[i + 3], a[j]) for i, j in zip((3, 7, 12, 16, 21, 28, 33, 37), range(42, 50))]
        self.scales = [float(a[i].item()) for i in (50, 51, 52, 53)]
        self.dw = mx.array(a[25])
        self.db = mx.array(a[26])
        self.rope = mx.array(a[1])
        self.reset()

    def reset(self):
        self.cache = None
        self.conv = mx.zeros((2, 880))
        self.pos = 0

    def rotate(self, x):
        n = x.shape[0]
        p = mx.arange(self.pos, self.pos + n) % self.rope.shape[1]
        c, s = (self.rope[0, p, None, :], self.rope[1, p, None, :])
        pairs = x.reshape(n, 8, 55, 2)
        r = mx.stack((-pairs[..., 1], pairs[..., 0]), axis=-1).reshape(x.shape)
        return x * c + r * s

    def __call__(self, x):
        n = x.shape[0]
        x = x + 0.5 * self.ff1b(gelu(self.ff1a(layernorm(x, self.norms[0]))))
        q, k, v = mx.split(self.qkv(layernorm(x, self.norms[1])), 3, axis=-1)
        q = self.rotate(q.reshape(n, 8, 110)).transpose(1, 0, 2)
        k = self.rotate(k.reshape(n, 8, 110)).transpose(1, 0, 2)
        v = v.reshape(n, 8, 110).transpose(1, 0, 2)
        q = quant(q, self.scales[0])
        k = quant(k, self.scales[1])
        v = quant(v, self.scales[3])
        if self.cache is not None:
            k = mx.concatenate((self.cache[0], k), axis=1)
            v = mx.concatenate((self.cache[1], v), axis=1)
        self.cache = (k[:, -512:], v[:, -512:])
        self.pos += n
        z = quant(mx.softmax(q @ k.transpose(0, 2, 1) / 110 ** 0.5, axis=-1), self.scales[2]) @ v
        x = x + self.out(z.transpose(1, 0, 2).reshape(n, 880))
        a, b = mx.split(self.pw1(layernorm(x, self.norms[2])), 2, axis=-1)
        z = a * mx.sigmoid(b)
        padded = mx.concatenate((self.conv, z, mx.zeros((2, 880))), axis=0)
        self.conv = z[-2:]
        y = mx.broadcast_to(self.db, z.shape)
        for i in range(5):
            y = y + padded[i:i + n] * self.dw[i]
        x = x + self.pw2(gelu(layernorm(y, self.norms[3])))
        x = x + 0.5 * self.ff2b(gelu(self.ff2a(layernorm(x, self.norms[4]))))
        return layernorm(x, self.norms[5])

class StreamEncoder:

    def __init__(self, source):
        m = source.model('asr/models/encoder.opt.stream.pthlite')
        t = m.tensors
        self.convs = []
        for n in m.nodes:
            if n['op'] in ('QConvInt8', 'QConv'):
                a = [t.get(k) for k in n['inputs']]
                w = a[2].astype(np.float32) * a[3][:, None, None, None]
                b = a[6] if n['op'] == 'QConvInt8' else a[4]
                self.convs.append((mx.array(w.transpose(0, 2, 3, 1)), mx.array(b)))

        def linear(stem, bias):
            w = t[stem + '/Constant_2_output_0'].astype(np.float32) * t[stem + '/Constant_1_output_0'][None, :]
            return (mx.array(w), mx.array(t[bias]))
        self.front = linear('/audio_encoder/acoustic_front_end_module/out', 'audio_encoder.acoustic_front_end_module.out.b')
        self.project = linear('/audio_proj', 'audio_proj.b')
        self.blocks = [StreamConformer(t, n) for n in m.nodes if n['op'] == 'QW4ConformerLayerRopeChunk']
        stem = 'audio_encoder.acoustic_backbone_module.after_norm'
        self.norm = mx.array(np.concatenate((t[stem + '.weight'], t[stem + '.bias'])))
        self.reset()

    def reset(self):
        self.frontcache = [mx.zeros((1, 1, 80, 1)), mx.zeros((1, 1, 40, 256))]
        for b in self.blocks:
            b.reset()

    def __call__(self, features):
        if features.shape != (16, 80):
            raise ValueError('Expected 16 feature frames')
        x = quant(mx.array(features)[None, :, :, None], 0.06953888)
        for i, (w, b) in enumerate(self.convs):
            old = self.frontcache[i]
            self.frontcache[i] = x[:, -1:]
            x = mx.concatenate((old, x), axis=1)
            x = mx.pad(x, ((0, 0), (0, 0), (1, 1), (0, 0)))
            x = mx.maximum(mx.conv2d(x, w, stride=2) + b, 0)
            if i == 0:
                x = quant(x, 0.00946512)
        x = quant(x.transpose(0, 1, 3, 2).reshape(4, -1), 0.04277448) @ self.front[0] + self.front[1]
        results = []
        for i in range(0, 4, 4):
            y = x[i:i + 4]
            for b in self.blocks:
                y = b(y)
                mx.eval(y, b.cache, b.conv)
            y = layernorm(y, self.norm).reshape(1, 3520)
            results.append(quant(y, 0.05674971) @ self.project[0] + self.project[1])
        return mx.concatenate(results, axis=0)
