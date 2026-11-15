"""Experimental non-streaming Conformer reconstruction for short utterances.
Chunk/state and quantized-attention parity are not yet established.
"""
import numpy as np
import mlx.core as mx
from quantization import unpack_q4


def layernorm(x, combined, eps=1e-5):
    gamma, beta = mx.split(combined, 2)
    y = x - mx.mean(x, axis=-1, keepdims=True)
    return y * mx.rsqrt(mx.mean(y*y, axis=-1, keepdims=True)+eps) * gamma + beta


def gelu(x):
    return 0.5 * x * (1 + mx.tanh(mx.clip((2/np.pi)**.5 * (x + .044715*x*x*x), -20, 20)))


class Linear:
    def __init__(self, w, scale, bias):
        self.weight = mx.array(unpack_q4(w).astype(np.float32) * scale[None, :])
        self.bias = mx.array(bias)
        mx.eval(self.weight, self.bias)
    def __call__(self, x): return x @ self.weight + self.bias


class Conformer:
    def __init__(self, tensors, node):
        t = [tensors.get(s) for s in node['inputs']]
        self.norms = [mx.array(t[i]) for i in (2, 9, 16, 22, 26, 33)]
        self.ff1a, self.ff1b, self.qkv, self.out, self.pw1, self.pw2, self.ff2a, self.ff2b = [
            Linear(t[i], t[i+1], t[i+2]) for i in (3, 6, 10, 13, 17, 23, 27, 30)]
        self.dw, self.db = mx.array(t[20]), mx.array(t[21])
        self.rope = mx.array(t[1])
        self.attr = node['attrs']

    def rotate(self, x):
        n = x.shape[0]
        c, s = self.rope[0,:n,None,:], self.rope[1,:n,None,:]
        pairs = x.reshape(n, 8, 55, 2)
        rotated = mx.stack((-pairs[...,1], pairs[...,0]), axis=-1).reshape(x.shape)
        return x*c + rotated*s

    def __call__(self, x):
        n = x.shape[0]
        x = x + .5 * self.ff1b(gelu(self.ff1a(layernorm(x,self.norms[0]))))
        q,k,v = mx.split(self.qkv(layernorm(x,self.norms[1])),3,axis=-1)
        q = self.rotate(q.reshape(n,8,110)).transpose(1,0,2)
        k = self.rotate(k.reshape(n,8,110)).transpose(1,0,2)
        v = v.reshape(n,8,110).transpose(1,0,2)
        t = mx.arange(n)
        chunk = self.attr['chunk_size']
        end = (t // chunk + 1) * chunk
        start = mx.maximum(t//chunk*chunk - self.attr['attn_left_context'],0)
        mask = (t[None,:] < end[:,None]) & (t[None,:] >= start[:,None])
        scores = q @ k.transpose(0,2,1) / 110**.5
        attn = mx.softmax(mx.where(mask,scores,-float('inf')),axis=-1) @ v
        x = x + self.out(attn.transpose(1,0,2).reshape(n,880))
        a,b = mx.split(self.pw1(layernorm(x,self.norms[2])),2,axis=-1)
        z = a*mx.sigmoid(b)
        padded = mx.pad(z,((2,2),(0,0)))
        # Preserve left context; right context stops at each 100-frame chunk.
        y = mx.broadcast_to(self.db, z.shape)
        for i in range(5):
            source = t+i-2
            valid = (source>=0) & (source<n) & (source<end)
            y = y + padded[i:i+n]*self.dw[i]*valid[:,None]
        x = x + self.pw2(gelu(layernorm(y,self.norms[3])))
        x = x + .5*self.ff2b(gelu(self.ff2a(layernorm(x,self.norms[4]))))
        return layernorm(x,self.norms[5])


class Encoder:
    def __init__(self, pack):
        m = pack.model('asr/models/non_s_encoder.opt.stream.pthlite')
        t = m.tensors
        self.convs = []
        for i in (0,2):
            stem = f'audio_encoder.acoustic_front_end_module.conv.{i}'
            self.convs.append((mx.array(t[stem+'.weight'].transpose(0,2,3,1)),mx.array(t[stem+'.bias'])))
        self.front = Linear(t['/out/Constant_output_0'],t['/out/Constant_1_output_0'],t['audio_encoder.acoustic_front_end_module.out.bias'])
        self.blocks = [Conformer(t,n) for n in m.nodes if n['op']=='QW4ConformerLayerRopeChunk']
        stem='audio_encoder.acoustic_backbone_module.model.after_norm'
        self.norm=mx.array(np.concatenate((t[stem+'.weight'],t[stem+'.bias'])))
        self.project = Linear(t['/audio_proj/Constant_output_0'],t['/audio_proj/Constant_1_output_0'],t['audio_proj.bias'])

    def __call__(self, features):
        if features.shape[0] > 1920: raise ValueError('Experimental short-utterance path limited to 1920 feature frames')
        x = mx.array(features)[None,:,:,None]
        for w,b in self.convs:
            x = mx.maximum(mx.conv2d(x,w,stride=2,padding=1)+b,0)
        x = x.transpose(0,1,3,2).reshape(x.shape[1],-1)
        x = self.front(x)
        for block in self.blocks:
            x = block(x); mx.eval(x)
        x = layernorm(x,self.norm)
        x = x[:x.shape[0]//4*4].reshape(-1,880*4)
        return self.project(x)
