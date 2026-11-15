"""Read FLUTE v1 / Panther FlatBuffers without loading any vendor code."""
import mmap
import struct
from pathlib import Path
import numpy as np


class Pack:
    def __init__(self, path):
        self.file = open(path, 'rb')
        self.data = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        if self.data[:9] != b'FLUTE\x01\0\0\0':
            raise ValueError('Expected locally extracted, plaintext FLUTE v1 pack')
        header, count = struct.unpack_from('<QI', self.data, 9)
        if not 21 <= header <= len(self.data):
            raise ValueError('Invalid directory size')
        self.entries = {}
        pos = 21
        for _ in range(count):
            end = self.data.find(b'\0', pos, header)
            if end < 0 or end + 25 > header:
                raise ValueError('Truncated directory')
            name = self.data[pos:end].decode()
            offset, size, unpacked = struct.unpack_from('<QQQ', self.data, end + 1)
            if size != unpacked or offset < header or offset + size > len(self.data):
                raise ValueError(f'Invalid entry: {name}')
            self.entries[name] = (offset, size)
            pos = end + 25

    def read(self, name):
        offset, size = self.entries[name]
        return self.data[offset:offset + size]

    def model(self, name):
        return Model(self.read(name))


class Model:
    def __init__(self, data):
        self.data = data
        root = self.fields(32 + self.u32(32))
        self.tensors = {}
        for table in self.tables(root[2]):
            f = self.fields(table)
            name = self.string(f[0])
            shape = tuple(self.vector(f[3], '<i8')) if 3 in f else ()
            dtype = self.u32(f[1]) if 1 in f else 0
            field, kind = {1: (15, '<f4'), 2: (6, '<i4'), 3: (5, 'i1'), 7: (11, '<i8')}[dtype]
            a = self.vector(f[field], kind)
            if a.size != int(np.prod(shape)):
                raise ValueError(f'{name}: tensor shape/length mismatch {shape}, {a.size}')
            self.tensors[name] = a.reshape(shape)
        self.nodes = []
        for graph in self.tables(root[1]):
            for table in self.tables(self.fields(graph)[5]):
                f = self.fields(table)
                attrs = {}
                for at in self.tables(f[4]) if 4 in f else []:
                    a = self.fields(at)
                    key = self.string(a[0])
                    if 10 in a: value = struct.unpack_from('<q', data, a[10])[0]
                    elif 11 in a: value = self.vector(a[11], '<i8').tolist()
                    elif 3 in a: value = struct.unpack_from('<f', data, a[3])[0]
                    elif 4 in a: value = self.string(a[4])
                    else: value = 0
                    attrs[key] = value
                node = {'op': self.string(f[0]), 'attrs': attrs}
                for label, field in [('inputs', 2), ('outputs', 3)]:
                    node[label] = [self.string(self.fields(t)[1]) for t in self.tables(f[field])] if field in f else []
                self.nodes.append(node)

    def u32(self, p): return struct.unpack_from('<I', self.data, p)[0]
    def ptr(self, p): return p + self.u32(p)
    def fields(self, p):
        vt = p - struct.unpack_from('<i', self.data, p)[0]
        size = struct.unpack_from('<H', self.data, vt)[0]
        offsets = struct.unpack_from('<' + 'H' * ((size - 4) // 2), self.data, vt + 4)
        return {i: p + o for i, o in enumerate(offsets) if o}
    def vector(self, p, dtype):
        q = self.ptr(p)
        return np.frombuffer(self.data, dtype=dtype, count=self.u32(q), offset=q + 4)
    def tables(self, p):
        q = self.ptr(p)
        return [self.ptr(q + 4 + 4 * i) for i in range(self.u32(q))]
    def string(self, p):
        q = self.ptr(p)
        return self.data[q + 4:q + 4 + self.u32(q)].decode()


from quantization import unpack_q4  # compatibility for earlier local scripts


if __name__ == '__main__':
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument('pack', type=Path)
    args = ap.parse_args()
    pack = Pack(args.pack)
    for name in pack.entries:
        if name.endswith('.pthlite'):
            m = pack.model(name)
            print(json.dumps({'name': name, 'tensors': len(m.tensors), 'bytes': sum(a.nbytes for a in m.tensors.values()), 'ops': sorted(set(n['op'] for n in m.nodes))}))
