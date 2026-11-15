"""Extract installed weights without constructing or loading a runtime bundle."""
import argparse,hashlib,json,struct
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
APP=Path('/Library/Input Methods/DoubaoIme.app/Contents/MacOS/DoubaoIme')
MODEL=Path.home()/'Library/Application Support/DoubaoIme/ASRModel/versions/1_8/asr_model.model'

def extract(app,model,output):
    binary=app.read_bytes()
    if hashlib.sha256(binary).hexdigest()!='5b48748fa6394811cba0f8c05905ffebb9a2cd02faf6d47b26deb3d9b86247e0':
        raise ValueError('Unsupported application version')
    with model.open('rb') as f:
        if hashlib.file_digest(f,'sha256').hexdigest()!='786d0ac0c8d14f451830589f408ec113385433086eea6e693a21cf0c6281d223':
            raise ValueError('Unsupported model version')
    arm=None
    for i in range(struct.unpack_from('>I',binary,4)[0]):
        cpu,_,offset,size,_=struct.unpack_from('>IIIII',binary,8+20*i)
        if cpu==0x100000c:arm=binary[offset:offset+size]
    if arm is None:raise ValueError('Missing ARM64 slice')
    # Recover the per-file XOR obfuscation pattern from the local installation.
    state=0xdeadbeef;key=bytearray(arm[0xe8a504:0xe8a50c])
    for offset in (0,4):
        state=(state*0x08088405+1)&0xffffffff
        struct.pack_into('<I',key,offset,struct.unpack_from('<I',key,offset)[0]^state)
    mask=np.array([key[i%8]^i for i in range(256)],dtype=np.uint8)
    names=['llm_decoder_embeding_logit','encoder.opt.stream','llm_decoder','non_s_draft','non_s_encoder.opt.stream','non_s_llm_embedding_logit','llm_decoder_embeding_wpe','non_s_llm_decoder']
    names=['asr/models/'+n+'.pthlite' for n in names]+['vad/encoder/vad_v4.pth.pthlite']
    entries=[('config.yaml',2809,12128),('asr/token/merges.txt',27925,620622),('asr/token/vocab.txt',648547,935789),('asr/token/add_special_tokens.txt',1584336,6803),('asr/feature_extractor/cmvn.txt',823842421,1584),('asr/feature_extractor/fbank.conf',823842332,89)]
    entries += [(n,x['base'],x['size']) for n,x in zip(names,json.loads((ROOT/'components.json').read_text()))]
    headsize=21+sum(len(n.encode())+25 for n,_,_ in entries)
    pos=headsize;directory=bytearray()
    for n,_,size in entries:
        directory+=n.encode()+b'\0'+struct.pack('<QQQ',pos,size,size);pos+=size
    # Exclusive create prevents accidentally replacing a working model pack.
    with model.open('rb') as src,output.open('xb') as dst:
        dst.write(b'FLUTE'+struct.pack('<IQI',1,headsize,len(entries))+directory)
        for n,offset,size in entries:
            src.seek(offset)
            for i in range(0,size,1048576):
                data=np.frombuffer(src.read(min(1048576,size-i)),dtype=np.uint8).copy()
                data^=np.resize(mask,len(data));dst.write(data.tobytes())
    return output

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--app',type=Path,default=APP);ap.add_argument('--model',type=Path,default=MODEL);ap.add_argument('--output',type=Path,default=Path(__file__).with_name('plain.model'))
    a=ap.parse_args();print(extract(a.app,a.model,a.output))
