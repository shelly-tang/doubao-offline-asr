import json,struct,pathlib,numpy as np
root=pathlib.Path(__file__).resolve().parent
prev=root
c=json.load(open(prev/'components.json'))
names=['asr/models/llm_decoder_embeding_logit.pthlite','asr/models/encoder.opt.stream.pthlite','asr/models/llm_decoder.pthlite','asr/models/non_s_draft.pthlite','asr/models/non_s_encoder.opt.stream.pthlite','asr/models/non_s_llm_embedding_logit.pthlite','asr/models/llm_decoder_embeding_wpe.pthlite','asr/models/non_s_llm_decoder.pthlite','vad/encoder/vad_v4.pth.pthlite']
entries=[('config.yaml',2809,12128),('asr/token/merges.txt',27925,620622),('asr/token/vocab.txt',648547,935789),('asr/token/add_special_tokens.txt',1584336,6803),('asr/feature_extractor/cmvn.txt',823842421,1584),('asr/feature_extractor/fbank.conf',823842332,89)]
entries += [(n,x['base'],x['size']) for n,x in zip(names,c)]
headsize=21+sum(len(n.encode())+1+24 for n,_,_ in entries)
pos=headsize;directory=b''
for n,_,sz in entries:directory+=n.encode()+b'\0'+struct.pack('<QQQ',pos,sz,sz);pos+=sz
binary=(root/'runtime.bundle').read_bytes();state=0xdeadbeef;keybuf=bytearray(binary[0xe8a504:0xe8a50c])
for offset in (0,4):
 state=(state*0x08088405+1)&0xffffffff
 struct.pack_into('<I',keybuf,offset,struct.unpack_from('<I',keybuf,offset)[0]^state)
key=bytes(keybuf);mask=np.array([key[i%8]^i for i in range(256)],dtype=np.uint8)
with open(pathlib.Path.home()/'Library/Application Support/DoubaoIme/ASRModel/versions/1_8/asr_model.model','rb') as src,open(root/'plain.model','wb') as dst:
 dst.write(b'FLUTE'+struct.pack('<IQI',1,headsize,len(entries))+directory)
 for n,off,sz in entries:
  src.seek(off);b=np.frombuffer(src.read(sz),dtype=np.uint8).copy();b^=np.resize(mask,sz);dst.write(b.tobytes());print(n,sz,flush=True)
