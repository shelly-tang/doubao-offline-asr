"""Experimental end-to-end MLX path; not equivalent to the verified vendor runner."""
import argparse,json,time,sys
import numpy as np
import mlx.core as mx
from weights import Pack
from features import fbank,read_wav
from encoder import Encoder
from decoder import Decoder
from text import tokenizer,prompt_parts

ap=argparse.ArgumentParser()
ap.add_argument('pack');ap.add_argument('wav');ap.add_argument('--max-tokens',type=int,default=100)
ap.add_argument('--json',action='store_true',help='Only JSON result on stdout')
ap.add_argument('--audio-embeddings',help='Debug: load previously saved local encoder output')
ap.add_argument('--save-audio-embeddings',help='Debug: save locally; ignored by git')
a=ap.parse_args()
if not 1<=a.max_tokens<=512:ap.error('--max-tokens must be between 1 and 512')
start=time.perf_counter();pack=Pack(a.pack)
tok=tokenizer(pack)
model_load=0.0
if a.audio_embeddings:
    embeds=mx.array(np.load(a.audio_embeddings))
else:
    features=fbank(read_wav(a.wav),pack)
    print(f'features={features.shape}',flush=True,file=sys.stderr)
    load_start=time.perf_counter();encoder=Encoder(pack);model_load+=time.perf_counter()-load_start
    embeds=encoder(features);mx.eval(embeds)
    del encoder
if a.save_audio_embeddings:np.save(a.save_audio_embeddings,np.array(embeds))
encoded=time.perf_counter()
print(f'audio_embeddings={embeds.shape}, elapsed={encoded-start:.3f}s',flush=True,file=sys.stderr)
load_start=time.perf_counter();model=Decoder(pack,activation_int8=False);model_load+=time.perf_counter()-load_start
prefix,suffix=prompt_parts(pack)
left=tok.encode(prefix).ids;right=tok.encode(suffix).ids
x=mx.concatenate((model.embedding[:,mx.array(left)].T,embeds,model.embedding[:,mx.array(right)].T),axis=0)
loaded=time.perf_counter()
h=model(embeddings=x)
ids=[]
for _ in range(a.max_tokens):
    logits=model.logits(h[-1:]);token=int(mx.argmax(logits[0]).item())
    if token==2:break
    ids.append(token);h=model([token])
raw=tok.decode(ids,skip_special_tokens=False)
from text import normalize_output
print(json.dumps({'text':normalize_output(raw),'raw_text':raw,'ids':ids,'elapsed_seconds':time.perf_counter()-start,'encoder_with_load_seconds':encoded-start,'load_seconds':model_load,'inference_seconds':time.perf_counter()-start-model_load,'decode_seconds':time.perf_counter()-loaded,'truncated':token!=2,'status':'experimental MLX; end-to-end smoke tested; no layerwise vendor parity'},ensure_ascii=False,indent=2))
