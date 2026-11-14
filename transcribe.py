#!/usr/bin/env python3
"""Offline Doubao 1.1.0 / model 1.8 proof-of-concept, Apple Silicon only."""
import argparse,json,pathlib,re,subprocess,tempfile,wave,sys
root=pathlib.Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('audio',type=pathlib.Path)
p.add_argument('--json',action='store_true',help='print full recognition JSON')
p.add_argument('--log',type=pathlib.Path,help='save engine diagnostics locally')
a=p.parse_args()
if not a.audio.is_file():p.error('audio file not found')
with tempfile.TemporaryDirectory(prefix='doubao-asr-') as tmp:
 tmp=pathlib.Path(tmp)
 subprocess.run(['/usr/bin/afconvert','-f','WAVE','-d','LEI16@16000','-c','1',str(a.audio.resolve()),str(tmp/'audio.wav')],check=True)
 with wave.open(str(tmp/'audio.wav')) as w:
  if w.getnframes()==0:p.error('audio contains no samples')
  (tmp/'audio.pcm').write_bytes(w.readframes(w.getnframes()))
 try:
  r=subprocess.run(['/usr/bin/sandbox-exec','-f',str(root/'offline.sb'),str(root/'run'),str(root/'runtime.bundle'),str(root/'plain.model'),str(tmp/'audio.pcm')],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,errors='replace',timeout=180)
 except subprocess.TimeoutExpired:
  sys.exit('Recognition timed out after 180 seconds.')
 if a.log:a.log.write_text(r.stdout)
 if r.returncode:
  print(r.stdout[-6000:],file=sys.stderr);sys.exit(r.returncode if r.returncode>0 else 1)
 outputs=[]
 for line in r.stdout.splitlines():
  if line.startswith('FINAL '):
   try:obj=json.loads(line[6:])
   except json.JSONDecodeError:continue
   if 'results' in obj:outputs.append(obj)
 if not outputs:
  print('No final recognition result; use --log to inspect diagnostics.',file=sys.stderr);sys.exit(2)
 result=outputs[-1]
 if a.json:print(json.dumps(result,ensure_ascii=False,indent=2))
 else:print(''.join(x.get('text','') for x in result.get('results',[])))
