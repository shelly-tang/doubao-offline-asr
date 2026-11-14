#!/usr/bin/env python3
import http.server,json,pathlib,tempfile,subprocess,time,re,threading,wave,urllib.parse
ROOT=pathlib.Path(__file__).resolve().parent
PORT=8765
LOCK=threading.Lock()
class Handler(http.server.BaseHTTPRequestHandler):
 def log_message(self,*args):pass
 def reply(self,status,data,kind='application/json'):
  body=json.dumps(data,ensure_ascii=False).encode() if kind=='application/json' else data
  self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(body)
 def valid(self):return self.headers.get('Host') in (f'localhost:{PORT}',f'127.0.0.1:{PORT}')
 def do_GET(self):
  if not self.valid():return self.reply(403,{'error':'Invalid host'})
  if self.path=='/':return self.reply(200,(ROOT/'web/index.html').read_bytes(),'text/html; charset=utf-8')
  self.reply(404,{'error':'Not found'})
 def do_POST(self):
  if not self.valid() or self.headers.get('Origin') not in (f'http://localhost:{PORT}',f'http://127.0.0.1:{PORT}'):return self.reply(403,{'error':'Invalid origin'})
  if self.path!='/transcribe':return self.reply(404,{'error':'Not found'})
  try:n=int(self.headers.get('Content-Length','0'))
  except ValueError:return self.reply(400,{'error':'Invalid length'})
  if not 44<n<=25_000_000:return self.reply(413,{'error':'录音为空或超过大小限制，请录制两分钟以内的音频。'})
  if not LOCK.acquire(False):return self.reply(409,{'error':'正在识别上一段录音，请稍候。'})
  try:
   start=time.monotonic()
   with tempfile.TemporaryDirectory(prefix='doubao-mic-') as tmp:
    tmp=pathlib.Path(tmp);audio=tmp/'audio.wav';audio.write_bytes(self.rfile.read(n))
    with wave.open(str(audio)) as w:duration=w.getnframes()/w.getframerate()
    if duration>121:raise ValueError('录音请限制在两分钟以内。')
    log=tmp/'engine.log'
    proc=subprocess.run(['python3',str(ROOT/'transcribe.py'),str(audio),'--json','--log',str(log)],capture_output=True,text=True,timeout=190)
    if proc.returncode:raise RuntimeError('识别失败：'+proc.stderr[-1200:])
    result=json.loads(proc.stdout);logs=log.read_text();metrics={}
    for field in ('load','inference'):
     match=re.search(field+r' seconds=([\d.]+)',logs);metrics[field]=float(match[1]) if match else None
    self.reply(200,{'text':''.join(x.get('text','') for x in result['results']),'duration':duration,'total':time.monotonic()-start,**metrics})
  except Exception as e:self.reply(500,{'error':str(e)})
  finally:LOCK.release()
if __name__=='__main__':
 print(f'本地录音测试：http://localhost:{PORT}',flush=True)
 http.server.ThreadingHTTPServer(('127.0.0.1',PORT),Handler).serve_forever()
