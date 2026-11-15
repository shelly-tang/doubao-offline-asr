#!/usr/bin/env python3
import sys,argparse,os
import http.server,json,pathlib,tempfile,subprocess,time,re,threading,wave,urllib.parse
ROOT=pathlib.Path(__file__).resolve().parent
PORT=8766
PACK=None
LOCK=threading.Lock()
def page():
 html=(ROOT.parent/'web/index.html').read_text()
 html=html.replace('豆包 · 本地语音实验','独立 MLX · 本地语音实验').replace('LOCAL VOICE LAB / 01','INDEPENDENT MLX / EXPERIMENTAL')
 html=html.replace('豆包原版离线模型 · 本机识别 · 录音不上传云端','原版权重 · 独立 MLX 推理 · 不加载豆包私有库')
 html=html.replace('建议先录制 5–20 秒，最长 2 分钟','实验版：建议 5–15 秒，17 秒自动停止；尚未验证真人准确率')
 html=html.replace('if(t>=120)','if(t>=17)')
 return html.encode()

class Handler(http.server.BaseHTTPRequestHandler):
 def log_message(self,*args):pass
 def reply(self,status,data,kind='application/json'):
  body=json.dumps(data,ensure_ascii=False).encode() if kind=='application/json' else data
  self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(body)
 def valid(self):return self.headers.get('Host') in (f'localhost:{PORT}',f'127.0.0.1:{PORT}')
 def do_GET(self):
  if not self.valid():return self.reply(403,{'error':'Invalid host'})
  if self.path=='/':return self.reply(200,page(),'text/html; charset=utf-8')
  self.reply(404,{'error':'Not found'})
 def do_POST(self):
  if not self.valid() or self.headers.get('Origin') not in (f'http://localhost:{PORT}',f'http://127.0.0.1:{PORT}'):return self.reply(403,{'error':'Invalid origin'})
  if self.path!='/transcribe':return self.reply(404,{'error':'Not found'})
  try:n=int(self.headers.get('Content-Length','0'))
  except ValueError:return self.reply(400,{'error':'Invalid length'})
  if not 44<n<=4_000_000:return self.reply(413,{'error':'录音为空或超过大小限制，请录制 18 秒以内的音频。'})
  if not LOCK.acquire(False):return self.reply(409,{'error':'正在识别上一段录音，请稍候。'})
  try:
   start=time.monotonic()
   with tempfile.TemporaryDirectory(prefix='doubao-mic-') as tmp:
    tmp=pathlib.Path(tmp);audio=tmp/'audio.wav';audio.write_bytes(self.rfile.read(n))
    with wave.open(str(audio)) as w:duration=w.getnframes()/w.getframerate()
    if duration>18 or duration<=0:raise ValueError('录音请限制在 18 秒以内。')
    converted=tmp/'pcm.wav'
    subprocess.run(['afconvert','-f','WAVE','-d','LEI16@16000','-c','1',str(audio),str(converted)],check=True,capture_output=True,timeout=20)
    command=['sandbox-exec','-D','DOUBAO_DATA='+str(pathlib.Path.home()/'Library/Application Support/DoubaoIme'),'-D','PRIVATE_FRAMEWORKS='+str(ROOT.parent/'Frameworks'),'-f',str(ROOT/'offline.sb'),sys.executable,str(ROOT/'transcribe.py'),str(PACK),str(converted),'--json']
    proc=subprocess.run(command,capture_output=True,text=True,timeout=120)
    if proc.returncode:raise RuntimeError('识别失败：'+proc.stderr[-1200:])
    result=json.loads(proc.stdout)
    if result['truncated']:raise RuntimeError('输出达到长度限制，当前结果可能不完整，请缩短录音。')
    self.reply(200,{'text':result['text'],'duration':duration,'total':time.monotonic()-start,'load':result['load_seconds'],'inference':result['inference_seconds']})
  except Exception as e:self.reply(500,{'error':str(e)})
  finally:LOCK.release()
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--model', '--pack', dest='pack', type=pathlib.Path, default=ROOT/'exported-model', help='Exported model directory or legacy local FLUTE pack');ap.add_argument('--port',type=int,default=8766);ap.add_argument('--streaming', action='store_true', help='Use experimental cached streaming MLX recording UI');args=ap.parse_args()
 PACK=args.pack.resolve();PORT=args.port
 if not (PACK.is_file() or (PACK/'model.json').is_file()):ap.error('找不到模型，请先运行 independent/export.py 或指定 --model')
 if args.streaming:
  from stream_server import main
  main(PACK, PORT)
  sys.exit(0)
 print(f'独立 MLX 录音测试：http://localhost:{PORT}',flush=True)
 http.server.ThreadingHTTPServer(('127.0.0.1',PORT),Handler).serve_forever()
