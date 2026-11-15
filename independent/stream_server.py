"""Loopback recording UI with a persistent sandboxed streaming worker."""
import argparse
import atexit
import base64
import http.server
import json
import pathlib
import secrets
import select
import subprocess
import sys
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parent


class Worker:
    def __init__(self, model):
        command = ['sandbox-exec', '-D', 'DOUBAO_DATA='+str(pathlib.Path.home()/'Library/Application Support/DoubaoIme'),
                   '-D', 'PRIVATE_FRAMEWORKS='+str(ROOT.parent/'Frameworks'), '-f', str(ROOT/'offline.sb'),
                   sys.executable, '-u', str(ROOT/'stream_worker.py'), str(model)]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        atexit.register(self.process.terminate)
        self.ready = self.receive()
        self.lock = threading.Lock()
        self.token = None
        self.updated = 0
        self.sequence = 0

    def receive(self):
        if not select.select([self.process.stdout], [], [], 120)[0]:
            self.process.kill()
            raise RuntimeError('Inference worker timed out; restart the server')
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError('Inference worker stopped; restart the server')
        result = json.loads(line)
        if 'error' in result:
            raise ValueError(result['error'])
        return result

    def call(self, data):
        self.process.stdin.write(json.dumps(data)+'\n')
        self.process.stdin.flush()
        return self.receive()


def main(model, port=8766):
    worker = Worker(model)
    origins = {f'http://localhost:{port}', f'http://127.0.0.1:{port}'}

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def reply(self, code, data, kind='application/json'):
            body = json.dumps(data, ensure_ascii=False).encode() if kind == 'application/json' else data
            self.send_response(code)
            for key, value in [('Content-Type',kind), ('Content-Length',str(len(body))),
                               ('Cache-Control','no-store'), ('X-Content-Type-Options','nosniff')]:
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)
        def valid(self):
            return 'http://'+self.headers.get('Host','') in origins
        def do_GET(self):
            if not self.valid(): return self.reply(403, {'error':'Invalid host'})
            if self.path == '/':
                return self.reply(200, (ROOT/'stream.html').read_bytes(), 'text/html; charset=utf-8')
            if self.path == '/status': return self.reply(200, worker.ready)
            self.reply(404, {'error':'Not found'})
        def do_POST(self):
            if not self.valid() or self.headers.get('Origin') not in origins:
                return self.reply(403, {'error':'Invalid origin'})
            try:
                n = int(self.headers.get('Content-Length','0'))
                if not 0 <= n <= 65536: raise ValueError('Invalid body size')
                body = self.rfile.read(n)
                with worker.lock:
                    if self.path == '/start':
                        if worker.token and time.monotonic()-worker.updated < 60:
                            return self.reply(409, {'error':'另一个录音会话正在使用模型。'})
                        worker.call({'reset':True})
                        worker.token = secrets.token_urlsafe(24)
                        worker.sequence = 0
                        result = {'session':worker.token, 'load':worker.ready['load']}
                    elif self.path in ('/feed','/finish','/cancel'):
                        if not worker.token or self.headers.get('X-Session') != worker.token:
                            return self.reply(409, {'error':'录音会话已失效，请重新开始。'})
                        if self.path == '/cancel':
                            worker.token = None
                            result = {'cancelled': True}
                        else:
                            if int(self.headers.get('X-Sequence','-1')) != worker.sequence:
                                raise ValueError('Out-of-order audio chunk')
                            if len(body)%2: raise ValueError('Expected PCM16')
                            start = time.monotonic()
                            result = worker.call({'pcm':base64.b64encode(body).decode(), 'finish':self.path=='/finish'})
                            result['request_seconds'] = time.monotonic()-start
                            worker.sequence += 1
                            if self.path == '/finish': worker.token = None
                    else: return self.reply(404, {'error':'Not found'})
                    worker.updated = time.monotonic()
                self.reply(200, result)
            except Exception as exc:
                self.reply(400, {'error':str(exc)})
    print(f'MLX 实验性流式录音：http://localhost:{port} (load {worker.ready["load"]:.2f}s)', flush=True)
    http.server.ThreadingHTTPServer(('127.0.0.1',port),Handler).serve_forever()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',type=pathlib.Path,default=ROOT/'exported-model')
    ap.add_argument('--port',type=int,default=8766)
    args = ap.parse_args()
    main(args.model.resolve(), args.port)
