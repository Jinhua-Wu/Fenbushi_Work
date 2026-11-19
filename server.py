from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from pathlib import Path
import os,json
from model import Config,simulate

class Handler(SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path!='/simulate': self.send_error(404); return
        try:
            if int(self.headers.get('Content-Length',0))>10000: raise ValueError('Request too large')
            cfg=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            result=simulate(Config(**cfg),True)
            data=json.dumps(result).encode()
            self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(data)
        except Exception as e:
            self.send_response(400);self.end_headers();self.wfile.write(str(e).encode())

if __name__=='__main__':
    os.chdir(Path(__file__).parent)
    print('Open http://127.0.0.1:8765/index.html ; Ctrl+C to stop',flush=True)
    ThreadingHTTPServer(('127.0.0.1',8765),Handler).serve_forever()
