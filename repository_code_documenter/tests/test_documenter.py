import json, threading, tempfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from core.ast_analyzer import RepositoryAnalyzer
from core.documenter import DocumentationGenerator
from core.llm import OllamaLLM, LLMError


def _index(d):
    p = Path(d)
    (p / 'a.py').write_text('CONST = 1\ndef helper(x):\n    return x + CONST\n')
    (p / 'b.py').write_text('from a import helper\ndef main():\n    return helper(2)\n')
    return RepositoryAnalyzer(d).analyze()


def test_prompt_contains_target_code_and_relationships():
    with tempfile.TemporaryDirectory() as d:
        idx = _index(d)
        main = next(n for n in idx.nodes.values() if n.name == 'main')
        prompt = DocumentationGenerator(None).build_prompt(main, [], idx.files, idx)
        assert 'return helper(2)' in prompt          # target code is present
        assert 'helper (a.py)' in prompt             # callee resolved from the graph


def test_prompt_is_bounded_for_huge_targets():
    with tempfile.TemporaryDirectory() as d:
        idx = _index(d)
        main = next(n for n in idx.nodes.values() if n.name == 'main')
        main.code = 'x = 1\n' * 200000
        retrieved = [{'id': f'r{i}', 'document': 'y' * 50000, 'hybrid_score': .5} for i in range(20)]
        prompt = DocumentationGenerator(None).build_prompt(main, retrieved, idx.files * 1000, idx)
        assert len(prompt) < 30000


def _server(handler):
    srv = HTTPServer(('127.0.0.1', 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_ollama_streaming_ok():
    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            assert body['options']['num_ctx'] >= 8192
            self.send_response(200); self.end_headers()
            for t, done in [('Hello ', False), ('world', False), ('', True)]:
                self.wfile.write((json.dumps({'response': t, 'done': done}) + '\n').encode())
        def log_message(self, *a): pass
    srv = _server(H)
    out = OllamaLLM(base_url=f'http://127.0.0.1:{srv.server_port}').generate('hi')
    assert out == 'Hello world'


def test_ollama_missing_model_message():
    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(404); self.end_headers(); self.wfile.write(b'{"error":"model not found"}')
        def log_message(self, *a): pass
    srv = _server(H)
    try:
        OllamaLLM(model='nope', base_url=f'http://127.0.0.1:{srv.server_port}').generate('hi')
        assert False
    except LLMError as e:
        assert 'ollama pull nope' in str(e)
