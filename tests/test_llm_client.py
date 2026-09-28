import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_client import call_once


def run_mock_chat_server():
    seen = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            seen["path"] = self.path
            seen["authorization"] = self.headers.get("Authorization")
            length = int(self.headers.get("Content-Length", "0"))
            seen["body"] = json.loads(self.rfile.read(length))

            payload = {
                "usage": {"completion_tokens": 7},
                "choices": [{"message": {"content": "ok"}}],
            }
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, seen


def main():
    server, seen = run_mock_chat_server()
    os.environ["TOGETHER_API_KEY"] = "test-key"

    _, tokens = call_once(
        f"http://127.0.0.1:{server.server_port}",
        "test-model",
        "hello",
        16,
        "chat",
    )

    server.shutdown()

    assert tokens == 7
    assert seen["path"] == "/v1/chat/completions"
    assert seen["authorization"] == "Bearer test-key"
    assert seen["body"]["model"] == "test-model"
    assert seen["body"]["messages"][0]["content"] == "hello"
    assert seen["body"]["max_tokens"] == 16
    print("llm_client chat smoke test passed")


if __name__ == "__main__":
    main()
