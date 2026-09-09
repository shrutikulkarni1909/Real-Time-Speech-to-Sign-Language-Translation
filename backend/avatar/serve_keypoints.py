from http.server import SimpleHTTPRequestHandler, HTTPServer
import os
import json
import time

from backend.avatar.push_to_avatar import push_gloss
from backend.gloss.infer import infer

BASE = os.path.dirname(os.path.abspath(__file__))
WLASL_DIR = os.path.abspath(os.path.join(BASE, "..", "datasets", "wlasl"))
# Project root (parent of backend/) — used to serve frontend/ from the same port as /keypoints
REPO_ROOT = os.path.abspath(os.path.join(BASE, "..", ".."))


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        # CORS: allow viewer on another port (e.g. legacy http.server :8000) or same origin
        origin = self.headers.get("Origin", "")
        if origin in ("http://127.0.0.1:8000", "http://localhost:8000", ""):
            self.send_header("Access-Control-Allow-Origin", origin or "*")
        else:
            self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200, "ok")
        self.end_headers()

    def do_POST(self):
        # Minimal API so the browser UI can submit recognized speech as text.
        # POST /api/text  { "text": "hello" }  -> infer -> push_gloss -> state.json
        # POST /api/gloss { "gloss": "HELLO" } -> push_gloss -> state.json
        path_only = self.path.split("?", 1)[0].split("#", 1)[0]
        if path_only not in ("/api/text", "/api/gloss"):
            self.send_response(404)
            self.end_headers()
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            payload = json.loads(raw.decode("utf-8") or "{}")
        except Exception:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": False, "error": "Invalid JSON"}).encode("utf-8"))
            return

        try:
            if path_only == "/api/text":
                text = str(payload.get("text", "")).strip()
                if not text:
                    raise ValueError("Missing 'text'")
                gloss = infer(text)
                tokens = push_gloss(gloss)
                # ✅ IMPROVED: Return detailed info for debugging
                out = {"ok": True, "text": text, "gloss": gloss, "tokens": tokens, 
                       "token_count": len(tokens)}
            else:
                gloss = str(payload.get("gloss", "")).strip()
                if not gloss:
                    raise ValueError("Missing 'gloss'")
                tokens = push_gloss(gloss)
                out = {"ok": True, "gloss": gloss, "tokens": tokens, 
                       "token_count": len(tokens)}

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(out).encode("utf-8"))
        except Exception as e:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": False, "error": str(e)}).encode("utf-8"))

    def translate_path(self, path):
        """Map URLs to disk: WLASL dataset for /keypoints and /runtime; repo root for everything else."""
        path = path.split("?", 1)[0].split("#", 1)[0]
        path = path.lstrip("/")
        root = WLASL_DIR if (path.startswith("keypoints/") or path.startswith("runtime/")) else REPO_ROOT
        # For this local demo server, keep routing simple and robust on Windows.
        return os.path.normpath(os.path.join(root, path))

if __name__ == "__main__":
    # Ensure runtime/state.json exists so the viewer can poll immediately
    runtime_dir = os.path.join(WLASL_DIR, "runtime")
    state_path = os.path.join(runtime_dir, "state.json")
    os.makedirs(runtime_dir, exist_ok=True)
    if not os.path.exists(state_path):
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump({"id": 0, "tokens": [], "ts": time.time()}, f)

    print("Serving at http://127.0.0.1:9000/")
    print(" - Open viewer: http://127.0.0.1:9000/frontend/avatar_viewer/")
    print(" - keypoints: /keypoints/<GLOSS>.json")
    print(" - runtime:   /runtime/state.json")
    print(" - POST text: /api/text  (use same origin as this server to avoid HTTP 501 on :8000)")
    HTTPServer(("127.0.0.1", 9000), Handler).serve_forever()
