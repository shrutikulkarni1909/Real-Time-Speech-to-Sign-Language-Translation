from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import json
import os
import urllib.parse
import requests

# ✅ Import your gloss inference (you said this is working now)
try:
    from backend.gloss.infer import infer
    print("✅ infer imported successfully")
except Exception as e:
    print(" infer import failed. Using fallback upper(). Error:", e)

    def infer(text: str) -> str:
        return text.upper()


def load_wlasl_map(json_path: str) -> dict:
    if not os.path.exists(json_path):
        print(f" WLASL JSON not found: {json_path}")
        return {}

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print(" Unexpected JSON format (expected list).")
        return {}

    sign_map = {}
    for entry in data:
        gloss = (entry.get("gloss") or "").strip().upper()
        instances = entry.get("instances") or []
        if not gloss or not instances:
            continue

        url = None
        for inst in instances:
            if isinstance(inst, dict) and inst.get("url"):
                url = inst["url"]
                break

        if url:
            sign_map[gloss] = url

    print(f"✅ Loaded {len(sign_map)} WLASL gloss->url mappings")
    return sign_map


# Put JSON here: project_root/wlasl/WLASL_v0.3.json
WLASL_JSON_PATH = os.path.join("wlasl", "WLASL_v0.3.json")
SIGN_MAP = load_wlasl_map(WLASL_JSON_PATH)


def gloss_to_urls(gloss: str):
    """Return proxied URLs so browser plays from localhost."""
    proxied = []
    for w in gloss.split():
        raw = SIGN_MAP.get(w.upper())
        if raw:
            proxied.append(
                "http://127.0.0.1:5000/proxy?u=" + urllib.parse.quote(raw, safe="")
            )
        else:
            proxied.append(None)
    return proxied


app = Flask(__name__)
CORS(app)


@app.route("/")
def home():
    return "Speech to Sign backend running"


@app.route("/convert", methods=["POST"])
def convert():
    data = request.get_json() or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "Empty input"}), 400

    try:
        gloss = infer(text)
    except Exception as e:
        print(" infer() crashed:", e)
        gloss = text.upper()

    urls = gloss_to_urls(gloss)

    return jsonify({"text": text, "gloss": gloss, "urls": urls})


# ✅ PROXY endpoint: streams remote mp4 through Flask so browser can play it
@app.route("/proxy", methods=["GET"])
def proxy():
    u = request.args.get("u", "")
    if not u:
        return jsonify({"error": "Missing url"}), 400

    try:
        r = requests.get(u, stream=True, timeout=15)
        r.raise_for_status()

        # Some servers don’t send proper content-type; force mp4-ish default
        content_type = r.headers.get("Content-Type", "video/mp4")

        def generate():
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    yield chunk

        return Response(generate(), content_type=content_type)

    except Exception as e:
        print(" Proxy failed:", e, "URL:", u)
        return jsonify({"error": "Proxy failed"}), 502


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
