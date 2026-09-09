import json
import os
import time

STATE_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "datasets",
    "wlasl",
    "runtime",
    "state.json"
)

KEYPOINTS_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "datasets",
    "wlasl",
    "keypoints"
)


def load_supported_tokens():
    """Auto-discover supported glosses from extracted keypoint files."""
    supported = set()
    if not os.path.isdir(KEYPOINTS_DIR):
        return supported

    for name in os.listdir(KEYPOINTS_DIR):
        if not name.lower().endswith(".json"):
            continue
        token = os.path.splitext(name)[0].strip().upper()
        if token:
            supported.add(token)
    return supported


def push_gloss(gloss_sentence: str):
    # clean: remove labels like "(daily)" or "(word-fallback)"
    gloss_sentence = gloss_sentence.split("(")[0].strip()
    tokens = [t.upper() for t in gloss_sentence.split() if t.strip()]

    # ✅ IMPROVED: Don't filter out tokens without keypoints
    # Instead, keep ALL tokens and let frontend handle missing keypoints gracefully
    # supported = load_supported_tokens()
    # tokens = [t for t in tokens if t in supported]
    
    # Log which tokens are missing keypoints (for debugging)
    supported = load_supported_tokens()
    missing_tokens = [t for t in tokens if t not in supported]
    if missing_tokens:
        print(f"⚠️  Missing keypoints for: {', '.join(missing_tokens)}")
        print(f"   (avatar will show fallback/placeholder for these)")
    
    # Keep ALL tokens - the frontend will handle missing ones
    # This allows long sentences to render partially instead of mostly disappearing

    # update state with new id
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)

    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            old = json.load(f)
            old_id = int(old.get("id", 0))
    except Exception:
        old_id = 0

    new_state = {"id": old_id + 1, "tokens": tokens, "ts": time.time()}

    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(new_state, f)

    return tokens
