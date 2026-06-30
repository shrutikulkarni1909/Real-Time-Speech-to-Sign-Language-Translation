from backend.gloss.normalize import normalize_english
from backend.gloss.dataset_loader import load_daily_pairs

USE_MODEL = False   # KEEP FALSE FOR NOW


# ---------------------------
# Load daily-used glosses
# ---------------------------

def load_daily_used_glosses():
    pairs = load_daily_pairs("backend/gloss/daily_pairs.txt")
    gloss_map = {}
    for eng, gloss in pairs:
        gloss_map[normalize_english(eng)] = gloss
    return gloss_map


DAILY_GLOSS_MAP = load_daily_used_glosses()


# ---------------------------
# Simplification
# ---------------------------

def simplify_text(text):
    remove_words = {
        "am", "is", "are", "was", "were",
        "the", "a", "an", "to", "of", "for"
    }
    return " ".join([w for w in text.split() if w not in remove_words])


# ---------------------------
# Safe inference
# ---------------------------

def infer(text):
    norm = normalize_english(text)
    simple = simplify_text(norm)

    # 1️⃣ Exact sentence / phrase match
    if simple in DAILY_GLOSS_MAP:
        return DAILY_GLOSS_MAP[simple]

    # 2️⃣ Word-by-word fallback
    gloss_words = []
    for w in simple.split():
        if w in DAILY_GLOSS_MAP:
            gloss_words.append(DAILY_GLOSS_MAP[w])
        else:
            gloss_words.append(w.upper())  # SAFE fallback

    return " ".join(gloss_words) 


# ---------------------------
# CLI
# ---------------------------

if __name__ == "__main__":
    print(" Dictionary-first ASL inference (model disabled)")

    while True:
        text = input("\nEnter English text (or 'q' to quit): ")
        if text.lower() == "q":
            break

        print("Gloss:", infer(text))
