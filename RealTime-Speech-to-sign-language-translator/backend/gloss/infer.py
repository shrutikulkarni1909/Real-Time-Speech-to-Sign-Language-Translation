from backend.gloss.normalize import normalize_english
from backend.gloss.dataset_loader import load_daily_pairs
from backend.gloss.word_resolver import resolve_word  # ✅ NEW: Import word resolver
from backend.gloss.asl_grammar import convert_to_asl_order  # ✅ NEW: Import ASL grammar

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
# ✅ IMPROVED: Better word removal strategy
# ---------------------------

def simplify_text(text):
    """
    Smart word removal - removes only common function words
    that don't carry core meaning, while keeping content words
    """
    # Words that are truly just grammatical markers
    # (Be careful not to remove too much!)
    remove_words = {
        "am", "is", "are", "was", "were",  # Be verb
        "a", "an",                          # Articles
        "the",                              # Definite article
    }
    
    # Keep: "to", "for", "of", "in", "on" - these often have meaning in ASL
    # Keep conjunctions - they can be signed
    
    words = text.split()
    kept_words = []
    
    for i, w in enumerate(words):
        # Always remove articles/be-verbs
        if w in remove_words:
            continue
        # Keep everything else
        kept_words.append(w)
    
    return " ".join(kept_words)


# ---------------------------
# ✅ IMPROVED: Fuzzy matching for unknown words
# ---------------------------

def find_closest_match(word, gloss_map, max_distance=2):
    """
    Find closest matching gloss using simple edit distance.
    Helps with typos and variations.
    """
    from difflib import get_close_matches
    
    matches = get_close_matches(word, gloss_map.keys(), n=1, cutoff=0.6)
    if matches:
        return gloss_map[matches[0]]
    return None


# ---------------------------
# ✅ IMPROVED: Multi-level inference
# ---------------------------

def infer(text):
    """
    Multi-level inference strategy:
    1. Apply ASL grammar preprocessing (reorder words)
    2. Try exact phrase match (fastest)
    3. Try word-by-word with intelligent resolution (lemmatization/stemming)
    4. Fallback to uppercase (fingerspelling)
    """
    
    if not text:
        return "(empty)"
    
    norm = normalize_english(text)
    
    # ✅ NEW: Apply ASL grammar preprocessing to reorder words
    # This converts English word order to ASL-friendly structure
    asl_ordered = convert_to_asl_order(norm, sign_dict=None, debug=False)
    
    # Use the reordered ASL-friendly version for lookup
    simple = simplify_text(asl_ordered)
    
    # ✅ LEVEL 1: Exact sentence / phrase match
    if simple in DAILY_GLOSS_MAP:
        return DAILY_GLOSS_MAP[simple] + "  (exact + ASL grammar)"
    
    # ✅ LEVEL 2: Word-by-word with intelligent resolution (exact → lemmatization → stemming → fingerspell)
    gloss_words = []
    unmatched_words = []
    fallback_types = []  # Track which fallback was used
    
    for w in simple.split():
        # ✅ Try intelligent resolution: exact → lemmatization → stemming
        resolved_gloss = resolve_word(w, DAILY_GLOSS_MAP, debug=False)
        
        if resolved_gloss:
            # Successfully resolved (through exact, lemma, or stem)
            gloss_words.append(resolved_gloss)
            fallback_types.append("resolved")
        else:
            # Not found in any form - fall back to fingerspelling
            unmatched_words.append(w)
            gloss_words.append(w.upper())
            fallback_types.append("fingerspell")
    
    result = " ".join(gloss_words)
    
    # Add note about which words fell back to fingerspelling vs lemmatization/stemming
    if unmatched_words:
        result += f"  (unmatched: {','.join(unmatched_words)}, ASL gram + lemmatization)"
    else:
        result += "  (full match: ASL grammar + lemmatization + word-by-word)"
    
    return result


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
