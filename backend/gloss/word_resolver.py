"""
Word Resolution Module with Lemmatization & Stemming Fallback
===============================================================

This module handles word-to-gloss mapping with intelligent fallback strategies:
1. Exact match (fastest)
2. Lemmatization (handles inflected forms like eat/eating, study/studies)
3. Stemming (handles different word forms)
4. None (mark as unknown for fingerspelling)

Usage:
    from backend.gloss.word_resolver import resolve_word
    
    sign_dict = {"eat": "EAT", "run": "RUN", "study": "STUDY"}
    resolve_word("eating", sign_dict)    # Returns "EAT"
    resolve_word("studies", sign_dict)   # Returns "STUDY"
    resolve_word("running", sign_dict)   # Returns "RUN"
"""

import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import wordnet
import re

# ✅ Initialize NLTK resources (downloads on first run)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('corpora/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)

# ✅ Initialize NLTK tools
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


IRREGULAR_VERB_LEMMAS = {
    "bought": "buy",
    "brought": "bring",
    "thought": "think",
    "taught": "teach",
    "caught": "catch",
    "went": "go",
    "gone": "go",
    "seen": "see",
    "saw": "see",
    "had": "have",
    "made": "make",
    "come": "come",
    "gave": "give",
    "taken": "take",
    "held": "hold",
    "found": "find",
    "kept": "keep",
    "felt": "feel",
    "left": "leave",
    "heard": "hear",
    "put": "put",
    "read": "read",
    "wrote": "write",
    "ran": "run",
    "sat": "sit",
    "stood": "stand",
    "lost": "lose",
}


def get_wordnet_pos(word):
    """
    Map POS tag to WordNet POS tag for accurate lemmatization.
    
    Args:
        word (str): The word to get POS for
        
    Returns:
        str: WordNet POS tag (wordnet.NOUN, wordnet.VERB, etc.)
    """
    try:
        tag = nltk.pos_tag([word])[0][1][0].upper()
        return {
            "J": wordnet.ADJ,
            "V": wordnet.VERB,
            "N": wordnet.NOUN,
            "R": wordnet.ADV,
        }.get(tag, wordnet.VERB)
    except Exception:
        return wordnet.VERB


def clean_word(word):
    """
    Clean and prepare word for processing:
    - Convert to lowercase
    - Remove non-alphanumeric characters
    - Strip whitespace
    
    Args:
        word (str): Raw word from text
        
    Returns:
        str: Cleaned word
    """
    if not word:
        return ""
    
    # Convert to lowercase
    word = word.lower()
    
    # Remove non-alphanumeric characters
    word = re.sub(r"[^a-z0-9]", "", word)
    
    # Strip whitespace
    word = word.strip()
    
    return word


def lemmatize_word(word):
    """
    Apply lemmatization to a word.
    Converts inflected forms to base form (e.g., eating → eat).
    
    Args:
        word (str): Cleaned word to lemmatize
        
    Returns:
        str: Lemmatized form of the word
    """
    if not word:
        return word
    
    # First check common irregular verbs
    if word in IRREGULAR_VERB_LEMMAS:
        return IRREGULAR_VERB_LEMMAS[word]
    
    try:
        pos = get_wordnet_pos(word)
        lemma = lemmatizer.lemmatize(word, pos=pos)
        
        # If POS-specific lemma did not change, try alternative tags
        if lemma == word:
            if pos != wordnet.VERB:
                lemma_verb = lemmatizer.lemmatize(word, pos=wordnet.VERB)
                if lemma_verb != word:
                    lemma = lemma_verb
            if lemma == word and pos != wordnet.NOUN:
                lemma_noun = lemmatizer.lemmatize(word, pos=wordnet.NOUN)
                if lemma_noun != word:
                    lemma = lemma_noun
        
        return lemma
    except Exception as e:
        print(f"  [lemmatization error for '{word}': {e}]")
        return word


def stem_word(word):
    """
    Apply Porter stemming to a word.
    More aggressive than lemmatization (e.g., running → run, studies → studi).
    
    Args:
        word (str): Cleaned word to stem
        
    Returns:
        str: Stemmed form of the word
    """
    if not word:
        return word
    
    try:
        return stemmer.stem(word)
    except Exception as e:
        print(f"  [stemming error for '{word}': {e}]")
        return word


def resolve_word(word, sign_dict, debug=False):
    """
    Intelligent word-to-sign resolution with fallback strategy.
    
    Resolution order:
    1. Exact match (word exists as-is in dictionary)
    2. Lemmatization (convert to base form, e.g., eating → eat)
    3. Stemming (extract root, e.g., running → run)
    4. None (not found)
    
    Args:
        word (str): The word to resolve (can be raw or pre-cleaned)
        sign_dict (dict): Dictionary mapping words to signs/glosses
                         Keys should be lowercase
        debug (bool): If True, print resolution steps
        
    Returns:
        str or None: The corresponding sign/gloss, or None if not found
        
    Examples:
        >>> sign_dict = {"eat": "EAT", "run": "RUN", "study": "STUDY"}
        >>> resolve_word("eating", sign_dict)
        'EAT'
        >>> resolve_word("studies", sign_dict)
        'STUDY'
        >>> resolve_word("running", sign_dict)
        'RUN'
        >>> resolve_word("unknown", sign_dict)
        None
    """
    
    # ===== STEP 1: Clean and normalize the word =====
    original_word = word
    word = clean_word(word)
    
    if not word:
        if debug:
            print(f"  [resolve_word] '{original_word}' → empty after cleaning")
        return None
    
    if debug:
        print(f"  [resolve_word] Processing: '{original_word}' → '{word}'")
    
    # ===== STEP 2: Try EXACT MATCH (fastest) =====
    if word in sign_dict:
        if debug:
            print(f"    ✓ EXACT MATCH: '{word}' → '{sign_dict[word]}'")
        return sign_dict[word]
    
    if debug:
        print(f"    ✗ No exact match for '{word}'")
    
    # ===== STEP 3: Try LEMMATIZATION =====
    lemma = lemmatize_word(word)
    if lemma != word and lemma in sign_dict:
        if debug:
            print(f"    ✓ LEMMATIZATION: '{word}' → '{lemma}' → '{sign_dict[lemma]}'")
        return sign_dict[lemma]
    
    if debug and lemma != word:
        print(f"    ✗ Lemmatization tried '{lemma}', not in dict")
    
    # ===== STEP 4: Try STEMMING =====
    stem = stem_word(word)
    if stem != word and stem in sign_dict:
        if debug:
            print(f"    ✓ STEMMING: '{word}' → '{stem}' → '{sign_dict[stem]}'")
        return sign_dict[stem]
    
    if debug and stem != word:
        print(f"    ✗ Stemming tried '{stem}', not in dict")
    
    # ===== STEP 5: NOT FOUND =====
    if debug:
        print(f"    ✗ UNRESOLVED: '{original_word}' (tried: exact, lemma={lemma}, stem={stem})")
    
    return None


# ===== CLI for Testing =====
if __name__ == "__main__":
    print("Word Resolver Test Suite")
    print("=" * 60)
    
    # Test dictionary
    test_dict = {
        "eat": "EAT",
        "run": "RUN",
        "study": "STUDY",
        "child": "CHILD",
        "go": "GO",
        "jump": "JUMP",
        "play": "PLAY",
        "walk": "WALK",
        "talk": "TALK",
        "help": "HELP",
    }
    
    # Test cases
    test_cases = [
        ("eating", "EAT"),          # Inflected verb (past participle)
        ("studies", "STUDY"),       # Inflected verb (3rd person singular)
        ("running", "RUN"),         # Present participle
        ("children", "CHILD"),      # Plural noun
        ("goes", "GO"),             # Inflected verb
        ("jumped", "JUMP"),         # Past tense
        ("playing", "PLAY"),        # Present participle
        ("walked", "WALK"),         # Past tense
        ("talking", "TALK"),        # Present participle
        ("helped", "HELP"),         # Past tense
        ("unknown_word", None),     # Word not in dict
        ("xyz", None),              # Random word
    ]
    
    print("\nRunning resolution tests:\n")
    passed = 0
    failed = 0
    
    for input_word, expected_sign in test_cases:
        result = resolve_word(input_word, test_dict, debug=False)
        status = "✓ PASS" if result == expected_sign else "✗ FAIL"
        
        print(f"{status} | {input_word:15} → {str(result):15} (expected: {expected_sign})")
        
        if result == expected_sign:
            passed += 1
        else:
            failed += 1
    
    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    
    # Interactive test
    print(f"\n{'=' * 60}")
    print("Interactive Test (type 'quit' to exit):")
    print(f"Dictionary keys: {list(test_dict.keys())}\n")
    
    while True:
        word = input("Enter word to resolve: ").strip()
        if word.lower() == "quit":
            break
        if not word:
            continue
        
        result = resolve_word(word, test_dict, debug=True)
        print(f"  → Result: {result}\n")
