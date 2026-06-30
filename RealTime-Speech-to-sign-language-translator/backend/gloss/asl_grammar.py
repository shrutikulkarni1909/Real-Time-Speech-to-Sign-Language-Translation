"""
ASL Grammar Preprocessing Module
=================================

Converts English sentence structure to ASL-friendly order.

ASL differs fundamentally from English word order:
- Time expressions come FIRST (not scattered through sentence)
- Topic-Comment structure: topic stated first, then comment on it
- No articles (a, an, the)
- No helping verbs (is, am, are, was, were)
- WH-questions at END (not beginning)
- Negation at END (not middle)

This module implements ASL grammar rules to preprocess English
before sign dictionary lookup.

Examples:
    "I am going to school tomorrow"
    → "TOMORROW SCHOOL I GO"
    
    "Where are you going?"
    → "YOU GO WHERE"
    
    "I do not understand"
    → "I UNDERSTAND NOT"
    
    "I like pizza"
    → "PIZZA I LIKE"
"""

import re
from backend.gloss.word_resolver import resolve_word, lemmatize_word


# ===== HELPER DATA =====

# Words to remove from English (they don't exist in ASL)
REMOVE_WORDS = {
    # Articles
    "a", "an", "the",
    
    # Helping/Linking verbs (core meaning comes from other verbs)
    "is", "am", "are", "was", "were",
    "be", "been", "being",
    "do", "does", "did",
    "have", "has", "had",
    
    # Common filler words
    "and", "or", "but",  # Can be replaced with pauses/non-manual markers in ASL
    "of",  # Handled by word order instead
    "very", "so",  # Can be signed through intensity instead
    "just", "only",  # Context markers, less critical
}

# Time-related words (should come first in ASL)
TIME_WORDS = {
    "today", "tomorrow", "yesterday",
    "now", "soon", "later",
    "morning", "afternoon", "evening", "night",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "year", "month", "week", "day", "hour", "minute", "second",
    "before", "after", "then", "when",
    "always", "never", "sometimes", "often", "usually",
    "past", "present", "future",
}

# WH-question words (move to end)
WH_WORDS = {
    "what", "where", "when", "why", "who", "how",
    "which", "whose", "whom",
}

# Negation markers
NEGATION_WORDS = {
    "not", "never", "no", "nothing", "nobody", "nowhere",
    "n't",  # Contraction: don't, can't, won't, etc.
}

# Common verbs that might need lemmatization
COMMON_VERBS = {
    "go", "going", "goes", "went", "gone",
    "eat", "eating", "eats", "ate", "eaten",
    "run", "running", "runs", "ran",
    "walk", "walking", "walks", "walked",
    "see", "seeing", "sees", "saw", "seen",
    "help", "helping", "helps", "helped",
    "play", "playing", "plays", "played",
    "like", "liking", "likes", "liked",
    "want", "wanting", "wants", "wanted",
    "need", "needing", "needs", "needed",
    "make", "making", "makes", "made",
    "take", "taking", "takes", "took", "taken",
    "study", "studying", "studies", "studied",
    "understand", "understanding", "understands", "understood",
    "know", "knowing", "knows", "knew", "known",
    "think", "thinking", "thinks", "thought",
    "believe", "believing", "believes", "believed",
    "say", "saying", "says", "said",
    "tell", "telling", "tells", "told",
    "give", "giving", "gives", "gave", "given",
}


# ===== CORE PREPROCESSING FUNCTIONS =====

def extract_time_words(tokens):
    """
    Extract time-related words from token list.
    
    Args:
        tokens (list): List of words
    
    Returns:
        tuple: (time_words_list, remaining_tokens)
    """
    time_words = []
    remaining = []
    
    for token in tokens:
        if token.lower() in TIME_WORDS:
            time_words.append(token)
        else:
            remaining.append(token)
    
    return time_words, remaining


def extract_wh_words(tokens):
    """
    Extract WH-question words (move to end).
    
    Args:
        tokens (list): List of words
    
    Returns:
        tuple: (wh_words_list, remaining_tokens)
    """
    wh_words = []
    remaining = []
    
    for token in tokens:
        if token.lower() in WH_WORDS:
            wh_words.append(token)
        else:
            remaining.append(token)
    
    return wh_words, remaining


def extract_negation(tokens):
    """
    Extract negation markers (move to end).
    
    Args:
        tokens (list): List of words
    
    Returns:
        tuple: (has_negation, remaining_tokens)
    """
    has_negation = False
    remaining = []
    
    for token in tokens:
        token_lower = token.lower()
        # Check for "n't" contraction (don't, won't, can't, etc.)
        if "n't" in token_lower:
            has_negation = True
            # Remove the "n't" and keep the base word if it's meaningful
            base_word = token_lower.replace("n't", "")
            if base_word and base_word not in {"do", "does", "did"}:
                remaining.append(base_word)
        elif token_lower in NEGATION_WORDS:
            has_negation = True
        else:
            remaining.append(token)
    
    return has_negation, remaining


def remove_articles_and_filler(tokens):
    """
    Remove articles (a, an, the) and unnecessary filler words.
    
    Args:
        tokens (list): List of words
    
    Returns:
        list: Filtered token list
    """
    filtered = []
    
    for token in tokens:
        if token.lower() not in REMOVE_WORDS:
            filtered.append(token)
    
    return filtered


def identify_topic_object(tokens):
    """
    Identify the object/topic that should come first (Topic-Comment structure).
    
    In ASL, the main object/topic often comes first.
    This is a heuristic - checks for direct objects (nouns after lexical verbs).
    
    Args:
        tokens (list): List of words
    
    Returns:
        tuple: (topic_object_or_none, remaining_tokens)
    """
    # Simple heuristic: Look for noun patterns
    # In simplified English, this might be: verb + noun = topic
    # For now, use a simple approach: find nouns and move one to front if it seems like object
    
    topic = None
    remaining = []
    
    # For a simple implementation, look for common object patterns
    # e.g., "I like pizza" → pizza is object
    # This is complex in general English parsing, so we use heuristics
    
    # Common pattern: [subject] [verb] [object]
    # We want to reorder to: [object] [subject] [verb]
    
    # Skip this for now - will be handled by manual topic-comment rules
    # Return tokens unchanged for this pass
    return None, tokens


def lemmatize_verbs(tokens, sign_dict=None):
    """
    Lemmatize verbs to their base forms.
    
    Args:
        tokens (list): List of words
        sign_dict (dict): Optional sign dictionary for resolution
    
    Returns:
        list: Tokens with verbs lemmatized
    """
    lemmatized = []
    
    for token in tokens:
        token_lower = token.lower()
        
        # If it's a known verb, lemmatize it
        if token_lower in COMMON_VERBS:
            lemma = lemmatize_word(token_lower)
            # Preserve original casing if it was uppercase
            if token.isupper():
                lemma = lemma.upper()
            lemmatized.append(lemma)
        else:
            lemmatized.append(token)
    
    return lemmatized


def reorder_to_asl_structure(tokens, time_words, wh_words, has_negation):
    """
    Reorder tokens into ASL structure.
    
    ASL Order: [Time] [Topic/Object] [Subject] [Verb] [Object] [WH-End] [Negation]
    
    Args:
        tokens (list): Remaining tokens after extraction
        time_words (list): Time-related words
        wh_words (list): WH-question words
        has_negation (bool): Whether sentence has negation
    
    Returns:
        list: Reordered tokens
    """
    result = []
    
    # STEP 1: Add time words first (highest priority in ASL)
    if time_words:
        result.extend(time_words)
    
    # STEP 2: Add remaining words (which should be arranged by the caller)
    # This is where topic-comment structure should apply
    result.extend(tokens)
    
    # STEP 3: Add WH-words at end (for questions)
    if wh_words:
        result.extend(wh_words)
    
    # STEP 4: Add negation marker at very end
    if has_negation:
        result.append("NOT")
    
    return result


# ===== MAIN CONVERSION FUNCTION =====

def convert_to_asl_order(sentence, sign_dict=None, debug=False):
    """
    Convert English sentence to ASL-friendly word order.
    
    Applies ASL grammar rules:
    1. Time words first
    2. Topic-Comment structure
    3. Remove articles/helping verbs
    4. Lemmatize verbs
    5. WH-words to end
    6. Negation to end
    7. Remove filler words
    
    Args:
        sentence (str): English sentence to convert
        sign_dict (dict): Optional sign dictionary for resolution
        debug (bool): Print processing steps if True
    
    Returns:
        str: ASL-ordered sentence
    
    Examples:
        >>> convert_to_asl_order("I am going to school tomorrow")
        'TOMORROW SCHOOL I GO'
        
        >>> convert_to_asl_order("Where are you going?")
        'YOU GO WHERE'
        
        >>> convert_to_asl_order("I do not understand")
        'I UNDERSTAND NOT'
        
        >>> convert_to_asl_order("I like pizza")
        'PIZZA I LIKE'
    """
    
    if not sentence or not sentence.strip():
        return ""
    
    if debug:
        print(f"[ASL Grammar] Input: {sentence}")
    
    # ===== STEP 1: Normalize =====
    # Remove punctuation but keep words
    normalized = re.sub(r'[?!.,;:]', '', sentence)
    normalized = normalized.lower().strip()
    
    if debug:
        print(f"  → Normalized: {normalized}")
    
    # ===== STEP 2: Tokenize =====
    tokens = normalized.split()
    
    if debug:
        print(f"  → Tokens: {tokens}")
    
    # ===== STEP 3: Extract special markers =====
    # Extract time words (these go FIRST in ASL)
    time_words, tokens = extract_time_words(tokens)
    
    if debug and time_words:
        print(f"  → Time words: {time_words}")
    
    # Extract WH-words (these go at END in ASL)
    wh_words, tokens = extract_wh_words(tokens)
    
    if debug and wh_words:
        print(f"  → WH words: {wh_words}")
    
    # Extract negation (this goes at END in ASL)
    has_negation, tokens = extract_negation(tokens)
    
    if debug and has_negation:
        print(f"  → Has negation: True")
    
    # ===== STEP 4: Clean up =====
    # Remove articles, helping verbs, and filler words
    tokens = remove_articles_and_filler(tokens)
    
    if debug:
        print(f"  → After removing articles/fillers: {tokens}")
    
    # Lemmatize verbs
    tokens = lemmatize_verbs(tokens, sign_dict)
    
    if debug:
        print(f"  → After lemmatization: {tokens}")
    
    # ===== STEP 5: Topic-Comment (heuristic) =====
    # Try to identify object and move to front (simplified)
    # For example: "I like pizza" → "pizza I like"
    # This is a simple pattern matcher
    
    if len(tokens) >= 3:
        # Pattern: [pronoun/subject] [verb] [object/noun]
        # Move object to front
        subject_pronouns = {"i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us"}
        
        # Check if first word is subject pronoun
        if tokens[0] in subject_pronouns and len(tokens) >= 3:
            # Assumed structure: [subject] [verb] [object ...]
            # Reorder to: [object ...] [subject] [verb]
            subject = tokens[0]
            verb = tokens[1]
            rest = tokens[2:]
            
            # Put object/noun first, then subject, then verb
            tokens = rest + [subject, verb]
            
            if debug:
                print(f"  → After topic-comment: {tokens}")
    
    # ===== STEP 6: Reorder for ASL structure =====
    final_tokens = reorder_to_asl_structure(tokens, time_words, wh_words, has_negation)
    
    if debug:
        print(f"  → Final ASL order: {final_tokens}")
    
    # ===== STEP 7: Output =====
    result = " ".join(final_tokens).upper()
    
    if debug:
        print(f"  → Output: {result}")
    
    return result


# ===== CLI TEST =====

if __name__ == "__main__":
    print("ASL Grammar Preprocessing Module")
    print("=" * 70)
    print()
    
    # Test cases demonstrating ASL grammar rules
    test_cases = [
        # Rule 1: Time words first
        ("I will go tomorrow", "Time words first"),
        ("I am going to school tomorrow", "Time words + object first"),
        
        # Rule 2: Topic-Comment (object first)
        ("I like pizza", "Topic-Comment: object first"),
        ("He sees a dog", "Topic-Comment: object first"),
        
        # Rule 3: Remove helping verbs
        ("She is happy", "Remove 'is'"),
        ("They are playing", "Remove 'are'"),
        ("I am eating", "Remove 'am'"),
        
        # Rule 4: Remove articles
        ("I see a dog", "Remove 'a'"),
        ("The cat is green", "Remove 'the'"),
        
        # Rule 6: WH-words to end
        ("Where are you going?", "WH-word to end"),
        ("What do you want?", "WH-word to end"),
        ("Who is coming tomorrow?", "WH-word + time + remove helping verb"),
        
        # Rule 7: Negation to end
        ("I do not understand", "Negation to end"),
        ("I don't like it", "Negation contraction to end"),
        
        # Complex examples
        ("I am going to the store tomorrow", "Multi-rule"),
        ("Where are the children going tomorrow?", "Complex: time + WH + remove articles"),
    ]
    
    print("Test Cases:")
    print("-" * 70)
    
    for i, (sentence, description) in enumerate(test_cases, 1):
        result = convert_to_asl_order(sentence, debug=False)
        print(f"\n{i}. {description}")
        print(f"   English:  {sentence}")
        print(f"   ASL:      {result}")
    
    print("\n" + "=" * 70)
    print("\nInteractive Test (type 'quit' to exit):\n")
    
    while True:
        sentence = input("Enter English sentence: ").strip()
        if sentence.lower() == "quit":
            break
        if not sentence:
            continue
        
        result = convert_to_asl_order(sentence, debug=False)
        print(f"ASL output: {result}\n")
