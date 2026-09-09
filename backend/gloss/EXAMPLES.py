"""
CODE EXAMPLES: Using Lemmatization & Stemming
==============================================

This file shows various ways to use the word resolution system
in your speech-to-sign pipeline.
"""

# ============================================================================
# EXAMPLE 1: Basic Word Resolution
# ============================================================================

def example_basic():
    """Resolve a single word"""
    from backend.gloss.word_resolver import resolve_word
    
    # Your sign dictionary
    sign_dict = {
        'eat': 'EAT',
        'run': 'RUN',
        'study': 'STUDY',
    }
    
    # Resolve inflected words
    print("Basic Resolution:")
    print(f"  eating → {resolve_word('eating', sign_dict)}")    # EAT
    print(f"  studies → {resolve_word('studies', sign_dict)}")  # STUDY
    print(f"  running → {resolve_word('running', sign_dict)}")  # RUN
    print(f"  unknown → {resolve_word('unknown', sign_dict)}")  # None


# ============================================================================
# EXAMPLE 2: Full Pipeline Integration (Already Done!)
# ============================================================================

def example_pipeline():
    """Use the full inference pipeline (no code needed!)"""
    from backend.gloss.infer import infer
    
    print("\nFull Pipeline:")
    
    # This already uses word resolution internally!
    result1 = infer("I am eating bread")
    print(f"  'I am eating bread' → {result1}")
    
    result2 = infer("The children play")
    print(f"  'The children play' → {result2}")
    
    result3 = infer("He studies and runs")
    print(f"  'He studies and runs' → {result3}")


# ============================================================================
# EXAMPLE 3: Batch Processing with Lemmatization
# ============================================================================

def example_batch_resolution():
    """Process multiple words at once"""
    from backend.gloss.word_resolver import resolve_word
    
    sign_dict = {
        'eat': 'EAT',
        'jump': 'JUMP',
        'help': 'HELP',
        'walk': 'WALK',
    }
    
    words_from_speech = ['eating', 'jumped', 'helping', 'walked', 'unknown']
    
    print("\nBatch Resolution:")
    for word in words_from_speech:
        resolved = resolve_word(word, sign_dict)
        if resolved:
            print(f"  ✓ {word:12} → {resolved}")
        else:
            print(f"  ✗ {word:12} → (unknown, will fingerspell)")


# ============================================================================
# EXAMPLE 4: With Debug Output
# ============================================================================

def example_debug():
    """See the resolution process step-by-step"""
    from backend.gloss.word_resolver import resolve_word
    
    sign_dict = {
        'eat': 'EAT',
        'run': 'RUN',
    }
    
    print("\nDebug Output:")
    print("\nResolving 'eating':")
    result = resolve_word('eating', sign_dict, debug=True)
    print(f"Final result: {result}\n")
    
    print("Resolving 'unknown':")
    result = resolve_word('unknown', sign_dict, debug=True)
    print(f"Final result: {result}")


# ============================================================================
# EXAMPLE 5: Custom Function Wrapping
# ============================================================================

def example_custom_wrapper():
    """Create a custom wrapper for your specific use case"""
    from backend.gloss.word_resolver import resolve_word
    
    # Load your sign dictionary
    SIGN_DICT = {
        'eat': 'EAT',
        'drink': 'DRINK',
        'run': 'RUN',
        'sleep': 'SLEEP',
    }
    
    def get_sign_for_word(word):
        """
        Get the sign for a word, with lemmatization fallback.
        Returns the sign, or None if not found.
        """
        sign = resolve_word(word, SIGN_DICT, debug=False)
        
        if sign:
            print(f"✓ Found sign for '{word}': {sign}")
            return sign
        else:
            print(f"✗ No sign for '{word}', will fingerspell: {word.upper()}")
            return None
    
    print("\nCustom Wrapper:")
    get_sign_for_word('drinking')    # → DRINK (lemmatized)
    get_sign_for_word('eaten')       # → EAT (lemmatized)
    get_sign_for_word('sleeping')    # → SLEEP (lemmatized)
    get_sign_for_word('xyz')         # → None (unknown)


# ============================================================================
# EXAMPLE 6: Integration with Whisper Output
# ============================================================================

def example_whisper_integration():
    """Process real Whisper transcription output"""
    from backend.gloss.infer import infer
    
    # Simulated Whisper output
    whisper_texts = [
        "I was eating pizza",
        "The children are playing",
        "She studies every day",
        "Running is fun",
    ]
    
    print("\nWhisper → Gloss Pipeline:")
    for text in whisper_texts:
        gloss = infer(text)
        print(f"\n  Speech: '{text}'")
        print(f"  Gloss:  {gloss}")


# ============================================================================
# EXAMPLE 7: Performance Comparison
# ============================================================================

def example_performance():
    """Compare performance with and without lemmatization"""
    import time
    from backend.gloss.word_resolver import resolve_word
    
    sign_dict = {
        'eat': 'EAT',
        'run': 'RUN',
        'study': 'STUDY',
    }
    
    inflected_words = [
        'eating', 'running', 'studies', 'studied', 'studying'
    ] * 20  # 100 words total
    
    print("\nPerformance Test:")
    print(f"  Resolving {len(inflected_words)} words...")
    
    start = time.time()
    for word in inflected_words:
        resolve_word(word, sign_dict)
    elapsed = time.time() - start
    
    avg_time = (elapsed / len(inflected_words)) * 1000
    print(f"  Total time: {elapsed:.3f}s")
    print(f"  Per word: {avg_time:.2f}ms")


# ============================================================================
# EXAMPLE 8: Error Handling
# ============================================================================

def example_error_handling():
    """Handle edge cases gracefully"""
    from backend.gloss.word_resolver import resolve_word
    
    sign_dict = {'eat': 'EAT', 'run': 'RUN'}
    
    print("\nError Handling:")
    
    # Empty string
    result = resolve_word('', sign_dict)
    print(f"  Empty string: {result}")  # None
    
    # None input (would fail)
    try:
        result = resolve_word(None, sign_dict)
    except Exception as e:
        print(f"  None input: {type(e).__name__}")
    
    # Special characters
    result = resolve_word('eating!', sign_dict)
    print(f"  'eating!': {result}")  # EAT (punctuation removed)
    
    # Very long word
    result = resolve_word('supercalifragilisticexpialidocious', sign_dict)
    print(f"  Very long word: {result}")  # None


# ============================================================================
# EXAMPLE 9: Building a Sign Lookup Cache
# ============================================================================

def example_cache_building():
    """Pre-compute lemmatizations for faster runtime"""
    from backend.gloss.word_resolver import resolve_word, lemmatize_word, stem_word
    
    sign_dict = {
        'eat': 'EAT',
        'run': 'RUN',
        'study': 'STUDY',
        'help': 'HELP',
    }
    
    print("\nBuilding Lemmatization Cache:")
    
    # Build a cache of common inflections
    cache = {}
    inflections = [
        'eating', 'ate', 'eaten',      # eat variants
        'running', 'ran',               # run variants
        'studying', 'studied',          # study variants
        'helping', 'helped',            # help variants
    ]
    
    for word in inflections:
        resolved = resolve_word(word, sign_dict)
        if resolved:
            cache[word] = resolved
            print(f"  Cache: {word:15} → {resolved}")
    
    # Now lookup is instant
    print(f"\nCache size: {len(cache)} entries")
    print(f"  eating → {cache.get('eating')}")  # Instant


# ============================================================================
# EXAMPLE 10: Custom Fallback Strategy
# ============================================================================

def example_custom_fallback():
    """Implement custom fallback logic"""
    from backend.gloss.word_resolver import resolve_word
    
    SIGN_DICT = {
        'eat': 'EAT',
        'run': 'RUN',
        'happy': 'HAPPY',
    }
    
    # Custom lookup with additional fallback
    def smart_word_lookup(word):
        # Try normal resolution first
        result = resolve_word(word, SIGN_DICT)
        if result:
            return result
        
        # If that fails, try some custom rules
        if 'ing' in word:
            # Might be a verb we don't recognize
            base = word.replace('ing', '')
            if base in SIGN_DICT:
                return SIGN_DICT[base]
        
        # If all fails, return uppercase for fingerspelling
        return word.upper()
    
    print("\nCustom Fallback:")
    print(f"  eating: {smart_word_lookup('eating')}")       # EAT
    print(f"  feeling: {smart_word_lookup('feeling')}")     # FEELING (no 'feel')
    print(f"  running: {smart_word_lookup('running')}")     # RUN


# ============================================================================
# MAIN: Run all examples
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("CODE EXAMPLES: LEMMATIZATION & STEMMING")
    print("=" * 70)
    
    example_basic()
    example_pipeline()
    example_batch_resolution()
    example_debug()
    example_custom_wrapper()
    example_whisper_integration()
    example_performance()
    example_error_handling()
    example_cache_building()
    example_custom_fallback()
    
    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)
