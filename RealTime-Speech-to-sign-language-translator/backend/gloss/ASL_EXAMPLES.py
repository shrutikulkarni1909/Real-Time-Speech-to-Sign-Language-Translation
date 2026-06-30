#!/usr/bin/env python3
"""
ASL GRAMMAR EXAMPLES
====================

This file demonstrates various ways to use the ASL grammar preprocessing
module in your speech-to-sign pipeline.
"""

# ============================================================================
# EXAMPLE 1: Basic ASL Grammar Conversion
# ============================================================================

def example_basic_conversion():
    """Simple conversion of English to ASL word order"""
    from backend.gloss.asl_grammar import convert_to_asl_order
    
    print("EXAMPLE 1: Basic ASL Conversion")
    print("-" * 70)
    
    examples = [
        "I am happy",
        "She is sad",
        "We are playing",
        "They like pizza",
    ]
    
    for sentence in examples:
        result = convert_to_asl_order(sentence)
        print(f"  {sentence:30} → {result}")
    
    print()


# ============================================================================
# EXAMPLE 2: Time Words
# ============================================================================

def example_time_words():
    """Time words are moved to the beginning"""
    from backend.gloss.asl_grammar import convert_to_asl_order
    
    print("EXAMPLE 2: Time Words First")
    print("-" * 70)
    
    examples = [
        "I will go tomorrow",
        "We are playing today",
        "She went yesterday",
        "They will study next week",
    ]
    
    for sentence in examples:
        result = convert_to_asl_order(sentence)
        print(f"  {sentence:35} → {result}")
    
    print()


# ============================================================================
# EXAMPLE 3: Topic-Comment Structure
# ============================================================================

def example_topic_comment():
    """Main object/topic comes first"""
    from backend.gloss.asl_grammar import convert_to_asl_order
    
    print("EXAMPLE 3: Topic-Comment Structure")
    print("-" * 70)
    
    examples = [
        "I like pizza",
        "She loves coffee",
        "They bought a house",
        "I see a dog",
    ]
    
    for sentence in examples:
        result = convert_to_asl_order(sentence)
        print(f"  {sentence:30} → {result}")
    
    print()


# ============================================================================
# EXAMPLE 4: Removing Helping Verbs
# ============================================================================

def example_remove_helping_verbs():
    """Helping verbs (is, am, are, was, were) are removed"""
    from backend.gloss.asl_grammar import convert_to_asl_order
    
    print("EXAMPLE 4: Remove Helping Verbs")
    print("-" * 70)
    
    examples = [
        "She is happy",
        "They are playing",
        "I am going",
        "He was running",
    ]
    
    for sentence in examples:
        result = convert_to_asl_order(sentence)
        print(f"  {sentence:30} → {result}")
    
    print()


# ============================================================================
# EXAMPLE 5: Removing Articles
# ============================================================================

def example_remove_articles():
    """Articles (a, an, the) are removed"""
    from backend.gloss.asl_grammar import convert_to_asl_order
    
    print("EXAMPLE 5: Remove Articles")
    print("-" * 70)
    
    examples = [
        "I see a dog",
        "The cat is green",
        "She has an apple",
        "I am in the house",
    ]
    
    for sentence in examples:
        result = convert_to_asl_order(sentence)
        print(f"  {sentence:30} → {result}")
    
    print()


# ============================================================================
# EXAMPLE 6: WH-Questions
# ============================================================================

def example_wh_questions():
    """WH-words (what, where, when, why, who) move to the end"""
    from backend.gloss.asl_grammar import convert_to_asl_order
    
    print("EXAMPLE 6: WH-Questions (WH-words at End)")
    print("-" * 70)
    
    examples = [
        "Where are you going?",
        "What do you want?",
        "Who is coming?",
        "When will you arrive?",
        "Why are you laughing?",
    ]
    
    for sentence in examples:
        result = convert_to_asl_order(sentence)
        print(f"  {sentence:35} → {result}")
    
    print()


# ============================================================================
# EXAMPLE 7: Negation
# ============================================================================

def example_negation():
    """Negation words/markers move to the end"""
    from backend.gloss.asl_grammar import convert_to_asl_order
    
    print("EXAMPLE 7: Negation at End")
    print("-" * 70)
    
    examples = [
        "I do not like it",
        "She doesn't understand",
        "They don't have time",
        "I have not seen it",
    ]
    
    for sentence in examples:
        result = convert_to_asl_order(sentence)
        print(f"  {sentence:35} → {result}")
    
    print()


# ============================================================================
# EXAMPLE 8: Combined Rules
# ============================================================================

def example_complex_sentences():
    """Complex sentences using multiple rules"""
    from backend.gloss.asl_grammar import convert_to_asl_order
    
    print("EXAMPLE 8: Complex Sentences (Multiple Rules)")
    print("-" * 70)
    
    examples = [
        "I am going to school tomorrow",
        "Where are you going tomorrow?",
        "The children are playing in the park",
        "She was eating an apple yesterday",
        "What do you study?",
    ]
    
    for sentence in examples:
        result = convert_to_asl_order(sentence)
        print(f"Input:  {sentence}")
        print(f"Output: {result}")
        print()


# ============================================================================
# EXAMPLE 9: Integration with Full Pipeline
# ============================================================================

def example_full_pipeline():
    """Use the complete inference pipeline (ASL grammar + lemmatization)"""
    from backend.gloss.infer import infer
    
    print("EXAMPLE 9: Full Pipeline (ASL Grammar + Lemmatization + Dictionary)")
    print("-" * 70)
    
    examples = [
        "I am going to school tomorrow",
        "Where are you going?",
        "The children are playing",
        "I do not like it",
        "She is eating pizza",
    ]
    
    for sentence in examples:
        result = infer(sentence)
        print(f"Input:  {sentence}")
        print(f"Output: {result}")
        print()


# ============================================================================
# EXAMPLE 10: Debug Output
# ============================================================================

def example_debug_output():
    """See the step-by-step processing"""
    from backend.gloss.asl_grammar import convert_to_asl_order
    
    print("EXAMPLE 10: Debug Output (Processing Steps)")
    print("-" * 70)
    
    sentences = [
        "I am going to school tomorrow",
        "Where are the children?",
    ]
    
    for sentence in sentences:
        print(f"\nConverting: {sentence}")
        result = convert_to_asl_order(sentence, debug=True)
        print(f"Final result: {result}\n")


# ============================================================================
# EXAMPLE 11: Custom Processing Function
# ============================================================================

def example_custom_pipeline():
    """Create a custom pipeline combining features"""
    from backend.gloss.asl_grammar import convert_to_asl_order
    from backend.gloss.word_resolver import resolve_word
    
    SIGN_DICT = {
        "I": "I",
        "you": "YOU",
        "like": "LIKE",
        "pizza": "PIZZA",
        "go": "GO",
        "school": "SCHOOL",
        "tomorrow": "TOMORROW",
    }
    
    def process_with_metrics(sentence):
        """Process sentence and show metrics"""
        # Apply ASL grammar
        asl_ordered = convert_to_asl_order(sentence)
        
        # Split into words
        words = asl_ordered.split()
        
        # Try to resolve each word
        resolved = []
        unresolved = []
        
        for word in words:
            sign = resolve_word(word, SIGN_DICT)
            if sign:
                resolved.append(sign)
            else:
                unresolved.append(word)
                resolved.append(word)  # Fingerspelling
        
        return {
            "original": sentence,
            "asl_order": asl_ordered,
            "glosses": " ".join(resolved),
            "resolved_count": len([w for w in words if resolve_word(w, SIGN_DICT)]),
            "total_words": len(words),
        }
    
    print("EXAMPLE 11: Custom Pipeline with Metrics")
    print("-" * 70)
    
    sentences = [
        "I like pizza",
        "I am going to school tomorrow",
        "You are going where?",
    ]
    
    for sentence in sentences:
        result = process_with_metrics(sentence)
        print(f"Original:  {result['original']}")
        print(f"ASL Order: {result['asl_order']}")
        print(f"Glosses:   {result['glosses']}")
        print(f"Resolved:  {result['resolved_count']}/{result['total_words']} words")
        print()


# ============================================================================
# EXAMPLE 12: Performance Benchmark
# ============================================================================

def example_performance():
    """Measure performance of ASL grammar conversion"""
    import time
    from backend.gloss.asl_grammar import convert_to_asl_order
    
    print("EXAMPLE 12: Performance Benchmark")
    print("-" * 70)
    
    sentences = [
        "I am going to school tomorrow",
        "Where are you going?",
        "The children are playing in the park",
    ] * 100  # 300 sentences total
    
    print(f"Processing {len(sentences)} sentences...")
    
    start = time.time()
    for sentence in sentences:
        convert_to_asl_order(sentence)
    elapsed = time.time() - start
    
    avg_time = (elapsed / len(sentences)) * 1000
    throughput = len(sentences) / elapsed
    
    print(f"Total time:     {elapsed:.3f}s")
    print(f"Per sentence:   {avg_time:.2f}ms")
    print(f"Throughput:     {throughput:.1f} sentences/sec")
    print()


# ============================================================================
# EXAMPLE 13: Batch Processing
# ============================================================================

def example_batch_processing():
    """Process multiple sentences efficiently"""
    from backend.gloss.infer import infer
    
    print("EXAMPLE 13: Batch Processing")
    print("-" * 70)
    
    conversation = [
        "Hello, how are you?",
        "I am doing great!",
        "Where are you going tomorrow?",
        "I am going to the store.",
        "What do you want to buy?",
        "I need some bread and milk.",
    ]
    
    for line in conversation:
        gloss = infer(line)
        print(f"Person: {line}")
        print(f"Sign:   {gloss}")
        print()


# ============================================================================
# MAIN: Run all examples
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ASL GRAMMAR PREPROCESSING - CODE EXAMPLES")
    print("=" * 70)
    print()
    
    example_basic_conversion()
    example_time_words()
    example_topic_comment()
    example_remove_helping_verbs()
    example_remove_articles()
    example_wh_questions()
    example_negation()
    example_complex_sentences()
    example_full_pipeline()
    example_debug_output()
    example_custom_pipeline()
    example_performance()
    example_batch_processing()
    
    print("=" * 70)
    print("All examples completed!")
    print("=" * 70)
