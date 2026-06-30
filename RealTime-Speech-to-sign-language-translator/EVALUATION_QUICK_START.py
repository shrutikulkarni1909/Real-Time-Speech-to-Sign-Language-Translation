#!/usr/bin/env python3
"""
QUICK START - EVALUATION SCRIPT
================================

This file shows the fastest way to get started with the evaluation.
"""

# STEP 1: Verify everything is working
# ====================================
# Before first use, verify all components:
#
#     python verify_pipeline.py
#
# You should see: "6/6 checks passed ✓"


# STEP 2: Run the evaluation
# ==========================
# Start the main evaluation script:
#
#     python evaluate_pipeline.py
#
# It will:
#   1. Say "🎤 Recording..." - Start speaking
#   2. Say "📝 Transcribing..." - Wait for Whisper
#   3. Ask "Enter expected sentence:" - Type what you said
#   4. Display metrics for that test
#   5. Ask "Save to CSV?" and "Run another test?"


# STEP 3: Review your results
# ============================
# After multiple tests, open the CSV file:
#
#     evaluation_results.csv
#
# In Excel or Python for analysis


# ============================================================================
# COMPLETE EXAMPLE SESSION
# ============================================================================

"""

$ python evaluate_pipeline.py

================================================================================
SPEECH-TO-SIGN TRANSLATOR - EVALUATION SCRIPT
================================================================================

This script evaluates the complete pipeline from microphone input
to sign output, measuring WER, latency, mapping accuracy, and more.

Type 'exit', 'quit', or 'q' to end the evaluation session.
================================================================================


*****  TEST #1  *****

🎤 Recording... (speak now, silence will stop recording)
✓ Recording complete: 4.23s

📝 Transcribing with Whisper (base model)...

📋 Enter the expected sentence (ground truth):
>>> I am going to school tomorrow

================================================================================
TEST RUN #1
================================================================================

📊 INPUTS & OUTPUTS:
  Speech Duration:        4.23 seconds
  Whisper Output:         'I am going to school tomorrow'
  Ground Truth:           'I am going to school tomorrow'
  ASL Converted:          'TOMORROW SCHOOL I GO'
  Mapped Signs:           TOMORROW SCHOOL I GO

📈 METRICS:
  Word Error Rate (WER):  0.0000 (0.00%)
  Real-Time Factor (RTF): 0.3456
  End-to-End Latency:     1.234 seconds
  Mapping Accuracy:       1.0000 (100.00%)
  Vocabulary Coverage:    1.0000 (100.00%)
  Unknown Word Ratio:     0.0000 (0.00%)

-


💾 Save results to CSV? (y/n, default=y): 
>>> y
✅ Results saved to evaluation_results.csv

🔄 Run another test? (y/n, default=y): 
>>> n

================================================================================
SESSION SUMMARY
================================================================================

Tests Completed: 1

  Average WER:            0.0000 (0.00%)
  Average RTF:            0.3456
  Average Latency:        1.234 seconds
  Average Mapping Acc:    1.0000 (100.00%)
  Average Coverage:       1.0000 (100.00%)
  Average Unknown Ratio:  0.0000 (0.00%)

================================================================================

✅ Evaluation session completed!
📊 Results saved to: evaluation_results.csv

"""

# ============================================================================
# WHAT THE METRICS MEAN
# ============================================================================

metrics_guide = {
    "WER (Word Error Rate)": {
        "Range": "0.0 - ∞",
        "Goal": "< 0.15 (85% accurate)",
        "What it is": "How many words Whisper got wrong",
        "Formula": "(Substitutions + Insertions + Deletions) / Total Words",
        "Example": {
            "You said": "I like pizza",
            "Whisper got": "I like pita",
            "WER": 0.33,  # 1 error out of 3 words
        }
    },
    
    "RTF (Real-Time Factor)": {
        "Range": "0.0 - ∞",
        "Goal": "< 1.0 (faster than real-time)",
        "What it is": "How efficiently the system processes audio",
        "Formula": "Processing Time / Audio Duration",
        "Example": {
            "Audio": "5 seconds",
            "Processing": "2 seconds",
            "RTF": 0.4,  # 4x faster than real-time
        }
    },
    
    "End-to-End Latency": {
        "Range": "0 - ∞ seconds",
        "Goal": "< 2 seconds",
        "What it is": "Total time from speaking to sign output",
        "Includes": "Recording, Whisper, ASL grammar, lemmatization, mapping",
        "Example": "1.234 seconds - feels natural"
    },
    
    "Mapping Accuracy": {
        "Range": "0.0 - 1.0",
        "Goal": "> 0.85 (85%)",
        "What it is": "% of words found in sign dictionary",
        "Formula": "Matched Words / Total Words",
        "Example": {
            "Sentence": "I like pizza",
            "All in dict": True,
            "Accuracy": 1.0
        }
    },
    
    "Vocabulary Coverage": {
        "Range": "0.0 - 1.0",
        "Goal": "> 0.90 (90%)",
        "What it is": "% of your unique vocabulary that we know",
        "Formula": "Known Words / Unique Words",
        "Example": {
            "Unique words": ["I", "like", "pizza"],
            "All known": True,
            "Coverage": 1.0
        }
    },
    
    "Unknown Word Ratio": {
        "Range": "0.0 - 1.0",
        "Goal": "< 0.10 (10%)",
        "What it is": "% of words that will be fingerspelled",
        "Formula": "Unknown Words / Total Words",
        "Example": {
            "Unknown": ["supercalifragilisticexpialidocious"],
            "Total": 4,
            "Ratio": 0.25  # 25% fingerspelled
        }
    }
}

# ============================================================================
# TYPICAL RESULTS FOR DIFFERENT VOCABULARY
# ============================================================================

"""

SCENARIO 1: Basic Vocabulary (I, you, like, pizza, etc.)
────────────────────────────────────────────────────────
WER:                0.05 (95% Whisper accurate) ✓✓✓
RTF:                0.35 (3x faster than real-time) ✓✓✓
Latency:            1.2 seconds ✓✓
Mapping Accuracy:   0.95 (95% of words signed) ✓✓✓
Coverage:           0.98 (98% known vocabulary) ✓✓✓
Unknown:            0.02 (2% fingerspelled) ✓✓✓

Result: EXCELLENT - Natural signing with minimal fingerspelling


SCENARIO 2: Mixed Vocabulary (common + some uncommon words)
──────────────────────────────────────────────────────────
WER:                0.12 (88% Whisper accurate) ✓✓
RTF:                0.42 (2.4x faster than real-time) ✓✓
Latency:            1.5 seconds ✓
Mapping Accuracy:   0.78 (78% of words signed) ✓
Coverage:           0.85 (85% known vocabulary) ✓
Unknown:            0.15 (15% fingerspelled) ✓

Result: GOOD - Mostly signing with some fingerspelling


SCENARIO 3: Complex Vocabulary (technical terms, rare words)
───────────────────────────────────────────────────────────
WER:                0.25 (75% Whisper accurate)
RTF:                0.48 (2x faster than real-time) ✓
Latency:            2.1 seconds
Mapping Accuracy:   0.62 (62% of words signed)
Coverage:           0.68 (68% known vocabulary)
Unknown:            0.32 (32% fingerspelled)

Result: FAIR - Mix of signing and fingerspelling


SCENARIO 4: Unsupported Language/Slang
───────────────────────────────────────
WER:                0.45 (55% Whisper accurate)
RTF:                0.45 (2.2x faster) ✓
Latency:            2.8 seconds
Mapping Accuracy:   0.35 (35% of words signed)
Coverage:           0.40 (40% known vocabulary)
Unknown:            0.60 (60% fingerspelled)

Result: POOR - Heavy reliance on fingerspelling

"""

# ============================================================================
# HOW TO IMPROVE RESULTS
# ============================================================================

improvements = {
    "High WER": [
        "1. Speak clearly and at natural pace",
        "2. Use better Whisper model: change WHISPER_MODEL to 'small' or 'medium'",
        "3. First run downloads model (~2GB for small, ~3GB for medium)",
        "4. Subsequent runs use cached model",
    ],
    
    "Low Mapping Accuracy": [
        "1. Expand backend/gloss/daily_pairs.txt with more words",
        "2. Check word_resolver.py lemmatization rules",
        "3. Use only common vocabulary in your sentences",
        "4. Focus on ASL-relevant vocabulary",
    ],
    
    "High Latency": [
        "1. System is still fast (need >2.5s to be problematic)",
        "2. If >2.5s: check CPU usage, close other apps",
        "3. Consider using 'tiny' model (faster, less accurate)",
        "4. Pre-load Whisper model before running evaluation",
    ],
    
    "High Unknown Ratio": [
        "1. Add missing words to daily_pairs.txt",
        "2. Improve lemmatization for base forms",
        "3. Extend word_resolver.py with custom rules",
        "4. Focus conversations on known vocabulary",
    ],
}

# ============================================================================
# FILES IN THIS EVALUATION SYSTEM
# ============================================================================

"""
evaluate_pipeline.py
    Main evaluation script
    - Live microphone recording
    - Whisper transcription
    - Pipeline execution
    - Metrics calculation
    - CSV export
    
verify_pipeline.py
    Pre-evaluation check
    - Verify all modules installed
    - Check dictionary loaded
    - Test pipeline functions
    - Detect audio devices
    Run this first!
    
EVALUATION_GUIDE.md
    Detailed user guide
    - How to run
    - Metric explanations
    - Example outputs
    - Troubleshooting
    
evaluation_results.csv
    Results output file
    - Created after each test
    - Contains all metrics
    - Can be analyzed in Excel
    
Backend modules being tested:
    backend/gloss/infer.py
        Main pipeline orchestration
    backend/gloss/asl_grammar.py
        English → ASL word order conversion
    backend/gloss/word_resolver.py
        Lemmatization & stemming fallback
    backend/gloss/normalize.py
        Text normalization
"""

# ============================================================================
# COMMANDS TO REM REMEMBER
# ============================================================================

commands = """
# First time setup
pip install nltk sounddevice openai-whisper

# Verify everything works
python verify_pipeline.py

# Run evaluation
python evaluate_pipeline.py

# Read guide
python EVALUATION_GUIDE.md

# Analyze results in Python
import pandas as pd
df = pd.read_csv('evaluation_results.csv')
print(df[['whisper_output', 'wer', 'latency']])
print("Average WER:", df['wer'].mean())

# View in Excel (Windows)
start evaluation_results.csv
"""

print(commands)
