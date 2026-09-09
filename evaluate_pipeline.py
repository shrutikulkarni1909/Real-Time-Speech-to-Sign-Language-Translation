#!/usr/bin/env python3
"""
REAL-TIME SPEECH-TO-SIGN LANGUAGE TRANSLATOR - EVALUATION SCRIPT
=================================================================

This script evaluates the accuracy and performance of the complete
speech-to-sign translation pipeline using live microphone input.

Pipeline Flow:
    Microphone → Whisper Transcription → ASL Grammar Preprocessing
    → Lemmatization/Stemming → Sign Dictionary Lookup → Output Metrics

Metrics Evaluated:
    - WER (Word Error Rate): Speech recognition accuracy
    - RTF (Real-Time Factor): Processing efficiency
    - End-to-End Latency: Total system response time
    - Mapping Accuracy: Sign dictionary lookup success
    - Vocabulary Coverage: Known words vs. total words
    - Unknown Word Ratio: Unfamiliar vocab percentage

Usage:
    python evaluate_pipeline.py

    The script will:
    1. Record audio from your microphone
    2. Transcribe using Whisper
    3. Ask for expected text (ground truth)
    4. Run through full pipeline
    5. Calculate metrics
    6. Optionally save results to CSV
"""

import time
import numpy as np
import sounddevice as sd
import whisper
import csv
from datetime import datetime
import os
import re

# Import pipeline modules
from backend.gloss.infer import infer, DAILY_GLOSS_MAP
from backend.gloss.asl_grammar import convert_to_asl_order
from backend.gloss.normalize import normalize_english
from backend.gloss.word_resolver import resolve_word


# ============================================================================
# CONFIGURATION
# ============================================================================

SAMPLE_RATE = 16000           # Whisper requires 16kHz
BLOCKSIZE = 1024              # Audio chunk size
SILENCE_THRESHOLD = 0.025     # RMS threshold for silence detection (LOWERED for less sensitivity)
GRACE_PERIOD_SECONDS = 1.5    # Don't detect silence for first 1.5 seconds (NEW - give user time to speak)
MAX_SILENCE_BLOCKS = 25       # ~2 seconds of silence to stop recording (INCREASED from 18)
MIN_AUDIO_SECONDS = 2.5       # Minimum audio to process (INCREASED from 0.8 to give more time)
MIN_SPEECH_RMS = 0.03         # Minimum RMS for valid speech (LOWERED from 0.06 to accept quieter audio)
MAX_AUDIO_SECONDS = 30        # Maximum recording duration
WHISPER_MODEL = "base"        # Use base model for faster inference
EVALUATION_FILE = "evaluation_results.csv"
DEBUG_AUDIO = False           # Set to True to see RMS values during recording

# Statistics tracking
session_stats = {
    "wer_scores": [],
    "rtf_scores": [],
    "latencies": [],
    "mapping_accuracies": [],
    "coverage_scores": [],
    "unknown_ratios": [],
    "gloss_accuracies": [],  # Track gloss prediction accuracy
    "test_count": 0
}


# ============================================================================
# AUDIO RECORDING FUNCTIONS
# ============================================================================

def has_speech_energy(audio):
    """
    Validate that audio has sufficient energy for speech.
    Prevents processing of pure noise or silence.
    """
    if len(audio) == 0:
        return False
    rms = np.sqrt(np.mean(audio ** 2))
    peak = np.max(np.abs(audio))
    has_rms = rms > MIN_SPEECH_RMS
    has_peak = peak > 0.1
    return has_rms and has_peak


def record_audio(timeout=MAX_AUDIO_SECONDS, debug=False):
    """
    Record audio from microphone until silence or max duration.
    
    Args:
        timeout (int): Maximum recording duration in seconds
        debug (bool): Print debug info
    
    Returns:
        tuple: (audio_data, duration_seconds, sample_rate)
    """
    print("\n🎤 Recording... (speak now, silence will stop recording)")
    print("📢 Go ahead, start speaking...")
    
    audio_buffer = []
    silence_blocks = 0
    start_time = time.time()
    grace_period_end = start_time + GRACE_PERIOD_SECONDS  # Don't detect silence for first N seconds
    
    def audio_callback(indata, frames, time_info, status):
        """Callback for audio stream"""
        if status:
            print(f"  ⚠️ Audio status: {status}")
        audio_buffer.append(indata.copy())
    
    def is_silent(chunk):
        """Check if audio chunk is below silence threshold"""
        rms = np.sqrt(np.mean(chunk ** 2))
        if DEBUG_AUDIO:
            print(f"    RMS: {rms:.4f} (threshold: {SILENCE_THRESHOLD})", end='\r')
        return rms < SILENCE_THRESHOLD
    
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            blocksize=BLOCKSIZE,
            callback=audio_callback,
            dtype=np.float32,  # Ensure float32 from the start
        ):
            while True:
                # Check if we've reached max duration
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    if debug:
                        print(f"  ⏱️ Max duration ({timeout}s) reached")
                    break
                
                # Check for silence (but not during grace period)
                current_time = time.time()
                if len(audio_buffer) > 0:
                    latest_chunk = audio_buffer[-1]
                    audio_duration = len(audio_buffer) * BLOCKSIZE / SAMPLE_RATE
                    
                    # Only detect silence after grace period
                    if current_time >= grace_period_end:
                        if is_silent(latest_chunk):
                            silence_blocks += 1
                            if debug or DEBUG_AUDIO:
                                print(f"  🔇 Silence: {silence_blocks}/{MAX_SILENCE_BLOCKS}", end='\r')
                        else:
                            silence_blocks = 0  # Reset silence counter when sound detected
                        
                        # Stop if enough silence detected AND minimum audio collected
                        if silence_blocks >= MAX_SILENCE_BLOCKS and audio_duration >= MIN_AUDIO_SECONDS:
                            if debug or DEBUG_AUDIO:
                                print(f"\n  ✓ Silence detected, stopping recording")
                            break
                    else:
                        # During grace period, just show elapsed time
                        remaining = grace_period_end - current_time
                        if debug or DEBUG_AUDIO:
                            print(f"  ⏱️ Speaking... ({remaining:.1f}s grace period remaining)", end='\r')
                
                time.sleep(0.01)  # Small delay to prevent busy waiting
    
    except KeyboardInterrupt:
        print("\n⏸️ Recording interrupted by user")
    
    # Combine audio chunks
    if audio_buffer:
        audio = np.concatenate(audio_buffer, axis=0).flatten()
        audio = audio.astype(np.float32)
        duration = len(audio) / SAMPLE_RATE
        
        # Validate audio has actual speech energy
        if not has_speech_energy(audio):
            print(f"  ⚠️ Warning: Audio energy is low")
            print(f"     Increase microphone volume in Windows Sound Settings")
        
        if debug or DEBUG_AUDIO:
            rms = np.sqrt(np.mean(audio ** 2))
            peak = np.max(np.abs(audio))
            print(f"\n  ✓ Recording complete: {duration:.2f}s (RMS: {rms:.4f}, Peak: {peak:.4f})")
        
        return audio, duration, SAMPLE_RATE
    
    return None, 0, SAMPLE_RATE


# ============================================================================
# SPEECH-TO-TEXT FUNCTIONS
# ============================================================================

def transcribe_audio(audio, sample_rate, debug=False):
    """
    Transcribe audio using OpenAI Whisper.
    
    Args:
        audio (np.array): Audio data
        sample_rate (int): Sample rate of audio
        debug (bool): Print debug info
    
    Returns:
        str: Transcribed text
    """
    if audio is None or len(audio) == 0:
        print("❌ No audio data to transcribe")
        return ""
    
    print(f"\n📝 Transcribing with Whisper ({WHISPER_MODEL} model)...")
    
    try:
        # Load model (cached after first run)
        print(f"  Loading {WHISPER_MODEL} model...")
        model = whisper.load_model(WHISPER_MODEL)
        
        # Ensure float32 format
        audio = audio.astype(np.float32)
        
        # CRITICAL FIX: Normalize properly for Whisper
        # Whisper internally normalizes, but we should give it clean audio
        # Remove DC offset (mean)
        audio = audio - np.mean(audio)
        
        # Normalize to approximately [-0.5, 0.5] range (conservative scaling)
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / (max_val * 2.0)  # Scale by 2x max to avoid clipping
        
        # Ensure no NaN or inf values
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        
        if debug or DEBUG_AUDIO:
            print(f"  Audio stats: RMS={np.sqrt(np.mean(audio**2)):.4f}, Peak={np.max(np.abs(audio)):.4f}")
        
        # Transcribe with optimized settings
        result = model.transcribe(
            audio,
            language="en",
            fp16=False,
            task="transcribe",
            temperature=0.0,
            best_of=1,
            beam_size=1,
        )
        
        text = result["text"].strip()
        
        if not text:
            print("  ⚠️ Warning: Whisper returned empty transcription")
        elif debug or DEBUG_AUDIO:
            print(f"  ✓ Transcription: '{text}'")
        
        return text
    
    except Exception as e:
        print(f"❌ Transcription error: {e}")
        return ""


# ============================================================================
# TEXT PROCESSING FUNCTIONS
# ============================================================================

def get_ground_truth():
    """
    Ask user to input the expected/ground truth text.
    
    Returns:
        str: Ground truth text from user
    """
    print("\n📋 What did you say? (Enter the sentence you just spoke)")
    ground_truth = input(">>> ").strip()
    return ground_truth


def get_expected_gloss():
    """
    Ask user to input the expected gloss (ASL signs).
    
    Returns:
        str: Expected gloss from user (space-separated sign names)
    """
    print("\n📋 Enter expected gloss (ASL signs, space-separated):")
    print("   Example: TOMORROW SCHOOL I GO")
    expected_gloss = input(">>> ").strip().upper()
    return expected_gloss


def apply_pipeline(transcribed_text, debug=False):
    """
    Apply the full pipeline: normalization → ASL grammar → lemmatization → mapping.
    
    Args:
        transcribed_text (str): Raw transcribed text from Whisper
        debug (bool): Print debug info
    
    Returns:
        dict: Pipeline results with intermediate outputs
    """
    if not transcribed_text:
        return {
            "normalized": "",
            "asl_converted": "",
            "mapped_signs": "",
            "unknown_words": [],
            "mapped_words": [],
            "all_words": []
        }
    
    # Step 1: Normalize
    normalized = normalize_english(transcribed_text)
    
    # Step 2: Apply ASL grammar preprocessing
    asl_converted = convert_to_asl_order(normalized, debug=False)
    
    if debug:
        print(f"  → Normalized: {normalized}")
        print(f"  → ASL Grammar: {asl_converted}")
    
    # Step 3: Word-by-word resolution and mapping
    words = asl_converted.split()
    mapped_signs = []
    unknown_words = []
    mapped_words = []
    
    for word in words:
        word_lower = word.lower().rstrip('.,!?;:')
        
        # Try to resolve word (exact, lemma, stem)
        sign = resolve_word(word_lower, DAILY_GLOSS_MAP, debug=False)
        
        if sign:
            # Word found in dictionary
            mapped_signs.append(sign)
            mapped_words.append(word_lower)
        else:
            # Word not found - will be fingerspelled
            mapped_signs.append(word.upper())  # Fingerspelling
            unknown_words.append(word_lower)
    
    result = {
        "normalized": normalized,
        "asl_converted": asl_converted,
        "mapped_signs": " ".join(mapped_signs),
        "unknown_words": unknown_words,
        "mapped_words": mapped_words,
        "all_words": words
    }
    
    if debug:
        print(f"  → Mapped signs: {result['mapped_signs']}")
        if unknown_words:
            print(f"  → Unknown words: {', '.join(unknown_words)}")
    
    return result


# ============================================================================
# METRIC CALCULATION FUNCTIONS
# ============================================================================

def calculate_wer(reference, hypothesis):
    """
    Calculate Word Error Rate (WER).
    
    WER = (Substitutions + Insertions + Deletions) / Total Reference Words
    
    Args:
        reference (str): Expected text (ground truth)
        hypothesis (str): Transcribed text
    
    Returns:
        float: WER score (0.0 = perfect, >1.0 = many errors)
    """
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    
    if len(ref_words) == 0:
        return 1.0 if len(hyp_words) > 0 else 0.0
    
    # Simple edit distance calculation (Levenshtein)
    # For production, consider using jiwer library
    
    # Create a matrix for dynamic programming
    d = {}
    
    for i in range(len(ref_words) + 1):
        d[i, 0] = i
    for j in range(len(hyp_words) + 1):
        d[0, j] = j
    
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                d[i, j] = d[i-1, j-1]
            else:
                substitution = d[i-1, j-1] + 1
                insertion = d[i, j-1] + 1
                deletion = d[i-1, j] + 1
                d[i, j] = min(substitution, insertion, deletion)
    
    # WER = errors / total reference words
    wer = d[len(ref_words), len(hyp_words)] / len(ref_words)
    
    return wer


def calculate_rtf(processing_time, audio_duration):
    """
    Calculate Real-Time Factor (RTF).
    
    RTF = processing_time / audio_duration
    RTF < 1.0 means system is faster than real-time
    RTF = 1.0 means real-time
    RTF > 1.0 means slower than real-time
    
    Args:
        processing_time (float): Time to process audio (seconds)
        audio_duration (float): Duration of audio (seconds)
    
    Returns:
        float: RTF score
    """
    if audio_duration == 0:
        return float('inf')
    return processing_time / audio_duration


def calculate_mapping_accuracy(pipeline_result):
    """
    Calculate mapping accuracy (% of words successfully mapped to signs).
    
    Args:
        pipeline_result (dict): Result from apply_pipeline()
    
    Returns:
        float: Accuracy between 0.0 and 1.0
    """
    total_words = len(pipeline_result["all_words"])
    
    if total_words == 0:
        return 1.0
    
    mapped_count = len(pipeline_result["mapped_words"])
    accuracy = mapped_count / total_words
    
    return accuracy


def calculate_vocabulary_coverage(pipeline_result):
    """
    Calculate vocabulary coverage (% of unique words in dictionary).
    
    Args:
        pipeline_result (dict): Result from apply_pipeline()
    
    Returns:
        float: Coverage between 0.0 and 1.0
    """
    unique_words = set(word.lower() for word in pipeline_result["all_words"])
    
    if len(unique_words) == 0:
        return 1.0
    
    # Count words in dictionary
    words_in_dict = 0
    for word in unique_words:
        if resolve_word(word, DAILY_GLOSS_MAP):
            words_in_dict += 1
    
    coverage = words_in_dict / len(unique_words)
    return coverage


def calculate_unknown_ratio(pipeline_result):
    """
    Calculate ratio of unknown words.
    
    Args:
        pipeline_result (dict): Result from apply_pipeline()
    
    Returns:
        float: Ratio between 0.0 and 1.0
    """
    total_words = len(pipeline_result["all_words"])
    
    if total_words == 0:
        return 0.0
    
    unknown_count = len(pipeline_result["unknown_words"])
    ratio = unknown_count / total_words
    
    return ratio


def calculate_gloss_accuracy(expected_gloss, predicted_gloss):
    """
    Calculate gloss accuracy using GER (Gloss Error Rate).
    Compares expected signs with predicted signs.
    
    Args:
        expected_gloss (str): Expected signs (space-separated)
        predicted_gloss (str): Predicted signs (space-separated)
    
    Returns:
        dict: 'ger' (0.0=perfect), 'accuracy' (0.0-1.0), 'matches', 'total'
    """
    expected_signs = expected_gloss.upper().split() if expected_gloss.strip() else []
    predicted_signs = predicted_gloss.upper().split() if predicted_gloss.strip() else []
    
    if len(expected_signs) == 0:
        return {"ger": 0.0 if len(predicted_signs) == 0 else 1.0, "accuracy": 1.0 if len(predicted_signs) == 0 else 0.0, "matches": 0, "total": 0}
    
    # Calculate Levenshtein distance
    d = {}
    for i in range(len(expected_signs) + 1):
        d[i, 0] = i
    for j in range(len(predicted_signs) + 1):
        d[0, j] = j
    
    for i in range(1, len(expected_signs) + 1):
        for j in range(1, len(predicted_signs) + 1):
            if expected_signs[i-1] == predicted_signs[j-1]:
                d[i, j] = d[i-1, j-1]
            else:
                substitution = d[i-1, j-1] + 1
                insertion = d[i, j-1] + 1
                deletion = d[i-1, j] + 1
                d[i, j] = min(substitution, insertion, deletion)
    
    # Calculate GER
    ger = d[len(expected_signs), len(predicted_signs)] / len(expected_signs)
    accuracy = 1.0 - ger
    
    # Count exact matches
    matches = sum(1 for es, ps in zip(expected_signs, predicted_signs) if es == ps)
    
    return {"ger": ger, "accuracy": accuracy, "matches": matches, "total": len(expected_signs)}


# ============================================================================
# RESULT PRINTING FUNCTIONS
# ============================================================================

def print_results(
    audio_duration,
    whisper_output,
    ground_truth,
    pipeline_result,
    wer,
    rtf,
    latency,
    mapping_accuracy,
    vocabulary_coverage,
    unknown_ratio,
    expected_gloss,
    gloss_accuracy_result,
    test_number
):
    """
    Print detailed results for a single test run.
    
    Args:
        audio_duration (float): Recording duration
        whisper_output (str): Transcribed text from Whisper
        ground_truth (str): Expected text
        pipeline_result (dict): Result from apply_pipeline()
        wer (float): Word Error Rate
        rtf (float): Real-Time Factor
        latency (float): Total latency
        mapping_accuracy (float): Mapping accuracy
        vocabulary_coverage (float): Vocabulary coverage
        unknown_ratio (float): Unknown word ratio
        expected_gloss (str): Expected gloss from user
        gloss_accuracy_result (dict): Result from calculate_gloss_accuracy()
        test_number (int): Test run number
    """
    print("\n" + "=" * 80)
    print(f"TEST RUN #{test_number}")
    print("=" * 80)
    
    print(f"\n📊 SPEECH-TO-TEXT:")
    print(f"  Speech Duration:        {audio_duration:.2f} seconds")
    print(f"  Whisper Output:         '{whisper_output}'")
    print(f"  Ground Truth:           '{ground_truth}'")
    
    print(f"\n📊 GLOSS COMPARISON:")
    print(f"  Expected Gloss:         {expected_gloss if expected_gloss else '(not provided)'}")
    print(f"  Predicted Gloss:        {pipeline_result['mapped_signs'][:80]}")
    
    print(f"\n📊 INTERMEDIATE STEPS:")
    print(f"  ASL Converted:          '{pipeline_result['asl_converted']}'")
    
    if pipeline_result["unknown_words"]:
        print(f"  Unknown Words:          {', '.join(pipeline_result['unknown_words'])}")
    
    print(f"\n📈 METRICS:")
    print(f"  Word Error Rate (WER):  {wer:.4f} ({wer*100:.2f}%)")
    print(f"    → Lower is better (0.0 = perfect transcription)")
    print(f"  Real-Time Factor (RTF): {rtf:.4f}")
    print(f"    → <1.0 = faster than real-time, 1.0 = real-time")
    print(f"  End-to-End Latency:     {latency:.3f} seconds")
    print(f"    → Total time from speech start to sign output")
    print(f"  Mapping Accuracy:       {mapping_accuracy:.4f} ({mapping_accuracy*100:.2f}%)")
    print(f"    → % of words successfully mapped to signs")
    print(f"  Vocabulary Coverage:    {vocabulary_coverage:.4f} ({vocabulary_coverage*100:.2f}%)")
    print(f"    → % of unique words found in dictionary")
    print(f"  Unknown Word Ratio:     {unknown_ratio:.4f} ({unknown_ratio*100:.2f}%)")
    print(f"    → % of words not in dictionary (fingerspelled)")
    
    if expected_gloss:
        print(f"\n📊 GLOSS ACCURACY:")
        print(f"  Gloss Error Rate (GER): {gloss_accuracy_result['ger']:.4f} ({gloss_accuracy_result['ger']*100:.2f}%)")
        print(f"    → Lower is better (0.0 = perfect gloss prediction)")
        print(f"  Gloss Accuracy:         {gloss_accuracy_result['accuracy']:.4f} ({gloss_accuracy_result['accuracy']*100:.2f}%)")
        print(f"    → Exact sign matches: {gloss_accuracy_result['matches']}/{gloss_accuracy_result['total']}")
    
    print("\n" + "-" * 80)


def print_session_summary():
    """
    Print summary statistics for entire session.
    """
    if session_stats["test_count"] == 0:
        print("\n❌ No tests completed yet.")
        return
    
    print("\n" + "=" * 80)
    print("SESSION SUMMARY")
    print("=" * 80)
    
    print(f"\n📊 Tests Completed: {session_stats['test_count']}")
    
    if session_stats["wer_scores"]:
        avg_wer = np.mean(session_stats["wer_scores"])
        print(f"  Average WER:            {avg_wer:.4f} ({avg_wer*100:.2f}%)")
    
    if session_stats["rtf_scores"]:
        avg_rtf = np.mean(session_stats["rtf_scores"])
        print(f"  Average RTF:            {avg_rtf:.4f}")
    
    if session_stats["latencies"]:
        avg_latency = np.mean(session_stats["latencies"])
        print(f"  Average Latency:        {avg_latency:.3f} seconds")
    
    if session_stats["mapping_accuracies"]:
        avg_accuracy = np.mean(session_stats["mapping_accuracies"])
        print(f"  Average Mapping Acc:    {avg_accuracy:.4f} ({avg_accuracy*100:.2f}%)")
    
    if session_stats["coverage_scores"]:
        avg_coverage = np.mean(session_stats["coverage_scores"])
        print(f"  Average Coverage:       {avg_coverage:.4f} ({avg_coverage*100:.2f}%)")
    
    if session_stats["unknown_ratios"]:
        avg_unknown = np.mean(session_stats["unknown_ratios"])
        print(f"  Average Unknown Ratio:  {avg_unknown:.4f} ({avg_unknown*100:.2f}%)")
    
    if session_stats["gloss_accuracies"]:
        avg_gloss = np.mean(session_stats["gloss_accuracies"])
        print(f"  Average Gloss Accuracy: {avg_gloss:.4f} ({avg_gloss*100:.2f}%)")
    
    print("\n" + "=" * 80)


# ============================================================================
# CSV EXPORT FUNCTIONS
# ============================================================================

def save_results_to_csv(
    test_number,
    audio_duration,
    whisper_output,
    ground_truth,
    asl_converted,
    mapped_signs,
    unknown_words,
    wer,
    rtf,
    latency,
    mapping_accuracy,
    vocabulary_coverage,
    unknown_ratio,
    expected_gloss,
    gloss_accuracy
):
    """
    Save test results to CSV file.
    
    Args:
        test_number (int): Test run number
        audio_duration (float): Recording duration
        whisper_output (str): Transcribed text
        ground_truth (str): Expected text
        asl_converted (str): ASL-reordered text
        mapped_signs (str): Final sign output
        unknown_words (list): Words not in dictionary
        wer (float): Word Error Rate
        rtf (float): Real-Time Factor
        latency (float): Total latency
        mapping_accuracy (float): Mapping accuracy
        vocabulary_coverage (float): Vocabulary coverage
        unknown_ratio (float): Unknown word ratio
        expected_gloss (str): Expected gloss from user
        gloss_accuracy (dict): Gloss accuracy metrics
    """
    file_exists = os.path.exists(EVALUATION_FILE)
    
    with open(EVALUATION_FILE, 'a', newline='') as csvfile:
        fieldnames = [
            'test_number', 'timestamp', 'audio_duration',
            'whisper_output', 'ground_truth', 'asl_converted',
            'expected_gloss', 'predicted_gloss', 'gloss_error_rate', 'gloss_accuracy',
            'mapped_signs', 'unknown_words',
            'wer', 'rtf', 'latency',
            'mapping_accuracy', 'vocabulary_coverage', 'unknown_ratio'
        ]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header if file doesn't exist
        if not file_exists:
            writer.writeheader()
        
        # Write data row
        writer.writerow({
            'test_number': test_number,
            'timestamp': datetime.now().isoformat(),
            'audio_duration': f"{audio_duration:.2f}",
            'whisper_output': whisper_output,
            'ground_truth': ground_truth,
            'asl_converted': asl_converted,
            'expected_gloss': expected_gloss,
            'predicted_gloss': mapped_signs[:100],
            'gloss_error_rate': f"{gloss_accuracy['ger']:.4f}" if expected_gloss else "N/A",
            'gloss_accuracy': f"{gloss_accuracy['accuracy']:.4f}" if expected_gloss else "N/A",
            'mapped_signs': mapped_signs[:100],
            'unknown_words': ','.join(unknown_words),
            'wer': f"{wer:.4f}",
            'rtf': f"{rtf:.4f}",
            'latency': f"{latency:.3f}",
            'mapping_accuracy': f"{mapping_accuracy:.4f}",
            'vocabulary_coverage': f"{vocabulary_coverage:.4f}",
            'unknown_ratio': f"{unknown_ratio:.4f}"
        })
    
    print(f"✅ Results saved to {EVALUATION_FILE}")


# ============================================================================
# MAIN EVALUATION LOOP
# ============================================================================

def run_evaluation_session():
    """
    Main evaluation loop - runs continuous test cycles.
    """
    print("\n" + "=" * 80)
    print("SPEECH-TO-SIGN TRANSLATOR - EVALUATION SCRIPT")
    print("=" * 80)
    print("\nThis script evaluates the complete pipeline from microphone input")
    print("to sign output, measuring WER, latency, mapping accuracy, and more.")
    print("\nType 'exit', 'quit', or 'q' to end the evaluation session.")
    print("=" * 80)
    
    test_number = 0
    
    while True:
        test_number += 1
        print(f"\n\n{'*' * 80}")
        print(f"TEST #{test_number}")
        print(f"{'*' * 80}")
        
        # ===== STEP 0: Wait for user to be ready =====
        print("\n📋 Press ENTER when you are ready to start recording")
        print("   (Speak after pressing ENTER)")
        input()
        
        # ===== STEP 1: Record audio =====
        total_start_time = time.time()
        audio, audio_duration, sample_rate = record_audio(debug=False)
        
        if audio is None or len(audio) == 0:
            print("⚠️ No audio recorded. Let's try again.")
            time.sleep(1)
            continue
        
        # ===== CONFIRMATION: Recording is complete =====
        print("\n✅ Recording complete!")
        print(f"   Duration: {audio_duration:.2f} seconds")
        time.sleep(1)  # Brief pause so user sees the message
        
        # ===== STEP 2: Transcribe with Whisper =====
        transcription_start = time.time()
        whisper_output = transcribe_audio(audio, sample_rate, debug=False)
        transcription_time = time.time() - transcription_start
        
        if not whisper_output:
            print("❌ Transcription failed. Let's try again.")
            time.sleep(1)
            continue
        
        # ===== STEP 3: Get ground truth from user =====
        print("\n📋 What did you say? (Enter the sentence you just spoke)")
        ground_truth = input(">>> ").strip()
        
        if not ground_truth:
            print("⚠️ No ground truth provided. Let's try again.")
            time.sleep(1)
            continue
        
        # ===== STEP 3.1: Get expected gloss from user =====
        expected_gloss = get_expected_gloss()
        
        # ===== STEP 4: Apply full pipeline =====
        pipeline_start = time.time()
        pipeline_result = apply_pipeline(whisper_output, debug=False)
        pipeline_time = time.time() - pipeline_start
        
        # ===== STEP 5: Calculate metrics =====
        total_end_time = time.time()
        
        wer = calculate_wer(ground_truth, whisper_output)
        rtf = calculate_rtf(transcription_time + pipeline_time, audio_duration)
        latency = total_end_time - total_start_time
        mapping_accuracy = calculate_mapping_accuracy(pipeline_result)
        vocabulary_coverage = calculate_vocabulary_coverage(pipeline_result)
        unknown_ratio = calculate_unknown_ratio(pipeline_result)
        gloss_accuracy_result = calculate_gloss_accuracy(expected_gloss, pipeline_result["mapped_signs"])
        
        # ===== STEP 6: Print results =====
        print_results(
            audio_duration,
            whisper_output,
            ground_truth,
            pipeline_result,
            wer,
            rtf,
            latency,
            mapping_accuracy,
            vocabulary_coverage,
            unknown_ratio,
            expected_gloss,
            gloss_accuracy_result,
            test_number
        )
        
        # ===== STEP 7: Update session statistics =====
        session_stats["wer_scores"].append(wer)
        session_stats["rtf_scores"].append(rtf)
        session_stats["latencies"].append(latency)
        session_stats["mapping_accuracies"].append(mapping_accuracy)
        session_stats["coverage_scores"].append(vocabulary_coverage)
        session_stats["unknown_ratios"].append(unknown_ratio)
        session_stats["gloss_accuracies"].append(gloss_accuracy_result["accuracy"])
        session_stats["test_count"] += 1
        
        # ===== STEP 8: Save to CSV =====
        save_to_csv = input("\n💾 Save results to CSV? (y/n, default=y): ").lower().strip()
        if save_to_csv != 'n':
            save_results_to_csv(
                test_number,
                audio_duration,
                whisper_output,
                ground_truth,
                pipeline_result["asl_converted"],
                pipeline_result["mapped_signs"],
                pipeline_result["unknown_words"],
                wer,
                rtf,
                latency,
                mapping_accuracy,
                vocabulary_coverage,
                unknown_ratio,
                expected_gloss,
                gloss_accuracy_result
            )
        
        # ===== STEP 9: Ask to continue BEFORE next recording =====
        print("\n" + "-" * 80)
        continue_eval = input("\n▶️  Ready for next test? (y/n, default=y): ").lower().strip()
        
        if continue_eval == 'n' or continue_eval in ['exit', 'quit', 'q']:
            break
        
        # Brief pause before next test
        print("\n⏳ Preparing for next test...")
        time.sleep(2)
    
    # Print final session summary
    print_session_summary()
    
    print(f"\n✅ Evaluation session completed!")
    print(f"📊 Results saved to: {EVALUATION_FILE}")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        run_evaluation_session()
    except KeyboardInterrupt:
        print("\n\n⏸️ Evaluation interrupted by user")
        print_session_summary()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
