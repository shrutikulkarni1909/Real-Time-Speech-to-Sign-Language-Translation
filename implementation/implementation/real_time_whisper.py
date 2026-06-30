import pyaudio
import wave
import os
from faster_whisper import WhisperModel
import time
import sys
import msvcrt
import pandas as pd
from fuzzywuzzy import fuzz
import re  # For regex preprocessing

# --- CONFIGURATION ---
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
WAVE_OUTPUT_FILENAME = "recorded_whisper_audio.wav"

WHISPER_MODEL_SIZE = "tiny"
DEVICE = "cuda" if "cuda" in os.environ.get("CONDA_DEFAULT_ENV", "") or os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
COMPUTE_TYPE = "int8" if DEVICE == "cpu" else "float16"

CORPUS_FILE = "asl.csv"  # <-- Updated dataset name
FUZZY_MATCH_THRESHOLD = 80
# ----------------------

GLOBAL_GLOSS_DF = None
GLOBAL_ENGLISH_SENTENCES = []

# --- Dataset Handling ---
def load_asl_corpus(file_path):
    """Loads the ASL dataset. Exits if file not found."""
    global GLOBAL_GLOSS_DF, GLOBAL_ENGLISH_SENTENCES
    
    if not os.path.exists(file_path):
        print(f"FATAL ERROR: Dataset file '{file_path}' not found. Please provide a valid CSV.")
        sys.exit(1)
    
    print(f"Loading ASL Corpus from: {file_path}...")
    try:
        GLOBAL_GLOSS_DF = pd.read_csv(file_path)
        # Proper regex preprocessing
        GLOBAL_GLOSS_DF['clean_english'] = GLOBAL_GLOSS_DF['english'].astype(str).str.lower().str.strip().apply(
            lambda x: re.sub(r'[^a-z0-9\s]', '', x)
        )
        GLOBAL_ENGLISH_SENTENCES = GLOBAL_GLOSS_DF['clean_english'].tolist()
        print(f"Corpus loaded successfully. Total sentences: {len(GLOBAL_GLOSS_DF)}")
    except Exception as e:
        print(f"ERROR loading corpus: {e}")
        sys.exit(1)

# --- Audio Recording ---
def record_audio():
    print("\n--- Starting Audio Recording ---")
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    frames = []
    print("Recording started. Press [ENTER] to stop.")
    
    while True:
        try:
            if msvcrt.kbhit() and msvcrt.getch() == b'\r':
                break
            
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
            sys.stdout.write(f"\rRecording... {len(frames) * CHUNK / RATE:.1f} seconds ")
            sys.stdout.flush()

        except KeyboardInterrupt:
            break
            
    print("\n--- Recording Finished ---")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    return WAVE_OUTPUT_FILENAME

# --- Whisper Transcription ---
def transcribe_with_whisper(audio_file_path, loaded_model):
    try:
        start_time = time.time()
        segments, info = loaded_model.transcribe(
            audio_file_path,
            beam_size=1,
            vad_filter=True,
            language="en"
        )
        full_transcript = " ".join(segment.text for segment in segments).strip()
        end_time = time.time()
        print(f"Transcription took: {end_time - start_time:.2f} seconds (Lang: {info.language})")
        return full_transcript
    except Exception as e:
        print(f"ERROR during Whisper transcription: {e}")
        return None

# --- ASL Gloss Matching ---
def dataset_gloss_generator(transcript):
    if not transcript or GLOBAL_GLOSS_DF is None:
        return None, "Dataset not loaded or transcript is empty."

    # Proper regex preprocessing
    clean_transcript = re.sub(r'[^a-z0-9\s]', '', transcript.lower().strip())

    best_match_score, best_match_index = -1, -1

    for index, corpus_sentence in enumerate(GLOBAL_ENGLISH_SENTENCES):
        score = fuzz.token_sort_ratio(clean_transcript, corpus_sentence)
        if score > best_match_score:
            best_match_score = score
            best_match_index = index
            if best_match_score == 100:
                break
    
    if best_match_score >= FUZZY_MATCH_THRESHOLD:
        matched_row = GLOBAL_GLOSS_DF.iloc[best_match_index]
        L1 = transcript
        L2 = matched_row['asl_gloss'].upper()
        L3 = matched_row['english']
        print(f"\n[INFO] Found Match: Score {best_match_score} | Corpus Sentence: '{L3}'")
        gloss_output = (
            f"L1 (Spoken Text): {L1}\n"
            f"L2 (ASL Gloss): {L2}\n"
            f"L3 (Matched English): {L3}"
        )
        return gloss_output
    else:
        return (
            f"L1 (Spoken Text): {transcript}\n"
            f"L2 (ASL Gloss): NO MATCH FOUND (Score: {best_match_score} - Needs {FUZZY_MATCH_THRESHOLD})\n"
            f"L3 (Matched English): N/A"
        )

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    print("Applied OMP Workaround.")

    load_asl_corpus(CORPUS_FILE)

    try:
        print(f"Pre-loading Whisper model '{WHISPER_MODEL_SIZE}'...")
        GLOBAL_WHISPER_MODEL = WhisperModel(WHISPER_MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
        print("Model loaded successfully.")
    except Exception as e:
        print("FATAL ERROR: Could not load Whisper model.")
        print(e)
        sys.exit(1)
        
    try:
        audio_path = record_audio()
        transcript = transcribe_with_whisper(audio_path, GLOBAL_WHISPER_MODEL)
        gloss_output = dataset_gloss_generator(transcript)

        print("\n" + "=" * 50)
        print("FINAL RESULT (DATASET LOOKUP)")
        print("=" * 50)
        print(gloss_output)
        print("=" * 50)
    finally:
        if os.path.exists(WAVE_OUTPUT_FILENAME):
            pass
