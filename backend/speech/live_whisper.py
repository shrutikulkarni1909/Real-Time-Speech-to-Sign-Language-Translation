import whisper
import sounddevice as sd
import numpy as np
import queue
import time
import argparse
import sys

from backend.avatar.push_to_avatar import push_gloss
from backend.gloss.infer import infer

SAMPLE_RATE = 16000
BLOCKSIZE = 1024

# ✅ IMPROVED: More robust silence detection with STRICTER thresholds
SILENCE_THRESHOLD = 0.04      # Higher threshold - only silence below this
MAX_SILENCE_BLOCKS = 18       # 1+ second of silence required to end sentence
MIN_AUDIO_SECONDS = 1.0       # Minimum audio length before processing
MIN_SPEECH_RMS = 0.06         # Very strict: require at least this RMS for real speech
MIN_AUDIO_ENERGY = 0.10       # Minimum energy level to process

# Debug mode - set to True to see RMS values
DEBUG = False  # Set to True to diagnose audio issues

# ✅ IMPROVED: Add debounce to prevent rapid false detections
last_detection_time = 0
DEBOUNCE_SECONDS = 1.5  # Minimum time between detections (increased from 1.0)

audio_queue = queue.Queue()


def audio_callback(indata, frames, time_info, status):
    audio_queue.put(indata.copy())


def is_silent(chunk):
    """Check if chunk is silent (below threshold)"""
    rms = np.sqrt(np.mean(chunk ** 2))
    return rms < SILENCE_THRESHOLD


def has_speech_energy(audio):
    """Validate that audio has sufficient energy for speech"""
    if len(audio) == 0:
        return False
    rms = np.sqrt(np.mean(audio ** 2))
    # Check both RMS and peak amplitude
    peak = np.max(np.abs(audio))
    has_rms = rms > MIN_SPEECH_RMS
    has_peak = peak > 0.1
    has_energy = rms > MIN_AUDIO_ENERGY
    result = has_rms and has_peak and has_energy
    if DEBUG:
        print(f"  [Energy check] RMS={rms:.4f} (need {MIN_SPEECH_RMS}), Peak={peak:.4f}, Energy={rms:.4f} (need {MIN_AUDIO_ENERGY}) -> {result}")
    return result


# ✅ IMPROVED: Audio preprocessing for better recognition
def preprocess_audio(audio):
    """
    Enhance audio quality before Whisper processing:
    1. Amplitude normalization
    2. Gentle silence trimming
    3. Amplification if too quiet
    """
    # Ensure input is float32
    audio = np.asarray(audio, dtype=np.float32)
    
    # Simple normalization: scale to [-1, 1] range
    max_val = np.max(np.abs(audio))
    if max_val > 0.001:  # Only normalize if signal exists
        audio = (audio / max_val).astype(np.float32)
    
    # Gentle silence trimming (remove leading/trailing near-silence)
    threshold = 0.005
    non_silent = np.where(np.abs(audio) > threshold)[0]
    if len(non_silent) > 0:
        audio = audio[non_silent[0]:non_silent[-1] + 1]
    else:
        # If no non-silent parts found, return empty to be caught as silent
        return np.array([], dtype=np.float32)
    
    # Amplify if signal is too quiet (for quiet speakers)
    rms = np.sqrt(np.mean(audio ** 2))
    if 0.001 < rms < 0.03:  # Quiet but not silent
        audio = (audio * (0.03 / rms)).astype(np.float32)
    
    # Ensure we don't clip and maintain float32
    audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
    
    if DEBUG:
        print(f"  [Audio] RMS: {rms:.4f}, max: {np.max(np.abs(audio)):.4f}")
    
    return audio


def main():
    global last_detection_time  # ✅ FIX: Declare global to access/modify the global variable
    
    parser = argparse.ArgumentParser(description="Live microphone → Whisper → gloss → avatar")
    parser.add_argument("--list-devices", action="store_true", help="Print audio devices and exit")
    parser.add_argument("--device", type=int, default=None, help="Input device index (see --list-devices)")
    parser.add_argument("--model", type=str, default="base", help="Whisper model: tiny, base, small, medium (larger = more accurate)")
    parser.add_argument("--debug", action="store_true", help="Show debug info (RMS levels, etc.)")
    parser.add_argument("--no-filter", action="store_true", help="Disable hallucination filtering (accept all transcriptions)")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    # Enable debug mode if requested
    global DEBUG
    DEBUG = args.debug

    print(f" Starting microphone → gloss (model: {args.model}, Ctrl+C to stop)")
    if DEBUG:
        print(" [DEBUG MODE ON - showing RMS levels]")
    if args.no_filter:
        print(" [FILTERING DISABLED - accepting all transcriptions]")
    # ✅ IMPROVED: Allow model selection, default to "base" for speed
    model = whisper.load_model(args.model)

    audio_buffer = []
    silence_blocks = 0

    try:
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            blocksize=BLOCKSIZE,
            callback=audio_callback,
            device=args.device,
        )
    except Exception as e:
        print("\nFailed to open microphone input stream.")
        print("Tip: run `python -m backend.speech.live_whisper --list-devices` then re-run with --device N")
        print("Error:", e)
        sys.exit(1)

    with stream:
        try:
            print(" Listening...")

            while True:
                chunk = audio_queue.get()
                audio_buffer.append(chunk)

                if is_silent(chunk):
                    silence_blocks += 1
                else:
                    silence_blocks = 0
                
                # Debug output
                if DEBUG:
                    rms = np.sqrt(np.mean(chunk ** 2))
                    print(f"  RMS: {rms:.5f} | Silence blocks: {silence_blocks}", end='\r')

                audio_len_sec = len(audio_buffer) * BLOCKSIZE / SAMPLE_RATE

                # Sentence boundary detected
                if (
                    silence_blocks >= MAX_SILENCE_BLOCKS
                    and audio_len_sec >= MIN_AUDIO_SECONDS
                ):
                    audio = np.concatenate(audio_buffer).flatten()
                    audio_buffer.clear()
                    silence_blocks = 0
                    
                    if DEBUG:
                        print(f"\n✓ Audio chunk captured: {len(audio)/SAMPLE_RATE:.2f}s")

                    # ✅ IMPROVED: Preprocess audio before Whisper
                    audio = preprocess_audio(audio)
                    
                    # Skip if preprocessing removed everything
                    if len(audio) < SAMPLE_RATE * 0.3:  # Less than 0.3 seconds
                        if DEBUG:
                            print(" ✗ Audio too short after preprocessing, skipping")
                        continue
                    
                    # Ensure float32 for Whisper compatibility
                    audio = audio.astype(np.float32)

                    # ✅ IMPROVED: Add debounce to prevent rapid false detections
                    current_time = time.time()
                    if current_time - last_detection_time < DEBOUNCE_SECONDS:
                        if DEBUG:
                            print(f" [debounced: too soon after last detection ({current_time - last_detection_time:.1f}s)]")
                        continue
                    last_detection_time = current_time

                    # ✅ IMPROVED: Validate audio has sufficient energy/speech
                    if not has_speech_energy(audio):
                        if DEBUG:
                            print(f" [insufficient speech energy, skipping]")
                        continue

                    if DEBUG:
                        print(f"✓ Processing with Whisper...")
                    
                    # ✅ IMPROVED: Better Whisper parameters with confidence filtering
                    result = model.transcribe(
                        audio,
                        language="en",
                        fp16=False,
                        condition_on_previous_text=False,  # Avoid context bias
                        task="transcribe",
                        temperature=0.0,  # Deterministic output
                        no_speech_threshold=0.6,  # Higher threshold to reduce false positives
                        logprob_threshold=-1.0,   # Filter low-confidence results
                        compression_ratio_threshold=2.4  # Filter compressed/repeated audio
                    )

                    text = result["text"].strip()
                    
                    if not text:
                        if DEBUG:
                            print(f" [empty transcription, skipping]")
                        continue
                    
                    # ✅ IMPROVED: AGGRESSIVE hallucination filtering (unless disabled)
                    if not args.no_filter:
                        # Common single-word hallucinations
                        common_hallucinations = {
                            "thank", "thanks", "thankyou", "thank you", "bye", "goodbye",
                            "hello", "hi", "hey", "yes", "no", "yeah", "nope",
                            "please", "sorry", "excuse", "welcome", "congratulations",
                            "happy", "birthday", "christmas", "new year", "okay", "ok"
                        }
                        
                        words_lower = text.lower().split()
                        
                        # REJECT: Single isolated common hallucination word
                        if len(words_lower) == 1 and words_lower[0] in common_hallucinations:
                            print(f" ✗ [REJECTED: Common hallucination '{text}']")
                            continue
                        
                        # REJECT: Text is JUST common words (no real content)
                        if all(w in common_hallucinations for w in words_lower):
                            print(f" ✗ [REJECTED: Only hallucination words '{text}']")
                            continue
                        
                        # REJECT: Empty or only punctuation
                        if not any(c.isalnum() for c in text):
                            print(f" ✗ [REJECTED: No alphanumeric content '{text}']")
                            continue
                        
                        # REJECT: Too long (likely gibberish or context from previous)
                        if len(words_lower) > 25:
                            print(f" ✗ [REJECTED: Too long ({len(words_lower)} words) '{text}']")
                            continue
                        
                        # REJECT: Contains excessive repeated words
                        if len(set(words_lower)) < len(words_lower) * 0.4:  # Less than 40% unique
                            print(f" ✗ [REJECTED: Too repetitive '{text}']")
                            continue

                    gloss = infer(text)

                    print("Speech : ", text)
                    print("Gloss : ", gloss)

                    tokens = push_gloss(gloss)
                    print("Avatar tokens : ", tokens)
                        

        except KeyboardInterrupt:
            print("\n Stopped listening. Exiting cleanly.")


if __name__ == "__main__":
    main()
