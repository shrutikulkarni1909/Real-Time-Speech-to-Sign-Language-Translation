# backend/speech/mic_to_text.py

import whisper
import sounddevice as sd
import numpy as np
import queue

SAMPLE_RATE = 16000
DURATION = 3
DEVICE_INDEX = None  # ← set mic index here if needed (e.g. 3)

audio_queue = queue.Queue()

print("Loading Whisper model...")
model = whisper.load_model("base")
print("Model loaded")

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())

def record_audio():
    audio_queue.queue.clear()

    print("Speak normally...")
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        device=DEVICE_INDEX,
        callback=audio_callback
    ):
        sd.sleep(DURATION * 1000)

    audio = []
    while not audio_queue.empty():
        audio.append(audio_queue.get())

    audio = np.concatenate(audio, axis=0).flatten()

    # 🔊 Amplify quietly spoken audio
    audio = audio * 3.0

    # Normalize
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val

    # Gentle silence trim
    threshold = 0.01
    non_silent = np.where(np.abs(audio) > threshold)[0]
    if len(non_silent) > 0:
        audio = audio[non_silent[0]:non_silent[-1]]

    return audio

def speech_to_text():
    audio = record_audio()
    result = model.transcribe(audio, fp16=False, language="en")
    return result["text"].strip()

if __name__ == "__main__":
    print("Recognized:", speech_to_text())
