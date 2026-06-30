import whisper
import sounddevice as sd
import numpy as np
import queue
import time

from backend.gloss.infer import infer

SAMPLE_RATE = 16000
BLOCKSIZE = 1024

SILENCE_THRESHOLD = 0.003
MAX_SILENCE_BLOCKS = 6   # ~0.4 sec
MIN_AUDIO_SECONDS = 1.5

audio_queue = queue.Queue()


def audio_callback(indata, frames, time_info, status):
    audio_queue.put(indata.copy())


def is_silent(audio):
    rms = np.sqrt(np.mean(audio ** 2))
    return rms < SILENCE_THRESHOLD


def main():
    print(" Starting microphone → gloss (Ctrl+C to stop)")
    model = whisper.load_model("base")

    audio_buffer = []
    silence_blocks = 0

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        blocksize=BLOCKSIZE,
        callback=audio_callback,
    ):
        try:
            print(" Listening...")

            while True:
                chunk = audio_queue.get()
                audio_buffer.append(chunk)

                if is_silent(chunk):
                    silence_blocks += 1
                else:
                    silence_blocks = 0

                audio_len_sec = len(audio_buffer) * BLOCKSIZE / SAMPLE_RATE

                # Sentence boundary detected
                if (
                    silence_blocks >= MAX_SILENCE_BLOCKS
                    and audio_len_sec >= MIN_AUDIO_SECONDS
                ):
                    audio = np.concatenate(audio_buffer).flatten()
                    audio_buffer.clear()
                    silence_blocks = 0

                    result = model.transcribe(
                        audio,
                        language="en",
                        fp16=False,
                        condition_on_previous_text=False
                    )

                    text = result["text"].strip()
                    if text:
                        print(" Speech:", text)
                        print(" Gloss :", infer(text))

        except KeyboardInterrupt:
            print("\n Stopped listening. Exiting cleanly.")


if __name__ == "__main__":
    main()
