import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel
import sounddevice as sd
import wavio
import whisper
import time

FS = 16000       # sample rate
CHUNK = 2        # seconds per audio chunk
MODEL = "base"   # tiny/base for fast CPU demo

model = whisper.load_model(MODEL)

def record_chunk(duration=CHUNK):
    audio = sd.rec(int(duration * FS), samplerate=FS, channels=1, dtype='float32')
    sd.wait()
    return audio

def save_chunk(audio, filename="temp.wav"):
    wavio.write(filename, audio, FS, sampwidth=2)
    return filename

def transcribe_chunk(audio_file):
    result = model.transcribe(audio_file)
    return result["text"]

class WhisperApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Speech to Text")
        self.setGeometry(100,100,500,200)

        self.label = QLabel("Press 'Start' and speak", self)
        self.label.setGeometry(20,20,460,50)

        self.button = QPushButton("Start", self)
        self.button.setGeometry(200,100,100,50)
        self.button.clicked.connect(self.start_listening)

    def start_listening(self):
        self.label.setText("Listening...")
        try:
            while True:
                start_time = time.time()
                audio_chunk = record_chunk()
                file = save_chunk(audio_chunk)
                text = transcribe_chunk(file)
                if text.strip():
                    self.label.setText(f"Recognized: {text}")
                    print("Transcribed Text:", text)
                elapsed = time.time() - start_time
                if elapsed < CHUNK:
                    time.sleep(CHUNK - elapsed)
        except KeyboardInterrupt:
            self.label.setText("Stopped.")
            print("Stopped listening.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WhisperApp()
    window.show()
    sys.exit(app.exec_())
