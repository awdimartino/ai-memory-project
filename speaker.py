# Imports
import re
import sounddevice as sd
from kokoro import KPipeline

# Files
from config import *

# Speech Buffer class to manage partial speech recognition results
class SpeechBuffer:
    def __init__(self):
        self.buffer = ""

    def add(self, text):
        self.buffer += text

    def pop_ready(self):
        parts = re.split(r'([.!?])', self.buffer)
        ready = []

        for i in range(0, len(parts) - 1, 2):
            ready.append(parts[i] + parts[i + 1])

        self.buffer = parts[-1]
        return ready

    def flush(self):
        leftover = self.buffer.strip()
        self.buffer = ""
        return [leftover] if leftover else []
    
# Speaker class to handle text-to-speech synthesis
class Speaker:
    def __init__(self):
        self.pipeline = KPipeline(
            lang_code="a",
            repo_id="hexgrad/Kokoro-82M"
        )

    audio_stream = sd.OutputStream(
        samplerate=24000,
        channels=1,
        dtype="float32"
    )
    audio_stream.start()

    def speak(self, text):
        for _, _, audio in self.pipeline(text, voice="af_heart"):
            self.audio_stream.write(audio.reshape(-1, 1))