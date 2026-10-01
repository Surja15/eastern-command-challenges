import sys
import soundfile as sf
from faster_whisper import WhisperModel

audio, sr = sf.read("quick_clean.wav", dtype="float32")
model = WhisperModel("small", compute_type="int8")

# vad_filter=True stops the "No way / a queen" loop hallucination
# initial_prompt provides context so Whisper doesn't mishear WWII terms
segments, info = model.transcribe(
    audio, 
    vad_filter=True, 
    initial_prompt="A WW2 audio recording of American soldiers talking about Christmas, Queen Mary, and a whetstone gift."
)

for s in segments:
    print(f"[{s.start:.1f}-{s.end:.1f}] {s.text}")
