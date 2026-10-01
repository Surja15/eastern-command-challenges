import sys
import soundfile as sf
from faster_whisper import WhisperModel

audio, sr = sf.read(sys.argv[1], dtype="float32")
model = WhisperModel("small", compute_type="int8")
segments, info = model.transcribe(audio, vad_filter=False, beam_size=5)
print("Language:", info.language)
for s in segments:
    print(f"[{s.start:.1f}-{s.end:.1f}] {s.text}")
