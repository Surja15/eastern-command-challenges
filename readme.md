cat << 'EOF' > fix.py
import sys
from faster_whisper import WhisperModel

model = WhisperModel("small", compute_type="int8")
prompt = "WW2 recording of American soldiers talking about Christmas, Queen Mary, Lieutenant Sam Brush, Captain Badley, and a whetstone gift."

segments, info = model.transcribe(sys.argv[1], vad_filter=True, initial_prompt=prompt)

for s in segments:
    print(f"[{s.start:.1f}-{s.end:.1f}] {s.text}")
EOF




python3 fix.py clean_audio.wav | tee clean_transcript.txt
