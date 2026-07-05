from faster_whisper import WhisperModel
import os

model = WhisperModel("small", device="cpu", compute_type="int8")

folder = "audios"

for file in os.listdir(folder):
    if not file.endswith(".mp3"):
        continue

    print(f"Transcribing: {file}")
    segments, info = model.transcribe(f"{folder}/{file}", language="hi")
    text = " ".join([seg.text for seg in segments])

    txt_filename = file.replace(".mp3", ".txt")
    with open(f"{folder}/{txt_filename}", "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Done: {file}")