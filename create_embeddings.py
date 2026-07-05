import requests
import os
import json
import pandas as pd

def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })
    return r.json()["embeddings"]

def merge_chunks(chunks, max_duration=30):
    """Group consecutive Whisper segments into bigger ~30-second blocks."""
    merged = []
    current = None

    for c in chunks:
        if current is None:
            current = {
                "number": c["number"],
                "title": c["title"],
                "start": c["start"],
                "end": c["end"],
                "text": c["text"]
            }
        else:
            current["end"] = c["end"]
            current["text"] += " " + c["text"]

        if current["end"] - current["start"] >= max_duration:
            merged.append(current)
            current = None

    if current:
        merged.append(current)

    return merged

jsons = os.listdir("jsons")
my_dicts = []
chunk_id = 0

for json_file in jsons:
    with open(f"jsons/{json_file}") as f:
        content = json.load(f)
    print(f"Creating Embeddings for {json_file}")

    # NEW: merge small segments into bigger blocks first
    merged_blocks = merge_chunks(content["chunks"], max_duration=30)

    texts_to_embed = [f"{c['title']}: {c['text']}" for c in merged_blocks]
    embeddings = create_embedding(texts_to_embed)

    for i, block in enumerate(merged_blocks):
        block["chunk_id"] = chunk_id
        block["embedding"] = embeddings[i]
        chunk_id += 1
        my_dicts.append(block)

df = pd.DataFrame.from_records(my_dicts)
df.to_pickle("chunks_with_embeddings.pkl")
print("Saved! Total chunks:", len(df))