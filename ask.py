import requests
import pandas as pd
import numpy as np
from google import genai
from dotenv import load_dotenv
import os

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })
    return r.json()["embeddings"]

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

df = pd.read_pickle("chunks_with_embeddings.pkl")

question = input("Ask your question: ")
question_embedding = create_embedding([question])[0]

df["similarity"] = df["embedding"].apply(lambda emb: cosine_similarity(emb, question_embedding))
top_results = df.sort_values("similarity", ascending=False).head(5)

# --- Expand each top match with neighboring chunks, staying within the same video ---
WINDOW = 4

expanded_chunks_list = []
for chunk_id in top_results["chunk_id"]:
    match_row = df[df["chunk_id"] == chunk_id].iloc[0]
    same_video = df[df["number"] == match_row["number"]]
    nearby = same_video[
        (same_video["chunk_id"] >= chunk_id - WINDOW) &
        (same_video["chunk_id"] <= chunk_id + WINDOW)
    ]
    expanded_chunks_list.append(nearby)

expanded_chunks = pd.concat(expanded_chunks_list).drop_duplicates(subset="chunk_id").sort_values("chunk_id")

print("\n--- Expanded context chunks ---\n")
for _, row in expanded_chunks.iterrows():
    print(f"[{row['title']}] (id={row['chunk_id']}) {row['text']}")

context = " ".join(expanded_chunks["text"].tolist())

prompt = f"""You are a helpful teaching assistant for a Python data science course.
Answer the student's question using ONLY the context below. If the context doesn't
contain the answer, say you don't have enough information.

Context:
{context}

Question: {question}

Answer:"""

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt
)

print("\n--- Answer ---\n")
print(response.text)