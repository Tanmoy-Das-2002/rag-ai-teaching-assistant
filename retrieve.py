import requests
import pandas as pd
import numpy as np

def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })
    embedding = r.json()["embeddings"]
    return embedding

# Load your saved chunks + embeddings
df = pd.read_pickle("chunks_with_embeddings.pkl")

# Take user's question
question = input("Ask your question: ")

# Embed the question
question_embedding = create_embedding([question])[0]

# Compute similarity between question and every chunk
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

df["similarity"] = df["embedding"].apply(lambda emb: cosine_similarity(emb, question_embedding))

# Get top 5 most relevant chunks
top_results = df.sort_values("similarity", ascending=False).head(5)

print("\nTop matching chunks:\n")
for _, row in top_results.iterrows():
    print(f"[{row['title']}] ({row['start']}s - {row['end']}s) similarity={row['similarity']:.3f}")
    print(row['text'])
    print("---")