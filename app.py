import streamlit as st
import requests
import pandas as pd
import numpy as np
from google import genai
from dotenv import load_dotenv
import os


# --- Configuration ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

@st.cache_data
def load_data():
    return pd.read_pickle("chunks_with_embeddings.pkl")

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

def get_answer(question, df, window=10, top_k=5, similarity_threshold=0.5):
    question_embedding = create_embedding([question])[0]
    df["similarity"] = df["embedding"].apply(lambda emb: cosine_similarity(emb, question_embedding))
    top_results = df.sort_values("similarity", ascending=False).head(top_k)

    best_score = top_results["similarity"].max()
    if best_score < similarity_threshold:
        return (
            "This topic doesn't seem to be covered in the current course videos. "
            "Try asking about topics like variables, data types, strings, typecasting, "
            "user input, or the calculator exercise.",
            []
        )

    expanded_chunks_list = []
    for chunk_id in top_results["chunk_id"]:
        match_row = df[df["chunk_id"] == chunk_id].iloc[0]
        same_video = df[df["number"] == match_row["number"]]
        nearby = same_video[
            (same_video["chunk_id"] >= chunk_id - window) &
            (same_video["chunk_id"] <= chunk_id + window)
        ]
        expanded_chunks_list.append(nearby)

    expanded_chunks = pd.concat(expanded_chunks_list).drop_duplicates(subset="chunk_id").sort_values("chunk_id")
    context = " ".join(expanded_chunks["text"].tolist())
    sources = expanded_chunks["title"].unique().tolist()

    prompt = f"""You are a helpful teaching assistant for a Python data science course.
Answer the student's question using ONLY the context below. If the context doesn't
contain the answer, say you don't have enough information.

Context:
{context}

Question: {question}

Answer:"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text, sources
    except Exception as e:
        return f"⚠️ Sorry, the AI service is temporarily unavailable. Please try again in a moment.\n\n(Error: {e})", []
    
# --- Streamlit UI ---
st.set_page_config(page_title="AI Teaching Assistant", page_icon="🎓")
st.title("🎓 AI Teaching Assistant")
st.caption("Ask questions about the Python Data Science course")

df = load_data()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask a question about the course...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, sources = get_answer(question, df)
            st.markdown(answer)
            if sources:
                st.caption(f"📚 Sources: {', '.join(sources)}")

    st.session_state.messages.append({"role": "assistant", "content": answer})