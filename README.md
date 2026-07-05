# 🎓 AI Teaching Assistant (RAG-based)

A Retrieval-Augmented Generation (RAG) system that answers questions about a Python
programming course by retrieving relevant context from course video transcripts and
generating grounded, accurate answers using Gemini.

## How it works

1. **Data collection**: Course videos (Hindi) downloaded via yt-dlp
2. **Transcription & Translation**: Audio converted to English text using faster-whisper
3. **Chunking**: Transcripts split into ~30-second contextual blocks per video
4. **Embedding**: Each chunk embedded using `bge-m3` (via Ollama, local)
5. **Retrieval**: User questions matched against chunks via cosine similarity
6. **Generation**: Gemini generates answers grounded strictly in retrieved context

## Tech stack

- **Transcription**: faster-whisper (GPU-accelerated via Google Colab)
- **Embeddings**: bge-m3 via Ollama (local, free)
- **Vector search**: Cosine similarity (numpy/pandas)
- **Generation**: Google Gemini 2.5 Flash
- **UI**: Streamlit

## Setup

1. Clone this repo
2. Install dependencies: `pip install -r requirements.txt`
3. Install [Ollama](https://ollama.com) and pull the embedding model: `ollama pull bge-m3`
4. Get a free [Gemini API key](https://aistudio.google.com/apikey)
5. Create a `.env` file with: `GEMINI_API_KEY=your_key_here`
6. Add your own course audio to an `audios/` folder and run the transcription/embedding
   pipeline (see `create_chunks.py`/`process_audio.py` and `create_embeddings.py`)
7. Run the app: `streamlit run app.py`

## Data source

Sample data used for this project's demo consists of Hindi Python programming
tutorial videos from [Code With Harry](https://www.youtube.com/@CodeWithHarry)'s
YouTube channel, used solely for personal learning and demonstration purposes.
Audio/transcript files are not included in this repository — only the pipeline
code that processes them. To run this project with your own data, provide your
own audio files (see Setup below).

## Scope note 

This demo is trained on 13 introductory Python videos covering variables, data types,
strings, typecasting, and user input. Questions outside this scope will correctly
return a "not covered" message rather than a fabricated answer — a deliberate design
choice to keep answers grounded and trustworthy.

## Key learnings

- Chunk granularity matters: individual sentence-level chunks caused irrelevant/filler
  content to outrank meaningful explanations. Merging into larger contextual blocks
  fixed this.
- Prepending video titles to chunk text before embedding improved topic-level retrieval.
- Similarity thresholding prevents hallucinated answers on out-of-scope questions.

## Limitations

This is a learning project built to demonstrate a working RAG pipeline end-to-end,
not a production-grade system. Some known limitations, and why they exist:

### 1. Limited knowledge scope

The assistant currently only knows about **13 introductory Python videos** (variables,
data types, strings, typecasting, user input, and a calculator exercise). Questions
outside this scope will correctly return a "not covered" message rather than a
fabricated answer. This is intentional — a similarity threshold check prevents the
system from hallucinating answers when nothing relevant exists in the knowledge base,
prioritizing trustworthiness over always having _an_ answer.

### 2. Translation quality (Hindi → English)

The source videos are in Hindi and were translated to English using Whisper's built-in
translation mode. Translation quality is generally good but imperfect, especially with
technical/programming vocabulary and "Hinglish" (English terms spoken within Hindi
sentences), which is common in Indian tech tutorials. Some nuance from the original
explanation may be lost or slightly reworded.

### 3. Chunking granularity trade-off

Early versions of this project chunked transcripts at the individual speech-segment
level (single sentences from Whisper). This caused short, generic phrases (e.g. "What
is it?", "How do you do it?") to sometimes outrank substantive explanations in
retrieval, since embedding similarity doesn't account for how _informative_ a sentence
is, only how semantically close it is to the query. This was fixed by merging
consecutive segments into ~30-second contextual blocks before embedding, which gives
retrieval more complete, self-contained context to work with. Very short or unusually
phrased questions may still occasionally retrieve suboptimal chunks.

### 4. No conversation memory in retrieval

Each question is treated independently for retrieval purposes — follow-up questions
like "what about the second one?" won't correctly resolve pronouns/references to
prior turns, even though the chat UI displays conversation history visually.

### 5. Runs on local + free-tier infrastructure

Embeddings run locally via Ollama (`bge-m3`), and generation uses Gemini's free tier,
which is occasionally subject to temporary rate limits or server availability issues
(handled gracefully with a fallback message, not a crash). This was a deliberate choice
to keep the project free to run and reproduce, at the cost of occasional latency or
temporary unavailability compared to paid, dedicated infrastructure.

## Possible future improvements

- Expand the dataset to cover more course topics
- Experiment with hybrid search (keyword + semantic) to catch exact-term queries better
- Add conversational memory for follow-up questions
- Evaluate retrieval quality more rigorously (e.g. a small labeled eval set)
