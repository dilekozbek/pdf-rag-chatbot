# PDF RAG Chatbot

Upload a PDF, ask questions, get answers grounded in the document.
Built with Streamlit + ChromaDB + Gemini.

## Demo

Live: _coming soon_

## Features

- PDF upload and chunking
- Semantic search via Gemini Embeddings
- Source attribution (which page the answer came from)
- Tested in Turkish

## Architecture

```
[PDF] → [pypdf] → [RecursiveCharacterTextSplitter]
                              ↓
                   [Gemini Embeddings]
                              ↓
                       [ChromaDB]
                              ↓
[Question] → [Embed] → [Top-K Retrieval] → [Gemini 2.5 Flash] → [Answer]
```

## Stack

- **Frontend:** Streamlit
- **Runtime:** Python 3.12
- **LLM:** Gemini 2.5 Flash (chat) + Gemini Embedding 001 (vectors)
- **Vector DB:** ChromaDB (persistent, local)
- **Chunking:** LangChain `RecursiveCharacterTextSplitter`

## Setup

```bash
git clone <repo>
cd pdf-rag-chatbot
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
echo "GEMINI_API_KEY=your_key" > .env
streamlit run app.py
```

## Notes

- **Chunking.** Char-based was rough — cut mid-word. Sentence-aware better. Settled on LangChain's `RecursiveCharacterTextSplitter`.
- **Embedding.** Default MiniLM weak in Turkish (couldn't answer "who wrote this PDF?"). Switched to Gemini Embedding 001 — top-K dropped from 8 to 3.
- **Grounding.** Forced "use only context, otherwise say no info" via prompt. Without it the model mixes its training data in.
- **Source pages.** Each chunk indexed with page number. Shown in UI.


