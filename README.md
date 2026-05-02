# AI Contract Analysis System

A RAG-based system that lets you upload legal contracts and ask questions about them. Answers are grounded strictly in the uploaded documents.

## Tech Stack

- **LLM** — Gemini 2.5 Flash (free tier)
- **Embeddings** — sentence-transformers/all-MiniLM-L6-v2 (local)
- **Vector DB** — ChromaDB (local persistence)
- **Backend** — FastAPI
- **Supported formats** — PDF, TXT, DOCX

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Pratyaksh-Singhal/Contract-Analysis
cd contract-analyser
```
### 2. Create and activate Virtual Environment 

```bash
python -m venv .venv

#Activate virtual environment
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Get a Gemini API Key

Go to [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) and create a free API key.

### 5. Set the API key

```bash
# Mac/Linux
export GEMINI_API_KEY=your_key_here

# Windows
set GEMINI_API_KEY=your_key_here
```

### 6. Run the server

```bash
python server.py
```

Open **http://localhost:8000** in your browser.

---

## How to Use

1. Drop a contract (PDF, TXT, or DOCX) into the upload area or drag and drop it
2. Click **Ingest contracts/ folder** if you already have files in the `contracts/` directory
3. Type your question in the chat and press Send
4. The system answers based only on the content of your uploaded documents

---

## CLI (optional)

If you prefer the terminal:

```bash
python main.py ingest        # ingest all files from contracts/
python main.py query         # start interactive Q&A
python main.py status        # check system status
python main.py clear         # wipe all ingested documents
```

---

## Project Structure

```
contract-analyzer/
├── contracts/           # drop your contract files here
├── ui/                  # frontend (single HTML file)
├── src/
│   ├── config.py        # all settings
│   ├── interfaces.py    # abstract base classes
│   ├── loaders.py       # PDF, TXT, DOCX loaders
│   ├── chunker.py       # text splitting
│   ├── vector_store.py  # ChromaDB wrapper
│   ├── ingestion.py     # ingestion pipeline
│   ├── llm.py           # Gemini and Ollama clients
│   ├── prompt_builder.py
│   └── query_engine.py
├── server.py            # FastAPI server
└── main.py              # CLI entry point
```

---

## Switching to Ollama (fully offline)

If you want to run without a Gemini key, update `src/config.py`:

```python
provider: str = "ollama"
model_name: str = "mistral"
```

Install Ollama from [ollama.com](https://ollama.com), then:

```bash
ollama pull mistral
ollama serve
```

Then run `python server.py` as normal.
