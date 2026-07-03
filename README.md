# RAG Research Citation Assistant API

A verifiable, high-performance Retrieval-Augmented Generation (RAG) backend built with FastAPI. It allows researchers to upload academic PDFs, store vector embeddings locally, and generate explicitly grounded literature reviews with page-level citations.

## 🚀 Key Features
* **Verifiable Citations:** Tracks page boundaries during ingestion and runs a post-generation regex grounding check to intercept and flag AI hallucinations.
* **High-Performance Execution:** CPU-bound Machine Learning tasks (text embedding, chunking) are executed in synchronous worker threads to prevent event loop blocking, enabling true concurrent request handling.
* **Integrated Frontend:** Includes a responsive, dark-mode single-page application (SPA) dashboard served directly from the root route.
* **Persistent Local Vector Storage:** Utilizes ChromaDB to persist document embeddings securely to disk without requiring external cloud databases.

## 🛠️ Tech Stack
* **Framework:** FastAPI (Python)
* **Vector Database:** ChromaDB (Local, persisted to disk)
* **Embedding Model:** SentenceTransformers (`all-MiniLM-L6-v2`)
* **Ingestion:** PyMuPDF (`fitz`) & LangChain Text Splitters
* **LLM Generation Layer:** Meta Llama-3.1-8b-instant (via Groq Cloud SDK)

## 📐 Architectural Decision Notes (ADR)
**1. LLM API Fallback:** The initial task brief specified `gpt-4o-mini` via the OpenAI API. Due to platform restrictions regarding developer trial accounts and billing paywalls, the generation layer was pivoted to Groq's Llama 3.1. 
* *Implementation:* The system still utilizes the standard `openai` Python SDK interface, ensuring strict architectural compatibility. Swapping back to an enterprise OpenAI endpoint requires changing only the `base_url` and the `model` string.

**2. Asynchronous Unblocking:**
To prevent ML vector encoding from freezing the ASGI event loop, the `/upload` and `/query` routes were deliberately designed as standard synchronous `def` functions rather than `async def`. This instructs FastAPI to run the blocking CPU workloads in an external threadpool.

## ⚙️ Setup & Installation

**1. Clone the repository:**
```bash
git clone [https://github.com/techno-freak14/rag-citation-assistant.git](https://github.com/techno-freak14/rag-citation-assistant.git)
cd rag-citation-assistant
2. Create and activate a virtual environment:
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
3. Install dependencies:
pip install -r requirements.txt
4.Configure Environment Variables:
Create a .env file in the root directory (or copy the example file) and add your API key:
cp .env.example .env
# Edit .env and set GROQ_API_KEY=your_key_here
5. Start the Server:
python -m uvicorn main:app --reload --env-file .env

##📡 API Endpoints
GET / - Serves the interactive web UI dashboard.

POST /upload - Ingests a PDF, chunks it by page, and upserts to ChromaDB.

POST /query - Retrieves semantic context and generates a grounded, cited summary.

GET /papers - Lists all successfully ingested documents.

DELETE /papers/{paper_id} - Purges a document's embeddings from the database.
