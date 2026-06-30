# RAG Research Citation Assistant API

A high-performance Retrieval-Augmented Generation (RAG) backend built with FastAPI. It allows users to parse academic PDFs, store vector embeddings locally, and generate cited literature reviews.

## Tech Stack
* **Framework:** FastAPI (Python)
* **Vector Database:** ChromaDB (Local, persisted to disk)
* **Embedding Model:** SentenceTransformers (`all-MiniLM-L6-v2`)
* **LLM Generation Layer:** Meta Llama-3.1-8b-instant (via Groq Cloud SDK)

## Architectural Decision Notes (API Fallback)
The task brief specified `gpt-4o-mini` via the OpenAI API. Due to platform restrictions regarding developer trial accounts and billing paywalls, the generation layer was successfully configured to use **Groq's Llama-3.1-8b-instant API**. 
* The implementation utilizes the standard `openai` Python SDK interface, ensuring strict architectural compatibility.
* Swapping back to an enterprise OpenAI endpoint requires changing only the `base_url` and the `model` string.

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/techno-freak14/rag-citation-assistant.git](https://github.com/techno-freak14/rag-citation-assistant.git)
   cd rag-citation-assistant
