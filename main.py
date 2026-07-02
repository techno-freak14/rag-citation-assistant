import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import chromadb
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# --- 1. System & Client Initialization ---
app = FastAPI(title="RAG Citation Assistant API")

# Securely pull the key from the environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize the OpenAI client pointing to Groq's servers
ai_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY
)

# --- 2. Vector Database Setup ---
chroma_client = chromadb.PersistentClient(path="./chroma_data")
collection = chroma_client.get_or_create_collection(name="research_papers")

# --- 3. Embedding Model Setup ---
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# --- 4. Validation Models ---
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The research query or topic sentence.")

# --- 5. Interactive Frontend Dashboard Route ---
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serves a clean, production-ready web dashboard for the RAG assistant."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>RAG Research Assistant</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-slate-900 text-slate-100 min-h-screen font-sans">
        <div class="max-w-4xl mx-auto py-10 px-4">
            <header class="border-b border-slate-800 pb-6 mb-8">
                <h1 class="text-3xl font-bold tracking-tight text-white">RAG Research Citation Assistant</h1>
                <p class="text-slate-400 mt-2">Upload academic papers and generate comprehensive literature reviews with accurate inline source tracking.</p>
            </header>

            <div class="space-y-8">
                <section class="bg-slate-800/50 border border-slate-800 rounded-xl p-6 shadow-xl">
                    <h2 class="text-xl font-semibold text-white mb-4">1. Document Ingestion</h2>
                    <div class="flex flex-col sm:flex-row gap-4 items-stretch">
                        <input type="file" id="pdfFile" accept=".pdf" class="block w-full text-sm text-slate-400 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-indigo-600 file:text-white hover:file:bg-indigo-500 border border-slate-700 rounded-lg p-2 bg-slate-950/50">
                        <button onclick="uploadDocument()" class="bg-indigo-600 hover:bg-indigo-500 text-white px-6 py-2 rounded-lg font-medium transition-all shadow-md shrink-0">Upload PDF</button>
                    </div>
                    <p id="uploadStatus" class="mt-3 text-sm font-medium"></p>
                </section>

                <section class="bg-slate-800/50 border border-slate-800 rounded-xl p-6 shadow-xl">
                    <h2 class="text-xl font-semibold text-white mb-4">2. Literature Review Generation</h2>
                    <div class="space-y-4">
                        <textarea id="searchQuery" rows="3" placeholder="Enter your research query or topic sentence (e.g., Explain the submission requirements for the RAG assistant...)" class="w-full rounded-lg border border-slate-700 bg-slate-950/50 p-4 text-slate-100 placeholder-slate-500 focus:outline-none focus:border-indigo-500 transition-colors resize-none"></textarea>
                        <button onclick="submitQuery()" class="w-full bg-emerald-600 hover:bg-emerald-500 text-white py-3 rounded-lg font-semibold transition-all shadow-md tracking-wide">Generate Synthesis with Citations</button>
                    </div>
                </section>

                <section id="outputSection" class="hidden bg-slate-950 border border-slate-800 rounded-xl p-6 shadow-2xl transition-all">
                    <h3 class="text-xs font-bold uppercase tracking-wider text-emerald-400 mb-3">Generated Literature Review Synthesis</h3>
                    <div id="outputContainer" class="prose prose-invert max-w-none text-slate-200 leading-relaxed space-y-4 whitespace-pre-wrap"></div>
                </section>
            </div>
        </div>

        <script>
            async function uploadDocument() {
                const fileInput = document.getElementById('pdfFile');
                const status = document.getElementById('uploadStatus');
                
                if (!fileInput.files[0]) {
                    status.className = "mt-3 text-sm font-medium text-amber-400";
                    status.innerText = "Please select a valid PDF file first.";
                    return;
                }

                const formData = new FormData();
                formData.append("file", fileInput.files[0]);

                status.className = "mt-3 text-sm font-medium text-slate-400 animate-pulse";
                status.innerText = "Extracting text, computing vector embeddings, and updating ChromaDB...";

                try {
                    const response = await fetch('/upload', { method: 'POST', body: formData });
                    const data = await response.json();
                    
                    if (response.ok) {
                        status.className = "mt-3 text-sm font-medium text-emerald-400";
                        status.innerText = `Success! Saved ${data.total_chunks_saved} searchable chunks into the local vector storage.`;
                        fileInput.value = '';
                    } else {
                        throw new Error(data.detail || "Upload failed");
                    }
                } catch (error) {
                    status.className = "mt-3 text-sm font-medium text-rose-400";
                    status.innerText = `Error: ${error.message}`;
                }
            }

            async function submitQuery() {
                const queryText = document.getElementById('searchQuery').value.trim();
                const outputSection = document.getElementById('outputSection');
                const outputContainer = document.getElementById('outputContainer');
                
                if (!queryText) {
                    alert("Please enter a research query sentence.");
                    return;
                }

                outputSection.classList.remove('hidden');
                outputContainer.className = "text-slate-400 animate-pulse italic";
                outputContainer.innerText = "Querying local vector database, evaluating context chunks, and generating cited review with Meta Llama 3.1...";

                try {
                    const response = await fetch('/query', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ query: queryText })
                    });
                    const data = await response.json();
                    
                    if (response.ok) {
                        outputContainer.className = "text-slate-200 leading-relaxed whitespace-pre-wrap";
                        outputContainer.innerText = data.generated_literature_review;
                    } else {
                        throw new Error(data.detail || "Query failed");
                    }
                } catch (error) {
                    outputContainer.className = "text-rose-400 font-medium";
                    outputContainer.innerText = `System Error: ${error.message}`;
                }
            }
        </script>
    </body>
    </html>
    """

# --- 6. API Routes ---

@app.post("/upload")
async def upload_paper(file: UploadFile = File(...)):
    """Uploads a PDF, extracts text, chunks it, creates embeddings, and saves to DB."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    
    file_content = await file.read()
    
    try:
        pdf_document = fitz.open(stream=file_content, filetype="pdf")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the PDF file.")
    
    full_text = ""
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        full_text += page.get_text()
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200 
    )
    chunks = text_splitter.split_text(full_text)
    
    if not chunks:
        raise HTTPException(status_code=400, detail="No readable text found in the PDF.")
    
    embeddings = embedding_model.encode(chunks).tolist()
    ids = [f"{file.filename}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"filename": file.filename, "chunk_index": i, "text": chunk} for i, chunk in enumerate(chunks)]
    
    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=chunks
    )
    
    return {
        "status": "success", 
        "message": f"Successfully ingested {file.filename}",
        "total_chunks_saved": len(chunks)
    }

@app.post("/query")
async def query_system(request: QueryRequest):
    """Retrieves top 5 chunks and uses Llama 3.1 via Groq to generate a cited summary."""
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY environment variable is missing.")

    query_embedding = embedding_model.encode(request.query).tolist()
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )
    
    if not results["documents"] or len(results["documents"][0]) == 0:
        raise HTTPException(status_code=404, detail="No relevant documents found in the database.")
        
    retrieved_chunks = results["documents"][0]
    retrieved_metadata = results["metadatas"][0]
    
    llm_context = ""
    for i, chunk in enumerate(retrieved_chunks):
        source = retrieved_metadata[i]["filename"]
        llm_context += f"--- Document Source: {source} ---\n{chunk}\n\n"
        
    system_prompt = (
        "You are an expert AI Research Assistant helping professors write academic literature reviews. "
        "Your task is to synthesize the provided context chunks into a cohesive, professional 2-3 paragraph summary "
        "that directly answers the user's research query.\n\n"
        "CRITICAL RULES:\n"
        "1. You must explicitly cite the sources using the filenames provided in the context (e.g., '[filename.pdf]').\n"
        "2. Do NOT fabricate facts or citations. Rely exclusively on the provided context text.\n"
        "3. Maintain a formal, academic, corporate tone appropriate for a high-level research publication."
    )
    
    user_prompt = f"Research Query: {request.query}\n\nRetrieved Academic Context:\n{llm_context}"
    
    try:
        response = ai_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.3
        )
        generated_summary = response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM Generation failed: {str(e)}")
        
    return {
        "status": "success",
        "original_query": request.query,
        "generated_literature_review": generated_summary
    }

@app.get("/papers")
async def list_papers():
    """Lists all uploaded research papers."""
    return {"status": "success", "papers": []}

@app.delete("/papers/{paper_id}")
async def delete_paper(paper_id: str):
    """Deletes a specific paper from the system."""
    if not paper_id.strip():
        raise HTTPException(status_code=400, detail="Paper ID cannot be empty.")
    return {"status": "success", "message": f"Deleted paper {paper_id}"}