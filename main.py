import os
import re
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

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

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
def serve_frontend():
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
def upload_paper(file: UploadFile = File(...)):
    """Uploads a PDF, extracts text page-by-page, creates embeddings, and upserts to DB."""
    # Priority 5 Fix: Case-insensitive check
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    
    # Priority 3 Fix: Read synchronously within standard def block
    file_content = file.file.read()
    
    # Priority 5 Fix: Hard size limit set to 10MB to prevent memory exhaustion
    if len(file_content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large. Maximum allowed size is 10MB.")
    
    try:
        pdf_document = fitz.open(stream=file_content, filetype="pdf")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the PDF file.")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200 
    )
    
    chunks = []
    metadatas = []
    
    # Priority 1 Fix: Process page-by-page to preserve boundaries
    for page_num in range(len(pdf_document)):
        page_text = pdf_document.load_page(page_num).get_text()
        page_chunks = text_splitter.split_text(page_text)
        
        for chunk in page_chunks:
            chunks.append(chunk)
            # Store the specific page number in metadata (1-indexed). 
            # Priority 5 Fix: Removed duplicate text payload from metadata.
            metadatas.append({"filename": file.filename, "page": page_num + 1})
            
    if not chunks:
        raise HTTPException(status_code=400, detail="No readable text found in the PDF.")
    
    # Priority 3 Fix: This blocking ML computation is now safely off the event loop
    embeddings = embedding_model.encode(chunks).tolist()
    ids = [f"{file.filename}_chunk_{i}" for i in range(len(chunks))]
    
    # Priority 5 Fix: Using upsert instead of add to handle duplicate re-uploads cleanly
    collection.upsert(
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
def query_system(request: QueryRequest):
    """Retrieves chunks, filters by relevance, generates a summary, and grounds citations."""
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY environment variable is missing.")

    query_embedding = embedding_model.encode(request.query).tolist()
    
    # Priority 5 Fix: Include distances to establish a relevance threshold
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        include=["documents", "metadatas", "distances"]
    )
    
    if not results["documents"] or len(results["documents"][0]) == 0:
        raise HTTPException(status_code=404, detail="No relevant documents found in the database.")
        
    retrieved_chunks = results["documents"][0]
    retrieved_metadata = results["metadatas"][0]
    retrieved_distances = results["distances"][0]
    
    llm_context = ""
    valid_source_indices = []
    
    # ChromaDB defaults to L2 distance where smaller is closer.
    DISTANCE_THRESHOLD = 1.6 
    
    # Format numbered sources for the LLM while filtering out noise
    for i, chunk in enumerate(retrieved_chunks):
        if retrieved_distances[i] > DISTANCE_THRESHOLD:
            continue
            
        source_idx = i + 1
        valid_source_indices.append(str(source_idx))
        filename = retrieved_metadata[i]["filename"]
        page = retrieved_metadata[i].get("page", "Unknown")
        
        # Priority 1 Fix: Explicit source indexing with page numbers
        llm_context += f"[Source {source_idx}] {filename}, p.{page}\n{chunk}\n\n"
        
    # If all chunks exceeded the threshold, intercept the generation
    if not valid_source_indices:
        return {
            "status": "success",
            "original_query": request.query,
            "generated_literature_review": "No highly relevant information was found in the uploaded documents to confidently answer this query."
        }
        
    system_prompt = (
        "You are an expert AI Research Assistant helping professionals write academic literature reviews. "
        "Synthesize the provided context chunks into a cohesive, professional 2-3 paragraph summary "
        "that directly answers the user's research query.\n\n"
        "CRITICAL RULES:\n"
        "1. You must explicitly cite the sources using the EXACT numbered tags provided (e.g., '[Source 1]').\n"
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
        
    # Priority 1 Fix: The Grounding Check Validation
    cited_indices = re.findall(r'\[Source (\d+)\]', generated_summary)
    
    for index in set(cited_indices):
        if index not in valid_source_indices:
            generated_summary += f"\n\n[SYSTEM WARNING: The model hallucinated an unverified citation: Source {index}]"
            
    return {
        "status": "success",
        "original_query": request.query,
        "generated_literature_review": generated_summary
    }

@app.get("/papers")
def list_papers():
    """Lists all uploaded research papers by extracting unique filenames from metadata."""
    # Priority 2 Fix: Endpoint executes a real database extraction
    data = collection.get(include=["metadatas"])
    filenames = sorted({m["filename"] for m in data["metadatas"] if m is not None})
    return {"status": "success", "papers": filenames}

@app.delete("/papers/{paper_id}")
def delete_paper(paper_id: str):
    """Deletes a specific paper and all its corresponding chunks from the system."""
    # Priority 2 Fix: Endpoint executes a real database deletion
    if not paper_id.strip():
        raise HTTPException(status_code=400, detail="Paper ID cannot be empty.")
    collection.delete(where={"filename": paper_id})
    return {"status": "success", "message": f"Deleted paper {paper_id}"}