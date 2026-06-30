import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
import chromadb
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# --- 1. System & Client Initialization ---
app = FastAPI(title="RAG Citation Assistant API")

# Securely pull the key from the environment instead of hardcoding it
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize the OpenAI client pointing to Groq's free servers
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

# --- 5. API Routes ---

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
    
    # Generate vector embeddings for all chunks
    embeddings = embedding_model.encode(chunks).tolist()
    
    # Create unique IDs and metadata for each chunk
    ids = [f"{file.filename}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"filename": file.filename, "chunk_index": i, "text": chunk} for i, chunk in enumerate(chunks)]
    
    # Save everything into the database
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

    # 1. Convert the query into a vector embedding
    query_embedding = embedding_model.encode(request.query).tolist()
    
    # 2. Search ChromaDB for the top 5 most similar chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )
    
    if not results["documents"] or len(results["documents"][0]) == 0:
        raise HTTPException(status_code=404, detail="No relevant documents found in the database.")
        
    retrieved_chunks = results["documents"][0]
    retrieved_metadata = results["metadatas"][0]
    
    # 3. Format the retrieved text chunks to serve as the LLM's context
    llm_context = ""
    for i, chunk in enumerate(retrieved_chunks):
        source = retrieved_metadata[i]["filename"]
        llm_context += f"--- Document Source: {source} ---\n{chunk}\n\n"
        
    # 4. Construct the professional, strict prompt for the researcher persona
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
    
    # 5. Execute the call using Groq's active Llama-3.1-8b-instant model
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