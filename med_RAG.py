import os
import pickle
import faiss
import uvicorn
import logging
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq

# --- Configure logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Your custom classes (simplified) ---
class EmbeddingManager:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded")

    def generate_embeddings(self, texts):
        return self.model.encode(texts, show_progress_bar=False)

class VectorStore:
    def __init__(self, dimension=384):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.documents = []

    def add_documents(self, documents, embeddings):
        embeddings = embeddings.astype('float32')
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        self.index.add(embeddings)
        self.documents.extend(documents)

    def similarity_search(self, query_embedding, k=5):
        query = query_embedding.astype('float32').reshape(1, -1)
        distances, indices = self.index.search(query, k)
        return [self.documents[i] for i in indices[0] if i != -1]

    def save(self, path):
        faiss.write_index(self.index, f"{path}.faiss")
        with open(f"{path}.pkl", "wb") as f:
            pickle.dump(self.documents, f)

    @classmethod
    def load(cls, path, dimension=384):
        faiss_file = f"{path}.faiss"
        pkl_file = f"{path}.pkl"
        if not os.path.exists(faiss_file) or not os.path.exists(pkl_file):
            raise FileNotFoundError(
                f"Vector store files not found. Please run 'ingest.py' first to create them."
            )
        index = faiss.read_index(faiss_file)
        with open(pkl_file, "rb") as f:
            docs = pickle.load(f)
        store = cls(dimension)
        store.index = index
        store.documents = docs
        logger.info(f"Loaded {len(docs)} documents from vector store")
        return store

class RAGRetriever:
    def __init__(self, vector_store, embedding_manager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query, top_k=5):
        emb = self.embedding_manager.generate_embeddings([query])[0]
        docs = self.vector_store.similarity_search(emb, k=top_k)
        # Filter out docs with None content
        filtered = []
        for doc in docs:
            if hasattr(doc, 'page_content') and doc.page_content:
                filtered.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata
                })
        logger.info(f"Retrieved {len(filtered)} documents for query: {query[:50]}...")
        return filtered

# --- Initialize FastAPI app ---
load_dotenv()
app = FastAPI(title="RAG API")
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Global RAG components (loaded once at startup) ---
embedding_manager = None
vectorstore = None
retriever = None
llm = None

@app.on_event("startup")
def load_rag_components():
    global embedding_manager, vectorstore, retriever, llm
    try:
        logger.info("Loading embedding model...")
        embedding_manager = EmbeddingManager()
        
        logger.info("Loading vector store...")
        vectorstore = VectorStore.load("./rag_vectorstore", dimension=384)
        retriever = RAGRetriever(vectorstore, embedding_manager)
        
        logger.info("Loading LLM...")
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama-3.1-8b-instant",
            temperature=0.1
        )
        # Test LLM with a simple prompt
        test_response = llm.invoke("Hello")
        logger.info(f"LLM test response: {test_response.content[:50]}...")
        
        logger.info("All components ready!")
    except Exception as e:
        logger.exception("Failed to initialize RAG components")
        raise

# --- API Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve a simple chat interface."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if retriever and llm:
        return {"status": "healthy", "documents": len(vectorstore.documents) if vectorstore else 0}
    return {"status": "not ready"}
@app.post("/query")
@app.post("/get_response")
async def query_endpoint(request: Request):
    """Process a question and return answer with sources (accepts JSON)."""
    logger.info("Received POST request")
    
    # Check components
    if not retriever or not llm:
        logger.error("RAG components not loaded")
        return JSONResponse({"error": "RAG components not loaded"}, status_code=500)
    
    try:
        # Parse JSON body
        data = await request.json()
        question = data.get("question")
        if not question:
            return JSONResponse({"error": "Missing 'question' field"}, status_code=400)
    except Exception as e:
        logger.exception("Failed to parse JSON")
        return JSONResponse({"error": f"Invalid JSON: {str(e)}"}, status_code=400)
    
    logger.info(f"Received query: {question}")
    
    try:
        # Retrieve relevant documents
        docs = retriever.retrieve(question, top_k=3)
        if not docs:
            logger.warning("No relevant documents found")
            return JSONResponse({"answer": "No relevant documents found to answer your question."})
        
        # Build context
        context = "\n\n".join([d['content'] for d in docs])
        logger.info(f"Context length: {len(context)} characters")
        
        prompt = f"""Use the following context to answer the question concisely.

Context:
{context}

Question: {question}

Answer:"""
        
        # Call LLM
        response = llm.invoke(prompt)
        answer = response.content
        logger.info(f"Generated answer of length {len(answer)}")
        
        # Build sources list
        sources = []
        for i, doc in enumerate(docs):
            sources.append({
                "index": i+1,
                "source": doc['metadata'].get('source', 'unknown'),
                "page": doc['metadata'].get('page', 'N/A'),
                "preview": doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content']
            })
        
        return JSONResponse({
            "question": question,
            "answer": answer,
            "sources": sources
        })
        
    except Exception as e:
        logger.exception("Error processing query")
        return JSONResponse({"error": f"Internal error: {str(e)}"}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)