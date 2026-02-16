# ingest.py
import os
import pickle
import faiss
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np

# ------------------------------------------------------------------
# 1. Load PDFs
# ------------------------------------------------------------------
print("Loading PDFs...")
# Path to your PDF folder – change if needed
pdf_path = "data/"   # relative to this script's location
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF folder not found: {pdf_path}")

loader = DirectoryLoader(pdf_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()
print(f"Loaded {len(documents)} pages")

# ------------------------------------------------------------------
# 2. Split documents into chunks
# ------------------------------------------------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)
chunks = text_splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks")

# ------------------------------------------------------------------
# 3. Generate embeddings
# ------------------------------------------------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')
texts = [doc.page_content for doc in chunks]
embeddings = model.encode(texts, show_progress_bar=True)
print(f"Embeddings shape: {embeddings.shape}")

# ------------------------------------------------------------------
# 4. Build FAISS index
# ------------------------------------------------------------------
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings.astype('float32'))

# ------------------------------------------------------------------
# 5. Save index and documents to disk
# ------------------------------------------------------------------
faiss.write_index(index, "./rag_vectorstore.faiss")
with open("./rag_vectorstore.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("✅ Vector store saved to ./rag_vectorstore.faiss and ./rag_vectorstore.pkl")