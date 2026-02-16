import os
import pickle
import faiss
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq

# Load environment variables (only needed for local testing)
load_dotenv()

# ------------------------------------------------------------------
# Cached resources – load once and reuse across sessions
# ------------------------------------------------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_vector_store():
    index = faiss.read_index("./rag_vectorstore.faiss")
    with open("./rag_vectorstore.pkl", "rb") as f:
        docs = pickle.load(f)
    return index, docs

@st.cache_resource
def load_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY not set. Please add it to your environment.")
        st.stop()
    return ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0.1
    )

# ------------------------------------------------------------------
# Retrieval function
# ------------------------------------------------------------------
def retrieve_docs(query, embedding_model, index, docs, top_k=3):
    query_emb = embedding_model.encode([query])[0].astype('float32').reshape(1, -1)
    distances, indices = index.search(query_emb, top_k)
    results = []
    for i in indices[0]:
        if i != -1:
            doc = docs[i]
            results.append({
                'content': doc.page_content,
                'metadata': doc.metadata
            })
    return results

# ------------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------------
st.set_page_config(page_title="Medical RAG", page_icon="🩺")
st.title("🩺 Medical RAG Q&A System")

# Load cached resources
with st.spinner("Loading models and vector store..."):
    embedding_model = load_embedding_model()
    index, documents = load_vector_store()
    llm = load_llm()

st.success("Ready! Ask a medical question.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("View sources"):
                for src in msg["sources"]:
                    st.caption(f"**{src['source']}** (page {src['page']})")
                    st.text(src["preview"])

# Chat input
if prompt := st.chat_input("Ask a question..."):
    # User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieve documents
    with st.spinner("Searching knowledge base..."):
        retrieved = retrieve_docs(prompt, embedding_model, index, documents)

    if not retrieved:
        answer = "No relevant documents found."
        sources = []
    else:
        context = "\n\n".join([d['content'] for d in retrieved])
        prompt_for_llm = f"""Use the following context to answer the question concisely.

Context:
{context}

Question: {prompt}

Answer:"""
        with st.spinner("Generating answer..."):
            response = llm.invoke(prompt_for_llm)
            answer = response.content

        sources = []
        for i, doc in enumerate(retrieved):
            sources.append({
                "source": doc['metadata'].get('source', 'unknown'),
                "page": doc['metadata'].get('page', 'N/A'),
                "preview": doc['content'][:200] + "..."
            })

    # Assistant response
    with st.chat_message("assistant"):
        st.markdown(answer)
        if sources:
            with st.expander("View sources"):
                for src in sources:
                    st.caption(f"**{src['source']}** (page {src['page']})")
                    st.text(src["preview"])

    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
