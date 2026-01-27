import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# LangChain & Vector Store
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Custom Modules
from src.helper import download_hugging_face_embeddings
from src.citation_manager import CitationManager

app = Flask(__name__)
load_dotenv()

# --- 1. Configuration & Initialization ---
INDEX_NAME = "multimodal-rag-v2"
embeddings = download_hugging_face_embeddings()

# Initialize Vector Store & Retriever
try:
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME,
        embedding=embeddings
    )
    retriever = docsearch.as_retriever(
        search_type="similarity", 
        search_kwargs={"k": 12}  # Increased k for better context
    )
    print(f"✅ Connected to Pinecone index: {INDEX_NAME}")
except Exception as e:
    print(f"❌ Error connecting to Pinecone: {e}")
    retriever = None

# --- 2. System Prompt Definition ---
MULTIMODAL_SYSTEM_PROMPT = """You are an expert financial document analyst. Your job is to find and explain specific data, figures, and tables in the document.

DOCUMENT STRUCTURE:
- Contains text, tables, and figures/charts.
- Figures are referenced as "Figure X: [Title]".
- Tables contain numerical data.
- Annexes contain detailed technical information.

DIRECTIONS:
1. SEARCH WIDELY: Look through all provided context for the specific Figure or Table requested.
2. NUMBERS ARE KEY: Extract specific percentages, growth rates, and years.
3. DESCRIBE VISUALS: If the context describes a figure (e.g., "Figure 3 shows a surplus"), report that analysis.
4. FALLBACK: If a specific figure isn't found, summarize the most relevant data from that same section.

CRITICAL INSTRUCTIONS:
- ALWAYS extract numbers and data.
- Reference exact figures, tables, and page numbers.
- ANALYZE trends (growth, decline, stability).
- CITE SOURCES explicitly within the text when possible.

CONTEXT PROVIDED:
{context}

USER QUESTION: {input}

YOUR ANSWER:"""

# --- 3. RAG Chain Setup ---
chat_model = ChatOpenAI(
    model_name="openai/gpt-4o-mini", 
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0.1,
    max_tokens=800
)

if retriever:
    prompt = ChatPromptTemplate.from_messages([
        ("system", MULTIMODAL_SYSTEM_PROMPT),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(chat_model, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    print("✅ RAG chain initialized successfully")
else:
    rag_chain = None
    print("⚠️ RAG chain not initialized due to missing retriever")

# --- 4. Routes ---

@app.route("/")
def index():
    return render_template("dashboard.html")

@app.route("/get", methods=["POST"])
def chat():
    try:
        msg = request.form.get("msg", "").strip()
        if not msg:
            return jsonify({"error": "No message received"}), 400
        
        print(f"📩 Incoming Query: {msg}")
        
        if rag_chain is None:
            return jsonify({
                "answer": "System is currently initializing or Pinecone is disconnected.",
                "sources": []
            })
        
        # Invoke RAG Chain
        response = rag_chain.invoke({"input": msg})
        answer = response.get("answer", "I couldn't generate an answer based on the document.")
        retrieved_docs = response.get("context", [])
        
        # Use CitationManager to extract and clean unique sources
        unique_sources = CitationManager.get_unique_sources(retrieved_docs)
        
        # Log for debugging
        print(f"🔍 Found {len(unique_sources)} unique sources.")

        return jsonify({
            "answer": answer,
            "sources": unique_sources,
            "success": True
        })
        
    except Exception as e:
        print(f"❌ Error in /get route: {e}")
        return jsonify({
            "error": str(e),
            "answer": "An error occurred while processing your request.",
            "sources": []
        }), 500

@app.route("/health", methods=["GET"])
def health():
    """Detailed health check endpoint"""
    from pinecone import Pinecone
    try:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        indexes = pc.list_indexes().names()
        pinecone_ok = INDEX_NAME in indexes
        
        return jsonify({
            "status": "healthy" if rag_chain and pinecone_ok else "degraded",
            "pinecone_connection": "connected" if pinecone_ok else "disconnected",
            "index_name": INDEX_NAME,
            "rag_chain_status": "ready" if rag_chain else "initializing",
            "environment": os.getenv("FLASK_ENV", "production")
        })
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

if __name__ == "__main__":
    # Hugging Face uses port 7860 by default
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)