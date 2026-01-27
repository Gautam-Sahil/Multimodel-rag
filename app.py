import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# LangChain Modern Core Imports
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

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
    # Using k=12 as per your requirement for better context
    retriever = docsearch.as_retriever(
        search_type="similarity", 
        search_kwargs={"k": 12}
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

CONTEXT:
{context}

QUESTION: {question}

YOUR ANSWER:"""

# --- 3. RAG Chain Setup (LCEL Style) ---
chat_model = ChatOpenAI(
    model="openai/gpt-4o-mini", 
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0.1,
    max_tokens=800
)

# Helper function to format retrieved documents into a single string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# The modern LCEL Chain
if retriever:
    prompt = ChatPromptTemplate.from_template(MULTIMODAL_SYSTEM_PROMPT)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | chat_model
        | StrOutputParser()
    )
    print("✅ LCEL RAG chain initialized successfully")
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
        
        if rag_chain is None:
            return jsonify({
                "answer": "System is currently initializing or Pinecone is disconnected.",
                "sources": []
            })
        
        # 1. Manually get docs first so we can extract sources for CitationManager
        retrieved_docs = retriever.invoke(msg)
        
        # 2. Invoke the chain for the answer
        answer = rag_chain.invoke(msg)
        
        # 3. Process sources using your existing CitationManager
        unique_sources = CitationManager.get_unique_sources(retrieved_docs)
        
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
    from pinecone import Pinecone
    try:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        indexes = [idx.name for idx in pc.list_indexes()]
        pinecone_ok = INDEX_NAME in indexes
        
        return jsonify({
            "status": "healthy" if rag_chain and pinecone_ok else "degraded",
            "pinecone_connection": "connected" if pinecone_ok else "disconnected",
            "rag_chain_status": "ready" if rag_chain else "initializing"
        })
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)
