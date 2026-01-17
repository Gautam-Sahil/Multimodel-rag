
from flask import Flask, render_template, request, jsonify
from src.helper import SimplePDFProcessor, download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

app = Flask(__name__)
load_dotenv()

# Initialize components (using SimplePDFProcessor from helper_simple)
pdf_processor = SimplePDFProcessor()

MULTIMODAL_SYSTEM_PROMPT = """You are an expert financial document analyst.Your job is to find and explain the specific data, figures, and tables qatar_test_doc.pdf.

DOCUMENT STRUCTURE:
- The document contains text, tables, and figures/charts
- Figures are referenced as "Figure X: [Title]"
- Tables contain numerical data
- Annexes contain detailed technical information

DIRECTIONS:
1. **SEARCH WIDELY**: Look through all provided context for the specific Figure or Table requested.
2. **NUMBERS ARE KEY**: If you find the figure, extract the specific percentages and growth rates.
3. **DESCRIBE VISUALS**: If the context says "Figure 3 shows a surplus," report that even if the raw grid of numbers isn't there.
4. **FALLBACK**: If you absolutely cannot find the specific figure, summarize the closest relevant economic data from the same section (e.g., if you can't find Figure 3, talk about the External Sector trends mentioned in the text).

CRITICAL INSTRUCTIONS:
1. **ALWAYS EXTRACT NUMBERS AND DATA**: When you see numbers (percentages, years, values), include them in your answer
2. **NEVER SAY "I cannot find this information"**: Instead, extract whatever information IS available in the context
3. **BE SPECIFIC**: Reference exact figures, tables, and page numbers
4. **ANALYZE TRENDS**: Describe what the data indicates about trends, growth, changes
5. **CITE SOURCES**: Always mention where the information came from

CONTEXT EXAMPLES:
- If context mentions "Figure 3 shows GDP growth of 3.2% in 2023", say: "Figure 3 indicates GDP growth was 3.2% in 2023"
- If context has a table with numbers, extract and summarize the key numbers
- If context mentions an annex, explain what the annex covers based on available information

CONTEXT PROVIDED:
{context}

USER QUESTION: {input}

YOUR ANSWER (must include specific numbers and analysis):"""

# Setup embeddings and vector store
embeddings = download_hugging_face_embeddings()
index_name = "multimodal-rag-v2"  # Your working index

try:
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )
    retriever = docsearch.as_retriever(
        search_type="similarity", 
        search_kwargs={"k": 15}
    )
    print(f"✅ Connected to Pinecone index: {index_name}")
except Exception as e:
    print(f"❌ Error connecting to Pinecone: {e}")
    retriever = None

# Chat model
chat_model = ChatOpenAI(
    model_name="openai/gpt-4o-mini", 
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0.1,
    max_tokens=500
)

# Create RAG chain
if retriever:
    prompt = ChatPromptTemplate.from_messages([
        ("system", MULTIMODAL_SYSTEM_PROMPT),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(chat_model, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    print("✅ RAG chain initialized")
else:
    rag_chain = None
    print("⚠️  RAG chain not initialized")

class SimpleCitationManager:
    """Simple citation manager"""
    @staticmethod
    def get_unique_sources(retrieved_docs):
        """Get unique sources from retrieved documents"""
        unique_sources = set()
        
        for doc in retrieved_docs:
            if hasattr(doc, 'metadata'):
                metadata = doc.metadata
            else:
                continue
                
            source = metadata.get("source", "Document")
            page = metadata.get("page", "")
            element_type = metadata.get("element_type", "text")
            
            if element_type == "table":
                unique_sources.add(f"📊 Table on Page {page}")
            elif element_type == "image":
                unique_sources.add(f"🖼️ Image on Page {page}")
            else:
                unique_sources.add(f"📄 Page {page}")
        
        return list(unique_sources)

citation_manager = SimpleCitationManager()

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chat():
    try:
        msg = request.form.get("msg", "").strip()
        if not msg:
            return jsonify({"error": "No message received"}), 400
        
        print(f"📩 Query: {msg}")
        
        if rag_chain is None:
            return jsonify({
                "answer": "System is initializing. Please try again in a moment.",
                "sources": []
            })
        
        # Get response with context
        response = rag_chain.invoke({"input": msg})
        answer = response.get("answer", "No answer generated.")
        
        # Get retrieved documents
        retrieved_docs = response.get("context", [])
        
        # Add citations
        if retrieved_docs:
            unique_sources = citation_manager.get_unique_sources(retrieved_docs)
            
            if unique_sources:
                answer += "\n\n**Sources:**\n"
                for source in unique_sources[:3]:
                    answer += f"- {source}\n"
        
        return jsonify({
            "answer": answer,
            "sources": [],
            "success": True
        })
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({
            "error": str(e),
            "answer": "I'm having trouble processing your request.",
            "sources": []
        }), 500

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    from pinecone import Pinecone
    try:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        indexes = pc.list_indexes().names()
        pinecone_ok = index_name in indexes
        
        return jsonify({
            "status": "healthy" if rag_chain else "degraded",
            "pinecone": "connected" if pinecone_ok else "disconnected",
            "index": index_name,
            "rag_chain": "ready" if rag_chain else "not_ready"
        })
    except:
        return jsonify({"status": "unhealthy"}), 500

if __name__ == "__main__":
    print(" Starting Multi-Modal RAG System")
    print("Endpoint: http://localhost:8080")
    print("Health check: http://localhost:8080/health")
    app.run(host="0.0.0.0", port=8080, debug=True)