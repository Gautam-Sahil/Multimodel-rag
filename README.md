# Multimodel-rag: Multi-Modal Document Intelligence RAG

A robust Retrieval-Augmented Generation (RAG) system designed to process complex PDFs containing text, tables, and images.

## 🚀 Features
- **Multi-Modal Ingestion**: Specialized parsing for narrative text, structural tables, and image references.
- **Smart Chunking**: Preserves tabular context using custom `[TABLE_START]` and `[TABLE_END]` markers.
- **Vector Search**: High-performance retrieval using Pinecone Serverless and HuggingFace embeddings.
- **Automated Citations**: Precise source attribution including page numbers and element types.
- **Interactive UI**: Clean Flask-based web interface for real-time document querying.

## 🛠️ Tech Stack
- **Framework**: LangChain v0.3
- **LLM**: GPT-4o-mini (via OpenRouter)
- **Vector DB**: Pinecone
- **Embeddings**: Sentence-Transformers (all-MiniLM-L6-v2)
- **PDF Engine**: pypdf, pdfplumber
- **Backend**: Flask, Python 3.10+

## 📁 Project Structure
```text
.
├── src/
│   ├── helper_simple.py      # Core Multi-modal PDF processing logic
│   ├── citation_manager.py   # Source attribution logic
│   └── prompt.py             # System prompt engineering
├── data/                     # Raw PDF documents
├── static/                   # CSS and frontend assets
├── templates/                # HTML templates for the UI
├── app.py                    # Flask application entry point
├── store_index.py            # Vector database indexing script
├── requirements.txt          # Python dependencies
└── .env                      # Environment variables (API Keys)
⚙️ Installation & Setup
Clone the repository:

Bash

git clone [https://github.com/your-username/Multimodel-rag.git](https://github.com/your-username/Multimodel-rag.git)
cd medexa-ai
Create a virtual environment:

Bash

conda create -n assingment python=3.10 -y
conda activate assingment
Install dependencies:

Bash

pip install -r requirements.txt
Configure Environment Variables: Create a .env file in the root directory:

Code snippet

PINECONE_API_KEY=your_pinecone_key
OPENROUTER_API_KEY=your_openrouter_key
Initialize the Vector Store:

Bash

python store_index.py
Run the Application:

Bash

python app.py
Visit http://localhost:8080 in your browser.

📊 Methodology
The system uses a Heuristic Multi-Modal Parser. By analyzing numerical density and row-patterns, it identifies tables and wraps them in HTML-like tags.
This allows the LLM to maintain spatial awareness of data, which is critical for financial and macroeconomic reports like the IMF's Qatar report.
