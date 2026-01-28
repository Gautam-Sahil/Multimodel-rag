# Multimodel-rag: Multi-Modal Document Intelligence RAG


<img width="1920" height="1080" alt="Screenshot 2026-01-17 230531" src="https://github.com/user-attachments/assets/54dbabd3-7f4b-459e-be92-91b5b9450145" />


<table>
  <tr>
    <td align="center">
      <img width="596" height="600" alt="Screenshot 2026-01-17 230801" src="https://github.com/user-attachments/assets/466d84ec-bcbd-4ab0-bd02-0bcae356b7be" />
    </td>
    <td align="center">
      <img width="594" height="600" alt="Screenshot 2026-01-17 230825" src="https://github.com/user-attachments/assets/9e8df02a-71a8-4540-8af9-77148249c456" />
    </td>
  </tr>
  <tr>
    <td align="center">
      <img width="593" height="600" alt="Screenshot 2026-01-17 230839" src="https://github.com/user-attachments/assets/272f79d8-3a53-414e-815c-2faa389b7f82" />
    </td>
    <td align="center">
      <img width="595" height="600" alt="Screenshot 2026-01-17 230853" src="https://github.com/user-attachments/assets/1c729c94-fd7c-4f7c-be1f-5ff8c0fa3345" />
    </td>
  </tr>
</table>



A robust Retrieval-Augmented Generation (RAG) system designed to process complex PDFs containing text, tables, graphs and images.

## 🚀 Live Demo
[Live Demo 1](https://huggingface.co/spaces/GautamxSahil/multimodal-rag-system)

[Live Demo 2](https://multimodel-rag-orz5.onrender.com/)


## 🚀 Vedio Demo
[Vedio Demo](https://drive.google.com/file/d/1_qYyjUpQB4kaS37vriHyUQVTR4mt8Y1q/view?usp=sharing)


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

Run these commands:

python store_index.py
Run the Application:

Bash

python app.py
Visit http://localhost:8080 in your browser.

📊 Methodology
The system uses a Heuristic Multi-Modal Parser. By analyzing numerical density and row-patterns, it identifies tables and wraps them in HTML-like tags.
This allows the LLM to maintain spatial awareness of data, which is critical for financial and macroeconomic reports like the IMF's Qatar report.
