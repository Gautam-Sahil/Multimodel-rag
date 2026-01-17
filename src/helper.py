
import os
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
import re

class SimplePDFProcessor:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def load_pdf_file(self, data_path):
        """PDF extraction without external dependencies"""
        all_docs = []
        
        if not os.path.exists(data_path):
            print(f"Warning: {data_path} does not exist")
            return all_docs
            
        for pdf_file in os.listdir(data_path):
            if pdf_file.endswith('.pdf'):
                pdf_path = os.path.join(data_path, pdf_file)
                print(f"📄 Processing: {pdf_file}")
                
                try:
                    with open(pdf_path, 'rb') as f:
                        reader = PdfReader(f)
                        
                        for page_num, page in enumerate(reader.pages, 1):
                            # Extract text from page
                            page_text = page.extract_text() or ""
                            
                            if not page_text.strip():
                                continue
                            
                            # Detect tables (simple heuristic)
                            lines = page_text.split('\n')
                            table_content = ""
                            text_content = ""
                            
                            for line in lines:
                                # Check if line looks like a table row
                                if (line.count('|') >= 2 or 
                                    line.count('\t') >= 2 or 
                                    re.search(r'\d+\.?\d*\s*[%$€£]', line)):
                                    table_content += line + "\n"
                                else:
                                    text_content += line + "\n"
                            
                            # Create document for this page
                            metadata = {
                                "source": pdf_file,
                                "page": page_num,
                                "element_type": "multimodal",
                                "has_tables": bool(table_content.strip()),
                                "has_images": "Figure" in page_text or "Chart" in page_text
                            }
                            
                            # Combine content with markers
                            content = ""
                            if text_content.strip():
                                content += text_content
                            
                            if table_content.strip():
                                content += "\n[TABLE_START]\n" + table_content + "\n[TABLE_END]\n"
                            
                            if metadata["has_images"]:
                                content += "\n[IMAGE_REFERENCE] Contains charts/figures\n"
                            
                            if content.strip():
                                all_docs.append({
                                    "content": content,
                                    "metadata": metadata
                                })
                        
                except Exception as e:
                    print(f"❌ Error processing {pdf_file}: {e}")
        
        print(f"✅ Extracted {len(all_docs)} pages")
        return all_docs
 
def text_split(self, documents, chunk_size=600, chunk_overlap=150):
    """Enhanced chunking that preserves figure/table context"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\nFigure ",  # Split at figures
            "\n\nTable ",   # Split at tables  
            "\n\nAnnex ",   # Split at annexes
            "\n\n",         # Split at paragraphs
            "\n",           # Split at lines
            " ",            # Split at words (last resort)
            ""
        ]
    )
    
    chunks = []
    for doc in documents:
        # Pre-process to ensure figures/tables are preserved
        content = doc["content"]
        
        # Add markers for better splitting
        content = content.replace("Figure ", "\n\nFigure ")
        content = content.replace("Table ", "\n\nTable ")
        content = content.replace("Annex ", "\n\nAnnex ")
        
        split_texts = text_splitter.split_text(content)
        
        for i, text_chunk in enumerate(split_texts):
            chunk_metadata = doc["metadata"].copy()
            chunk_metadata["chunk_id"] = f"{chunk_metadata['source']}_p{chunk_metadata['page']}_c{i}"
            
            # Enhanced element type detection
            if re.search(r'\[TABLE_START\]|Table \d+:|[\d.]+%|\d+\.\d+', text_chunk):
                chunk_metadata["element_type"] = "table"
                chunk_metadata["content_type"] = "numerical_data"
            elif re.search(r'Figure \d+:|Chart|graph|diagram', text_chunk, re.IGNORECASE):
                chunk_metadata["element_type"] = "image"
                chunk_metadata["content_type"] = "visual_analysis"
            elif "Annex" in text_chunk:
                chunk_metadata["element_type"] = "annex"
                chunk_metadata["content_type"] = "technical_appendix"
            else:
                chunk_metadata["element_type"] = "text"
                chunk_metadata["content_type"] = "descriptive"
            
            chunks.append({
                "text": text_chunk,
                "metadata": chunk_metadata
            })
    
    return chunks

def download_hugging_face_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")