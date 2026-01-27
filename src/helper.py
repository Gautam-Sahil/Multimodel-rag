import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import re

class SimplePDFProcessor:
    def __init__(self):
        pass
        
    def load_pdf_file(self, data_path):
        """PDF extraction with absolute page numbering fix"""
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
                        
                        # Use enumerate(..., 1) to ensure Page 1 is physically the first page
                        for page_num, page in enumerate(reader.pages, 1):
                            page_text = page.extract_text() or ""
                            
                            if not page_text.strip():
                                continue
                            
                            # Heuristic for multi-modal elements
                            lines = page_text.split('\n')
                            table_content = ""
                            text_content = ""
                            
                            for line in lines:
                                if (line.count('|') >= 2 or line.count('\t') >= 2 or 
                                    re.search(r'\d+\.?\d*\s*[%$€£]', line)):
                                    table_content += line + "\n"
                                else:
                                    text_content += line + "\n"
                            
                            metadata = {
                                "source": pdf_file,
                                "page": int(page_num),  # Explicitly store as integer
                                "element_type": "text", # Default, will be refined in split
                                "has_tables": bool(table_content.strip()),
                                "has_images": bool(re.search(r'Figure|Chart|Graph', page_text, re.I))
                            }
                            
                            content = text_content
                            if table_content.strip():
                                content += "\n[TABLE_START]\n" + table_content + "\n[TABLE_END]\n"
                            
                            all_docs.append({
                                "content": content,
                                "metadata": metadata
                            })
                except Exception as e:
                    print(f"❌ Error processing {pdf_file}: {e}")
        
        return all_docs

    def text_split(self, documents, chunk_size=800, chunk_overlap=150):
        """Split documents while preserving page-specific metadata"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = []
        for doc in documents:
            split_texts = text_splitter.split_text(doc["content"])
            
            for i, text_chunk in enumerate(split_texts):
                chunk_metadata = doc["metadata"].copy()
                
                # Determine element type for this specific chunk
                if "[TABLE_START]" in text_chunk or re.search(r'Table \d+', text_chunk):
                    chunk_metadata["element_type"] = "table"
                elif re.search(r'Figure \d+|Chart|Graph', text_chunk, re.I):
                    chunk_metadata["element_type"] = "image"
                else:
                    chunk_metadata["element_type"] = "text"
                
                chunks.append({
                    "text": text_chunk,
                    "metadata": chunk_metadata
                })
        return chunks

def download_hugging_face_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
