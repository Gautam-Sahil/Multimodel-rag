
from typing import List, Dict, Any

class CitationManager:
    def __init__(self):
        self.citation_format = "Page {page}, {element_type}"
    
    def generate_citation(self, metadata: Dict[str, Any]) -> str:
        """Generate unique citation from metadata"""
        if not metadata:
            return "📄 Unknown source"
        
        source = metadata.get("source", "Document")
        page = metadata.get("page", "N/A")
        element_type = metadata.get("element_type", "text")
        chunk_id = metadata.get("chunk_id", "")
        
        # Create unique citation key
        citation_key = f"{source}|{page}|{element_type}|{chunk_id}"
        
        # Format readable citation
        if element_type == "table":
            return f"📊 Table on Page {page}"
        elif element_type == "image":
            return f"🖼️ Image on Page {page}"
        else:
            return f"📄 Page {page}"
    
    def format_answer_with_citations(self, answer: str, retrieved_chunks: List) -> str:
        """Add deduplicated citations to the answer"""
        if not retrieved_chunks:
            return answer
        
        # Deduplicate citations
        unique_citations = {}
        citations_by_type = {"text": [], "table": [], "image": []}
        
        for chunk in retrieved_chunks:
            if hasattr(chunk, 'metadata'):
                metadata = chunk.metadata
            elif isinstance(chunk, dict) and 'metadata' in chunk:
                metadata = chunk['metadata']
            else:
                continue
            
            source = metadata.get("source", "")
            page = metadata.get("page", "")
            element_type = metadata.get("element_type", "text")
            
            # Create unique key for deduplication
            citation_key = f"{source}|{page}|{element_type}"
            
            if citation_key not in unique_citations:
                # Format citation
                if element_type == "table":
                    citation_text = f"📊 Table on Page {page}"
                elif element_type == "image":
                    citation_text = f"🖼️ Image on Page {page}"
                else:
                    citation_text = f"📄 Page {page}"
                
                unique_citations[citation_key] = citation_text
                citations_by_type[element_type].append(citation_text)
        
        # Only add citations if we have unique ones
        if not unique_citations:
            return answer
        
        # Format citations section
        citations_text = "\n\n**Sources:**\n"
        
        # Add tables first (if any)
        if citations_by_type["table"]:
            citations_text += "\n**Tables:**\n"
            for citation in sorted(set(citations_by_type["table"])):
                citations_text += f"- {citation}\n"
        
        # Add images (if any)
        if citations_by_type["image"]:
            citations_text += "\n**Images:**\n"
            for citation in sorted(set(citations_by_type["image"])):
                citations_text += f"- {citation}\n"
        
        # Add text citations
        if citations_by_type["text"]:
            citations_text += "\n**Text References:**\n"
            for citation in sorted(set(citations_by_type["text"])):
                citations_text += f"- {citation}\n"
        
        return answer + citations_text
    
    def get_unique_sources(self, retrieved_chunks: List) -> List[str]:
        """Get unique sources for JSON response"""
        unique_sources = set()
        
        for chunk in retrieved_chunks:
            if hasattr(chunk, 'metadata'):
                metadata = chunk.metadata
            elif isinstance(chunk, dict) and 'metadata' in chunk:
                metadata = chunk['metadata']
            else:
                continue
            
            source = metadata.get("source", "")
            page = metadata.get("page", "")
            element_type = metadata.get("element_type", "text")
            
            # Format based on element type
            if element_type == "table":
                unique_sources.add(f"📊 Table on Page {page} of '{source}'")
            elif element_type == "image":
                unique_sources.add(f"🖼️ Image on Page {page} of '{source}'")
            else:
                unique_sources.add(f"📄 Page {page} of '{source}'")
        
        return list(unique_sources)