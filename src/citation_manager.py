from typing import List, Dict, Any

class CitationManager:
    @staticmethod
    def get_unique_sources(retrieved_docs: List) -> List[str]:
        """Generate clean, unique, and formatted citations"""
        unique_citations = set()
        
        for doc in retrieved_docs:
            # Handle both dictionary and Document object metadata
            meta = getattr(doc, 'metadata', doc.get('metadata', {}) if isinstance(doc, dict) else {})
            
            if not meta:
                continue
            
            # Clean page numbers (removes .0)
            try:
                page_val = int(float(meta.get("page", 0)))
            except:
                page_val = "N/A"
            
            e_type = meta.get("element_type", "text")
            
            if e_type == "table":
                citation = f"📊 Table on Page {page_val}"
            elif e_type == "image":
                citation = f"🖼️ Image on Page {page_val}"
            else:
                citation = f"📄 Page {page_val}"
            
            unique_citations.add(citation)
            
        # Return sorted list for consistent UI display
        return sorted(list(unique_citations))