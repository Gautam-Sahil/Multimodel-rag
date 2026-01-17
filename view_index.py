from pinecone import Pinecone
from dotenv import load_dotenv
import os
import json

load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("multimodal-rag-v2")

print("🔍 Analyzing Pinecone Index...")

# Get ALL vectors with metadata (in batches)
def get_all_vectors(limit=100):
    """Get all vectors from index"""
    all_vectors = []
    query_vectors = [
        [0.1]*384, 
        [0.5]*384,  
        [-0.5]*384,
        [1.0]*384,  
        [-1.0]*384  
    ]
    
    seen_ids = set()
    for query_vec in query_vectors:
        results = index.query(
            vector=query_vec,
            top_k=50,
            include_metadata=True,
            include_values=False
        )
        
        for match in results.matches:
            if match.id not in seen_ids:
                seen_ids.add(match.id)
                all_vectors.append(match)
                
                if len(all_vectors) >= limit:
                    return all_vectors
    
    return all_vectors

# Get sample vectors
vectors = get_all_vectors(50)

if not vectors:
    print("❌ No vectors found in index!")
    exit(1)

print(f"✅ Found {len(vectors)} vectors")

# Analyze element types
print("\n📊 Element Type Distribution:")
element_counts = {}
page_distribution = {}

for vector in vectors:
    metadata = vector.metadata
    elem_type = metadata.get('element_type', 'unknown')
    page = metadata.get('page', 'unknown')
    
    element_counts[elem_type] = element_counts.get(elem_type, 0) + 1
    page_distribution[page] = page_distribution.get(page, 0) + 1

print("Element Types:")
for elem_type, count in element_counts.items():
    print(f"  {elem_type}: {count} ({count/len(vectors)*100:.1f}%)")

print("\n📄 Page Distribution (top 10):")
for page, count in sorted(page_distribution.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  Page {page}: {count} chunks")

# Show samples of each type
print("\n🔍 Sample Content by Type:")

for elem_type in element_counts.keys():
    print(f"\n📌 {elem_type.upper()} Samples:")
    samples = [v for v in vectors if v.metadata.get('element_type') == elem_type][:2]
    
    for i, sample in enumerate(samples):
        print(f"  Sample {i+1}:")
        print(f"    ID: {sample.id}")
        print(f"    Page: {sample.metadata.get('page', 'N/A')}")
        print(f"    Source: {sample.metadata.get('source', 'N/A')}")
        
        # Get content preview
        content = ""
        if 'text' in sample.metadata:
            content = sample.metadata['text']
        elif 'page_content' in sample.metadata:
            content = sample.metadata['page_content']
        
        if content:
            preview = content[:150].replace('\n', ' ')
            print(f"    Preview: {preview}...")
        
        # Check for special markers
        if content and '[TABLE_START]' in content:
            print(f"    ✅ Contains table data")
        if content and '[IMAGE_REFERENCE]' in content:
            print(f"    ✅ Contains image reference")
        if content and 'Figure' in content:
            print(f"    ✅ Mentions figures")
        if content and 'Table' in content:
            print(f"    ✅ Mentions tables")

# Save detailed view to file
print("\n💾 Saving detailed analysis to 'pinecone_analysis.json'...")

analysis = {
    "total_vectors": len(vectors),
    "element_distribution": element_counts,
    "page_distribution": page_distribution,
    "samples": []
}

for i, vector in enumerate(vectors[:20]):
    analysis["samples"].append({
        "id": vector.id,
        "score": float(vector.score),
        "metadata": vector.metadata
    })

with open('pinecone_analysis.json', 'w', encoding='utf-8') as f:
    json.dump(analysis, f, indent=2, ensure_ascii=False)

print("✅ Analysis complete!")
print(f"📁 Check 'pinecone_analysis.json' for full details")