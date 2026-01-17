
from dotenv import load_dotenv
import os
from src.helper import SimplePDFProcessor, download_hugging_face_embeddings
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

print("🚀 Starting Multi-Modal RAG Setup (Simple Version)...")

# Initialize processor
processor = SimplePDFProcessor()

# Load and process PDFs
print("📄 Loading PDFs from data/ folder...")
documents = processor.load_pdf_file('data/')

if len(documents) == 0:
    print("⚠️  No documents found! Creating sample content...")
    os.makedirs('data', exist_ok=True)
    
    # Create a sample document
    sample_content = """IMF Article IV Consultation - Qatar

Executive Summary:
Qatar's economy shows strong growth with GDP increasing by 3.2% in 2023. 
Inflation remains controlled at 2.1%.

Table 1: Economic Indicators
Year | GDP Growth | Inflation | FDI Inflow
2022 | 3.1% | 6.5% | $5.2B
2023 | 3.2% | 5.8% | $4.8B
2024 | 3.5% | 4.3% | $5.5B

Figure 1: FDI Trends
Foreign Direct Investment shows positive trend with focus on non-hydrocarbon sectors.

Key Recommendations:
1. Enhance business environment
2. Improve access to finance
3. Diversify economy"""
    
    with open('data/sample_qatar.pdf', 'w', encoding='utf-8') as f:
        f.write(sample_content)
    
    documents = processor.load_pdf_file('data/')

print(f"✅ Extracted {len(documents)} pages")

# Split into chunks
print("✂️  Chunking documents...")
chunks = processor.text_split(documents)
print(f"✅ Created {len(chunks)} chunks")

# Show chunk types
element_types = [chunk["metadata"].get("element_type", "text") for chunk in chunks]
print(f"📊 Element distribution: {set(element_types)}")

# Load embeddings
print("🔤 Loading embeddings...")
embeddings = download_hugging_face_embeddings()

# Connect to Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    print("❌ PINECONE_API_KEY not found in .env file")
    exit(1)

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "multimodal-rag-v2"  # New index name

# Create or connect to index
print(f"📊 Setting up Pinecone index '{index_name}'...")
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print(f"✅ Created new index: {index_name}")
else:
    print(f"✅ Using existing index: {index_name}")

# Convert to Document objects
print("📄 Converting to Document format...")
docs_to_upsert = []
for chunk in chunks:
    docs_to_upsert.append(Document(
        page_content=chunk["text"],
        metadata=chunk["metadata"]
    ))

# Store in Pinecone
print(f"💾 Storing {len(docs_to_upsert)} chunks in Pinecone...")
try:
    docsearch = PineconeVectorStore.from_documents(
        documents=docs_to_upsert,
        embedding=embeddings,
        index_name=index_name
    )
    print("🎉 Successfully indexed documents!")
    
    # Verify storage
    index = pc.Index(index_name)
    stats = index.describe_index_stats()
    print(f"📊 Index stats: {stats.total_vector_count} vectors stored")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Trying alternative storage method...")
    
    # Alternative: Manual upsert
    from tqdm import tqdm
    
    index = pc.Index(index_name)
    
    # Create embeddings
    print("Creating embeddings...")
    texts = [doc.page_content for doc in docs_to_upsert]
    embeddings_list = embeddings.embed_documents(texts)
    
    # Prepare vectors
    vectors = []
    for i, (doc, embedding) in enumerate(zip(docs_to_upsert, embeddings_list)):
        vectors.append({
            "id": doc.metadata.get("chunk_id", f"chunk_{i}"),
            "values": embedding,
            "metadata": doc.metadata
        })
    
    # Upsert in batches
    batch_size = 100
    for i in tqdm(range(0, len(vectors), batch_size), desc="Uploading"):
        batch = vectors[i:i+batch_size]
        index.upsert(vectors=batch)
    
    print(f"✅ Uploaded {len(vectors)} vectors")

print("\n" + "="*50)
print("✅ Setup Complete!")
print(f"📚 Documents: {len(documents)} pages")
print(f"🔗 Chunks: {len(chunks)}")
print(f"🗂️  Index: {index_name}")
print("="*50)
print("\n🎯 Next: Update app.py to use index 'multimodal-rag-v2'")
print("🚀 Run: python app.py")