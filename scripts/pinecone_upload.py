# pinecone_upload.py
import json
import time
import os
import sys
from tqdm import tqdm
from dotenv import load_dotenv

# Ensure the src/ folder is in the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from pinecone import Pinecone, ServerlessSpec
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from packages.core_logic.model_factory import get_embeddings_model

load_dotenv()

# -----------------------------
# Config
# -----------------------------
PROJECT_ROOT = os.getenv("PROJECT_ROOT", ".")
DATA_FILE = os.path.join(PROJECT_ROOT, "data", "vietnam_travel_dataset.json")
BATCH_SIZE = 32

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

if not INDEX_NAME:
    raise ValueError("PINECONE_INDEX_NAME environment variable is not set")

# -----------------------------
# Initialize client & index
# -----------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create managed index if it doesn't exist
existing_indexes = [index.name for index in pc.list_indexes()]
if INDEX_NAME not in existing_indexes:
    from packages.core_logic.config import VECTOR_DIM
    print(f"Creating managed index: {INDEX_NAME} with dimension {VECTOR_DIM}")
    pc.create_index(
        name=INDEX_NAME,
        dimension=VECTOR_DIM,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",          
            region="us-east-1"    
        )
    )
else:
    print(f"Index {INDEX_NAME} already exists.")

# Connect to the index
index = pc.Index(INDEX_NAME)

# -----------------------------
# Helper functions
# -----------------------------
def chunked(iterable, n):
    """Yield successive n-sized chunks from iterable."""
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]

# -----------------------------
# Main upload
# -----------------------------
def main():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        nodes = json.load(f)

    # 1. Convert to LangChain Document objects
    initial_docs = []
    for node in nodes:
        semantic_text = node.get("semantic_text") or (node.get("description") or "")[:1000]
        if not semantic_text.strip():
            continue
        meta = {
            "id": node.get("id"),
            "type": node.get("type"),
            "name": node.get("name"),
            "city": node.get("city", node.get("region", "")),
            "tags": node.get("tags", [])
        }
        initial_docs.append(Document(page_content=semantic_text, metadata=meta))

    # 2. Use LangChain text splitter to chunk documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    split_docs = text_splitter.split_documents(initial_docs)
    print(f"Split {len(initial_docs)} entities into {len(split_docs)} semantic chunks.")

    # 3. Generate unique IDs for each split chunk to avoid collisions
    id_counters = {}
    doc_ids = []
    for doc in split_docs:
        parent_id = doc.metadata["id"]
        id_counters[parent_id] = id_counters.get(parent_id, 0) + 1
        doc_ids.append(f"{parent_id}-chunk-{id_counters[parent_id]}")

    # 4. Initialize LangChain PineconeVectorStore adapter
    embeddings_model = get_embeddings_model()
    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embeddings_model
    )

    print(f"Preparing to upload {len(split_docs)} chunks to Pinecone...")

    # Upload in batches to avoid rate limits or timeouts
    zipped = list(zip(split_docs, doc_ids))
    for batch in tqdm(list(chunked(zipped, BATCH_SIZE)), desc="Uploading batches"):
        batch_docs = [item[0] for item in batch]
        batch_ids = [item[1] for item in batch]
        
        vectorstore.add_documents(documents=batch_docs, ids=batch_ids)
        time.sleep(0.2)

    print("All items uploaded successfully.")

if __name__ == "__main__":
    main()
