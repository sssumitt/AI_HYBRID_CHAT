# 🧠 Hybrid AI Travel Assistant Challenge

### Goal
Build and debug a hybrid AI assistant that answers travel queries using:
- Pinecone (semantic vector DB)
- Neo4j (graph context)
- OpenAI Chat Models

### Steps
1.Set your API keys in `config.py`
2.create virtual environment & install all dependencies
3.Run ‘load_to_neo4j.py’
4.Run ‘visualize_graph.py’
5.Run `python pinecone_upload.py`
6.Run `python hybrid_chat.py`
7.Ask: `create a romantic 4 day itinerary for Vietnam`
8.Modify “hybrid_chat.py” to improve the outcome.

### Deliverables
ask 1: Setup & Data Upload
Run pinecone_upload.py to create the Pinecone index and upload embeddings.
Fix any missing dependencies or environment issues.
Confirm embeddings successfully appear in your Pinecone dashboard.
✅ Deliverable:
 Screenshot of successful upsert batches and index details.