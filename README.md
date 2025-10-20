# AI Hybrid Travel Assistant

This project is a **full-stack, AI-powered travel assistant** designed to provide **personalized travel itineraries**.  
It leverages a **hybrid Retrieval-Augmented Generation (RAG)** architecture, combining **semantic vector search** with a **structured knowledge graph** to generate creative, factually grounded, and context-aware responses.

---

## Core Architecture

The system's intelligence is derived from a **multi-stage RAG pipeline** that bridges the gap between semantic understanding and factual knowledge.

### Semantic Search (Pinecone)
- The user's query is first converted into an embedding and sent to **Pinecone**.  
- Pinecone performs a *fuzzy semantic search* to identify a list of the most conceptually relevant entities (e.g., locations, activities) from a large knowledge base.

### Targeted Graph Retrieval (Neo4j)
- The list of relevant entity IDs from Pinecone is then used as the input for a targeted **Cypher query** against a **Neo4j knowledge graph**.  
- This step retrieves the explicit, factual relationships between the identified entities, grounding the creative suggestions in real-world context.

### Summarization & Generation (LLM)
- Both the semantic concepts from Pinecone and the factual relationships from Neo4j are synthesized by an **LLM** into a concise summary.  
- This clean, dense context is then passed to a final LLM call, which generates the coherent, user-facing travel itinerary.

This hybrid approach ensures that the final output is both **imaginative** and **logically sound**, providing a superior user experience compared to a single-database RAG system.

---

## Project Structure

The project is organized using a modern `src` layout to clearly separate the installable application code from project management and data files.

```

AI_HYBRID_CHAT/
├── data/
│   └── vietnam_travel_dataset.json
├── scripts/
│   ├── load_to_neo4j.py
│   └── pinecone_upload.py
├── src/
│   ├── api/
│   │   └── app/
│   ├── cli/
│   │   └── main.py
│   └── packages/
│       └── core_logic/
├── .env.example
├── pyproject.toml
└── README.md

````

---

## Technical Stack

- **Backend:** Python 3.12+  
- **API Framework:** FastAPI  
- **Vector Database:** Pinecone  
- **Graph Database:** Neo4j  
- **Caching:** Upstash Redis  
- **LLM Provider:** OpenAI  
- **Frontend:** React (Next.js), Tailwind CSS  
- **Package Management:** uv  
- **Web Server:** Uvicorn  

---

## Setup and Installation

Follow these steps to set up the development environment.

### 1. Prerequisites
- Python 3.12.3 or higher  
- `uv` (recommended) or `pip`

### 2. Clone the Repository
```bash
git clone <your-repository-url>

cd AI_HYBRID_CHAT

```


### 3. Configure Environment Variables

Create a `.env` file from the provided template:

```bash
cp .env.example .env
```

Open the newly created `.env` file and populate it with your API keys and service credentials (**OpenAI**, **Pinecone**, **Neo4j**, **Upstash Redis**).
You must also set:

```
PROJECT_ROOT=<absolute-path-to-your-project-directory>
```

### 4. Create and Activate a Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies.

```bash
# Create the virtual environment using uv
uv venv

# Activate the environment
# On macOS / Linux:
source .venv/bin/activate

# On Windows (Command Prompt):
.venv\Scripts\activate
```

### 5. Install Dependencies

Install the project and all its dependencies in "editable" mode.
This allows you to make changes to the source code without needing to reinstall.

```bash
uv pip install -e ".[scripts,dev]"
```

---

## Running the Application

### Data Ingestion

Before running the application, you must populate the databases using the provided scripts.
Run these commands from the project root:

```bash
# Load data into Neo4j
python scripts/load_to_neo4j.py

# Load data into Pinecone
python scripts/pinecone_upload.py
```

### Run the Command-Line Interface (CLI)

For quick testing and development, you can use the CLI.

```bash
python src/cli/main.py
```

### Run the API Server

To serve the React frontend, run the FastAPI application using Uvicorn.

```bash
uvicorn src.api.app.main:app --reload
```

The API will be available at:
* [http://localhost:8000](http://localhost:8000)

```
