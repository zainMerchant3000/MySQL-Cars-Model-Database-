"""
Basic RAG demo using:
  - sentence-transformers  (free local embeddings)
  - ChromaDB               (free local vector store)
  - Ollama                 (free local LLM — requires `ollama serve` running)

Install deps:
    pip install -r requirements_rag.txt

Run Ollama (one-time setup):
    ollama pull llama3          # download ~4GB model
    ollama serve                # start the server (separate terminal)

Then run:
    python rag_demo.py
"""

import chromadb
from sentence_transformers import SentenceTransformer
import ollama

# ── CONFIG ────────────────────────────────────────────────────────────────────

EMBED_MODEL = "all-MiniLM-L6-v2"   # small, fast, free
LLM_MODEL   = "llama3"              # any model you've pulled via ollama
COLLECTION  = "demo_docs"
TOP_K       = 3

# ── DOCUMENTS ─────────────────────────────────────────────────────────────────

DOCUMENTS = [
    "The Eiffel Tower is located in Paris, France. It was built in 1889 by Gustave Eiffel.",
    "Python is a high-level programming language created by Guido van Rossum in 1991.",
    "The mitochondria is the powerhouse of the cell and produces ATP through cellular respiration.",
    "Mount Everest is the tallest mountain on Earth, standing at 8,849 meters above sea level.",
    "The Great Wall of China stretches over 13,000 miles and was built over many centuries.",
    "Albert Einstein developed the theory of relativity, summarized by E=mc².",
    "The Amazon River is the largest river in the world by discharge volume, located in South America.",
    "Photosynthesis is the process by which plants convert sunlight and CO2 into glucose and oxygen.",
]


def build_index(docs: list[str], embed_model: SentenceTransformer) -> chromadb.Collection:
    client = chromadb.Client()

    # delete collection if it already exists (so re-runs are idempotent)
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass

    collection = client.create_collection(COLLECTION)

    print(f"Embedding {len(docs)} documents...")
    vectors = embed_model.encode(docs).tolist()

    collection.add(
        ids        = [str(i) for i in range(len(docs))],
        documents  = docs,
        embeddings = vectors,
    )
    print("Index built.\n")
    return collection


def retrieve(question: str, collection: chromadb.Collection,
             embed_model: SentenceTransformer) -> list[str]:
    q_vector = embed_model.encode([question]).tolist()
    results  = collection.query(query_embeddings=q_vector, n_results=TOP_K)
    return results["documents"][0]   # list of top-K chunk strings


def generate(question: str, context_chunks: list[str]) -> str:
    context = "\n".join(f"- {c}" for c in context_chunks)
    prompt  = (
        "Use only the context below to answer the question. "
        "If the answer is not in the context, say 'I don't know'.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )
    response = ollama.chat(
        model    = LLM_MODEL,
        messages = [{"role": "user", "content": prompt}],
    )
    return response["message"]["content"].strip()


def ask(question: str, collection: chromadb.Collection,
        embed_model: SentenceTransformer) -> None:
    print(f"Q: {question}")
    chunks = retrieve(question, collection, embed_model)
    print(f"   Retrieved chunks: {chunks}")
    answer = generate(question, chunks)
    print(f"A: {answer}\n")


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    embed_model = SentenceTransformer(EMBED_MODEL)
    collection  = build_index(DOCUMENTS, embed_model)

    ask("Where is the Eiffel Tower?",         collection, embed_model)
    ask("Who created Python?",                collection, embed_model)
    ask("What does the mitochondria do?",     collection, embed_model)
    ask("How tall is Mount Everest?",         collection, embed_model)
    ask("What is the capital of Australia?",  collection, embed_model)  # not in docs → "I don't know"
