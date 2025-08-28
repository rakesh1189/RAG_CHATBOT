import os
import math
from typing import List, Tuple, Dict
import numpy as np
from openai import OpenAI

from pdf_utils import extract_text_per_page, chunk_pages
from vecstore import VectorStore
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", None)
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")
VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR", "./storage/chroma")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("TOP_K", "8"))
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", None)

# ⬇️ pass api_key explicitly so env loading order doesn’t matter
if OPENAI_BASE_URL:
    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
else:
    client = OpenAI(api_key=OPENAI_API_KEY)
store = VectorStore(VECTOR_STORE_DIR)

def embed_texts(texts: List[str]) -> List[List[float]]:
    # Batch to avoid payload too large
    batch = 96
    out = []
    for i in range(0, len(texts), batch):
        resp = client.embeddings.create(model=EMBED_MODEL, input=texts[i:i+batch])
        out.extend([d.embedding for d in resp.data])
    return out

def upload_pdf(file_bytes: bytes) -> Tuple[str, int]:
    pages, total = extract_text_per_page(file_bytes)
    chunks, meta = chunk_pages(pages, CHUNK_SIZE, CHUNK_OVERLAP)

    embeddings = embed_texts(chunks)
    doc_id = __import__("uuid").uuid4().hex
    ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
    metadatas = [ {"page_start": ps, "page_end": pe, "chunk_index": i} for i, (ps, pe) in enumerate(meta) ]

    store.add(doc_id, ids=ids, embeddings=embeddings, metadatas=metadatas, documents=chunks)
    return doc_id, total

def retrieve(doc_id: str, question: str, top_k: int = TOP_K):
    q_emb = embed_texts([question])[0]
    res = store.query(doc_id, q_emb, top_k=top_k)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    # Convert cosine distance to similarity for readability
    sims = [1 - d for d in dists]
    return docs, metas, sims

def build_prompt(question: str, contexts: List[str], metas: List[dict]) -> List[Dict]:
    context_blocks = []
    for i, (ctx, m) in enumerate(zip(contexts, metas), start=1):
        context_blocks.append(f"[Source {i} | pages {m['page_start']}-{m['page_end']}]\n{ctx}")
    context_text = "\n\n".join(context_blocks)

    system = (
        "You are a meticulous contract and document analysis assistant. "
        "Answer the user's question **strictly** using the provided context snippets. "
        "If the answer is not present, say you couldn't find it in the document. "
        "Always cite sources as [Source N] with page ranges."
    )
    user = (
        f"Question: {question}\n\n"
        f"Context:\n{context_text}\n\n"
        "Instructions:\n"
        "- Provide a concise, accurate answer first.\n"
        "- Then list the sources you used by their [Source N] labels with page ranges.\n"
        "- If uncertain, ask a brief clarifying follow-up."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

def answer_question(doc_id: str, question: str, history: List[dict]) -> Dict:
    contexts, metas, sims = retrieve(doc_id, question)
    # Limit total context length to avoid over-long prompts
    max_chars = 12000
    selected = []
    running = 0
    for c, m in zip(contexts, metas):
        if running + len(c) > max_chars:
            break
        selected.append((c, m))
        running += len(c)

    msgs = build_prompt(question, [c for c, _ in selected], [m for _, m in selected])
    # Optionally include brief history
    for turn in history[-4:]:
        if "role" in turn and "content" in turn:
            msgs.insert(1, {"role": turn["role"], "content": turn["content"]})

    resp = client.chat.completions.create(model=CHAT_MODEL, messages=msgs, temperature=0.2)
    text = resp.choices[0].message.content

    sources = []
    for (c, m), sim in zip(selected, sims[:len(selected)]):
        preview = c[:400].replace("\n", " ")
        sources.append({
            "page_start": m["page_start"],
            "page_end": m["page_end"],
            "score": round(float(sim), 4),
            "preview": preview + ("..." if len(c) > 400 else "")
        })
    return {"answer": text, "sources": sources}
