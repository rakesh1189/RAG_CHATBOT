import fitz  # PyMuPDF
import re
from typing import List, Tuple

def extract_text_per_page(file_bytes: bytes) -> Tuple[List[str], int]:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = []
    for i in range(len(doc)):
        text = doc[i].get_text("text") or ""
        text = normalize_text(text)
        pages.append(text)
    return pages, len(doc)

def normalize_text(text: str) -> str:
    text = text.replace('\u00ad', '')  # soft hyphen
    text = re.sub(r'[\t\r\f]+', ' ', text)
    text = re.sub(r'\s+\n', '\n', text)
    text = re.sub(r'\n{2,}', '\n\n', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_pages(pages: List[str], chunk_size: int = 1500, overlap: int = 200):
    chunks = []
    meta = []
    buf = ""
    start_page = 1
    for idx, page in enumerate(pages, start=1):
        buf += f"\n\n[Page {idx}]\n" + page
        while len(buf) >= chunk_size:
            chunk = buf[:chunk_size]
            # find a nice split near the end (end of sentence / newline) to avoid cutting words
            cut = chunk.rfind('. ')
            if cut < chunk_size * 0.6:
                cut = chunk.rfind('\n')
            if cut < chunk_size * 0.4:
                cut = chunk.rfind(' ')
            if cut <= 0:
                cut = chunk_size
            piece = chunk[:cut].strip()
            chunks.append(piece)
            meta.append((start_page, idx))
            # keep overlap
            buf = buf[cut - overlap if cut - overlap > 0 else 0:]
            start_page = start_page  # start_page remains the first page covered by this rolling buffer
    # tail
    if buf.strip():
        chunks.append(buf.strip())
        meta.append((start_page, len(pages)))
    return chunks, meta
