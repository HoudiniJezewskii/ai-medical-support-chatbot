"""Retrieval: download trusted sources, split them into chunks, and find
the chunks most relevant to a question.
"""

import json
import os
import re

import numpy as np
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

DEFAULT_SOURCES_PATH = "data/sources.json"
DEFAULT_CACHE_PATH = "data/source_cache.json"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ----------------------------------------------------------------------
# Loading sources
# ----------------------------------------------------------------------

def load_source_list(path=DEFAULT_SOURCES_PATH):
    """Read the list of source pages from a JSON file.

    Sources with "enabled": false are stripped out of the returned list.
    If a source has no "enabled" field, it is treated as enabled.
    """
    with open(path, "r", encoding="utf-8") as file:
        sources = json.load(file)
        
    return [source for source in sources if source.get("enabled", True)]


def clean_html_text(html):
    """Extract readable text from an HTML page."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove elements that do not contain useful article content.
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    main_content = soup.find("main") or soup.find("article") or soup.body

    if main_content is None:
        return ""

    text = main_content.get_text(" ", strip=True)
    return re.sub(r"\s+", " ", text)


def fetch_page(source):
    """Download and clean one source page.
    'source' is a dict with title/url/tier."""
    response = requests.get(
        source["url"],
        timeout=30,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "text/html,application/xhtml+xml",
        },
    )
    response.raise_for_status()

    text = clean_html_text(response.text)

    if len(text) < 100:
        raise ValueError("The page returned too little readable text.")

    return {
        "title": source["title"],
        "url": source["url"],
        "tier": source.get("tier"),
        "text": text,
    }


def load_documents(sources, cache_path=DEFAULT_CACHE_PATH, refresh=False):
    """Get the cleaned text of every source.

    Prefer using a cache, set refresh=True to download everything again.
    """
    if cache_path and os.path.exists(cache_path) and not refresh:
        with open(cache_path, "r", encoding="utf-8") as file:
            documents = json.load(file)
        print(f"Loaded {len(documents)} sources from cache ({cache_path}).")
        return documents

    documents = []
    failed = []

    for source in sources:
        try:
            documents.append(fetch_page(source))
            print(f"Loaded: {source['title']}")
        except Exception as error:
            failed.append((source["title"], str(error)))
            print(f"Could not load {source['title']}: {error}")

    print(f"\nLoaded {len(documents)} sources. Failed: {len(failed)}")

    if cache_path and documents:
        with open(cache_path, "w", encoding="utf-8") as file:
            json.dump(documents, file, ensure_ascii=False, indent=2)
        print(f"Saved cache to {cache_path}.")

    return documents


# ----------------------------------------------------------------------
# Chunking
# ----------------------------------------------------------------------

def create_chunks(text, chunk_size=300, overlap=50):
    """Split text into overlapping pieces of about 'chunk_size' words."""
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_text = " ".join(words[start:end])

        if chunk_text.strip():
            chunks.append(chunk_text)

        start += chunk_size - overlap

    return chunks


def build_chunks(documents, chunk_size=300, overlap=50):
    """Chunk every document and keep track of where each chunk came from."""
    chunks = []

    for document in documents:
        pieces = create_chunks(document["text"], chunk_size, overlap)

        for chunk_number, chunk_text in enumerate(pieces):
            chunks.append({
                "text": chunk_text,
                "title": document["title"],
                "url": document["url"],
                "tier": document.get("tier"),
                "chunk_number": chunk_number,
            })

    print(f"Created {len(chunks)} chunks.")
    return chunks


# ----------------------------------------------------------------------
# Retrieval
# ----------------------------------------------------------------------

class Retriever:
    """Finds the chunks most similar in meaning to a question."""

    def __init__(self, chunks, embedding_model_name=DEFAULT_EMBEDDING_MODEL):
        self.chunks = chunks
        self.embedding_model = SentenceTransformer(embedding_model_name)

        self.chunk_embeddings = self.embedding_model.encode(
            [chunk["text"] for chunk in chunks],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        print(f"Embedding shape: {self.chunk_embeddings.shape}")

    def retrieve(self, question, number_of_results=3, min_score=0.0):
        """Return the best chunks for a question, each with a similarity score.

        Chunks scoring below 'min_score' are dropped, so the result can be
        empty. The default of 0.0 keeps everything.
        """
        question_embedding = self.embedding_model.encode(
            question,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        scores = self.chunk_embeddings @ question_embedding
        best_indices = np.argsort(scores)[::-1][:number_of_results]

        results = []

        for index in best_indices:
            if scores[index] < min_score:
                continue

            result = self.chunks[index].copy()
            result["score"] = float(scores[index])
            results.append(result)

        return results


def build_retriever(
    sources_path=DEFAULT_SOURCES_PATH,
    cache_path=DEFAULT_CACHE_PATH,
    refresh=False,
):
    """One-call setup: load sources, chunk them, and build a Retriever."""
    sources = load_source_list(sources_path)
    documents = load_documents(sources, cache_path=cache_path, refresh=refresh)
    chunks = build_chunks(documents)
    return Retriever(chunks)