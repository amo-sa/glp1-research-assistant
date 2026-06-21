"""
Step 2: Parse cached PubMed XML, chunk abstracts by section, embed, and
store in a local Chroma vector database.

Pipeline: parse_articles() -> chunk_article() -> embed + store in Chroma.

Chunking strategy: structure-aware, not fixed-size. PubMed abstracts are
often pre-segmented by the source into labeled sections (INTRODUCTION,
METHODS, RESULTS, CONCLUSION). We chunk along those existing boundaries
instead of splitting every N characters, so each chunk is a semantically
coherent unit rather than an arbitrary slice. Abstracts without labeled
sections fall back to being treated as a single chunk.

Each chunk is embedded independently and stored with metadata (pmid,
section, title, journal) so retrieved chunks can always be traced back
to their source article for citation.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import chromadb

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "glp1_abstracts"


def parse_articles(xml_text: str) -> list[dict]:
    """Parse one batch's raw XML into a list of clean article records.

    Each record: {pmid, title, journal, sections: [{label, text}, ...]}
    Articles with no usable abstract text are skipped (some PubMed records
    are title-only, e.g. letters or corrections).
    """
    root = ET.fromstring(xml_text)
    articles = []

    for article_elem in root.findall(".//PubmedArticle"):
        pmid_elem = article_elem.find(".//PMID")
        title_elem = article_elem.find(".//ArticleTitle")
        journal_elem = article_elem.find(".//Journal/Title")

        if pmid_elem is None or title_elem is None:
            continue  # malformed record, skip rather than crash the whole batch

        pmid = pmid_elem.text
        title = title_elem.text or ""
        journal = journal_elem.text if journal_elem is not None else "Unknown journal"

        sections = []
        for abstract_text_elem in article_elem.findall(".//Abstract/AbstractText"):
            label = abstract_text_elem.get("Label", "ABSTRACT")  # unlabeled -> generic fallback
            text = abstract_text_elem.text
            if text:  # skip empty sections
                sections.append({"label": label, "text": text})

        if not sections:
            continue  # no abstract text at all, nothing to chunk/embed

        articles.append({
            "pmid": pmid,
            "title": title,
            "journal": journal,
            "sections": sections,
        })

    return articles


def chunk_article(article: dict) -> list[dict]:
    """Turn one parsed article into a list of chunk records ready to embed.

    One chunk per section (structure-aware chunking). Each chunk carries
    enough metadata to be cited back to its source article.
    """
    chunks = []
    for section in article["sections"]:
        chunk_id = f"{article['pmid']}_{section['label']}"
        chunks.append({
            "id": chunk_id,
            "text": section["text"],
            "metadata": {
                "pmid": article["pmid"],
                "title": article["title"],
                "journal": article["journal"],
                "section": section["label"],
            },
        })
    return chunks


def load_all_chunks(cache_dir: Path) -> list[dict]:
    """Parse every cached batch XML file in cache_dir and chunk all articles."""
    all_chunks = []
    batch_files = sorted(cache_dir.glob("batch_*.xml"))
    print(f"Found {len(batch_files)} cached batch files in {cache_dir}")

    for batch_file in batch_files:
        xml_text = batch_file.read_text()
        articles = parse_articles(xml_text)
        for article in articles:
            all_chunks.extend(chunk_article(article))

    print(f"Parsed into {len(all_chunks)} chunks total")
    return all_chunks


def embed_and_store(chunks: list[dict]):
    """Embed every chunk and store in a persistent local Chroma collection."""
    # Imported here, not at module level, so parsing/chunking logic (and tests
    # of it) don't require sentence-transformers/torch to even be installed.
    from sentence_transformers import SentenceTransformer

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    # cosine similarity explicitly, not left to whatever Chroma's default is
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    collection.upsert(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=metadatas,
    )
    print(f"Stored {collection.count()} chunks in Chroma collection '{COLLECTION_NAME}'")


def main():
    # find the most recently modified query-hash subfolder under data/raw
    candidate_dirs = [d for d in RAW_DIR.iterdir() if d.is_dir()]
    if not candidate_dirs:
        raise RuntimeError(f"No cached batches found in {RAW_DIR}. Run ingest_pubmed.py first.")
    cache_dir = max(candidate_dirs, key=lambda d: d.stat().st_mtime)

    chunks = load_all_chunks(cache_dir)
    embed_and_store(chunks)


if __name__ == "__main__":
    main()