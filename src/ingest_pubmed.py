"""
Step 1: Ingest GLP-1 related abstracts from PubMed.

Pattern: esearch (query -> list of PMIDs) then efetch (PMIDs -> full records).
This mirrors how PubMed's E-utilities API is actually structured -- search and
fetch are separate endpoints with separate cost profiles, so we keep them as
separate functions rather than one combined call.

Raw API responses are cached to data/raw/ as XML so downstream steps (chunking,
embedding) never have to re-hit the network. Re-running this script is safe --
it's idempotent: if the cache file already exists, we skip the fetch.
"""

import os
import time
from pathlib import Path
import hashlib

import requests
from dotenv import load_dotenv

load_dotenv()

NCBI_API_KEY = os.getenv("NCBI_API_KEY")
NCBI_EMAIL = os.getenv("NCBI_EMAIL")

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"

# With an API key: 10 req/sec allowed -> we sleep 0.11s between calls to stay safely under.
# Without a key: 3 req/sec allowed -> sleep 0.34s.
SLEEP_SECONDS = 0.11 if NCBI_API_KEY else 0.34

# Search query: PubMed's own query syntax, not free text.
# [tiab] = restrict to title/abstract fields, narrowing results to what's actually
# about the drugs (vs. passing mentions). OR groups the known GLP-1 drug names.
SEARCH_QUERY = (
    '(semaglutide[tiab] OR tirzepatide[tiab] OR liraglutide[tiab] '
    'OR "GLP-1 receptor agonist"[tiab]) AND (weight loss[tiab] OR obesity[tiab])'
)

MAX_RESULTS = 250

def query_hash(query: str) -> str:
    """Short, filesystem-safe fingerprint of the search query.
 
    Used to namespace the cache directory -- if SEARCH_QUERY changes, this
    hash changes too, so we automatically get a fresh cache folder instead
    of silently reading stale batches cached under the old query.
    """
    return hashlib.sha256(query.encode("utf-8")).hexdigest()[:10]


def esearch(query: str, retmax: int) -> list[str]:
    """Step 1: search PubMed, get back a list of PMIDs (PubMed IDs)."""
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": retmax,
        "retmode": "json",
        "email": NCBI_EMAIL,
    }
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY

    resp = requests.get(f"{BASE_URL}/esearch.fcgi", params=params)
    resp.raise_for_status()
    data = resp.json()
    pmids = data["esearchresult"]["idlist"]
    print(f"esearch found {len(pmids)} PMIDs for query")
    return pmids


def efetch_batch(pmids: list[str], batch_index: int, cache_dir: Path) -> str:
    """Step 2: fetch full records (title, abstract, journal, etc.) for a batch of PMIDs.

    Returns raw XML text. We cache this to disk before any parsing happens --
    parsing logic can have bugs, but the cached raw response means we never
    have to re-hit the API just because we want to fix a parsing bug.

    cache_dir is namespaced by query_hash() so a changed SEARCH_QUERY never
    reads stale batches left over from a previous query.
    """
    cache_path = cache_dir / f"batch_{batch_index:03d}.xml"
    if cache_path.exists():
        print(f"  batch {batch_index}: cache hit, skipping fetch")
        return cache_path.read_text()

    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "abstract",
        "retmode": "xml",
        "email": NCBI_EMAIL,
    }
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY

    resp = requests.get(f"{BASE_URL}/efetch.fcgi", params=params)
    resp.raise_for_status()
    cache_path.write_text(resp.text)
    print(f"  batch {batch_index}: fetched {len(pmids)} records, cached to {cache_path.name}")
    return resp.text


def chunked(lst: list, size: int):
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def main():
    cache_dir = RAW_DIR / query_hash(SEARCH_QUERY)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not NCBI_EMAIL:
        print("WARNING: NCBI_EMAIL not set in .env -- NCBI asks for this as a courtesy "
              "so they can contact you if your usage causes problems. Not strictly "
              "required to function, but set it before a real run.")

    pmids = esearch(SEARCH_QUERY, MAX_RESULTS)

    # efetch in batches of 50 -- NCBI's documented best practice for the XML endpoint,
    # rather than one massive request or one request per record.
    batch_size = 50
    for i, batch in enumerate(chunked(pmids, batch_size)):
        efetch_batch(batch, i, cache_dir)
        time.sleep(SLEEP_SECONDS)

    print(f"\nDone. {len(pmids)} PMIDs across {(len(pmids) + batch_size - 1) // batch_size} batches.")
    print(f"Raw XML cached in {cache_dir}")


if __name__ == "__main__":
    main()