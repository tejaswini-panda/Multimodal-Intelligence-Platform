"""
Embedding Layer (PRD Section 5.4 - "Embedding and Multimodal Indexing")

What is an embedding?
A piece of text -> a list of numbers (a vector), e.g. "learning rate" -> [0.12, -0.44, 0.08, ...]
Texts with SIMILAR MEANING end up with vectors that are close together in this
number-space. That's what lets us search by meaning instead of exact keywords.

We use Gemini's embedding model (gemini-embedding-001) via the NEW `google-genai` SDK.
NOTE: the older `google-generativeai` package is deprecated as of 2025 - if any
tutorial/video online shows `import google.generativeai as genai`, that's outdated.
Always use `from google import genai` (this file) going forward.
"""

import os
import time
from google import genai
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
_client = genai.Client(api_key=API_KEY) if API_KEY else None

EMBEDDING_MODEL = "gemini-embedding-001"


def _embed_with_retry(contents, task_type: str, max_retries: int = 4):
    """
    Call Gemini embeddings with retries for temporary errors (429/5xx).
    `contents` can be a single string or a list of strings.
    """
    if not _client:
        raise RuntimeError(
            "GEMINI_API_KEY not set. Copy .env.example to .env and add your key."
        )

    retry_keywords = (
        "429", "500", "502", "503", "504",
        "unavailable", "rate limit", "resource exhausted", "timeout",
    )

    for attempt in range(1, max_retries + 1):
        try:
            return _client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=contents,
                config={"task_type": task_type},
            )
        except Exception as exc:
            retryable = any(k in str(exc).lower() for k in retry_keywords)
            if not retryable or attempt == max_retries:
                raise
            wait = attempt * 3
            print(f"Embedding error ({exc}). Retrying in {wait}s (attempt {attempt}/{max_retries})...")
            time.sleep(wait)


def get_embedding(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
    """
    Convert a piece of text into an embedding vector using Gemini.

    task_type:
      - "RETRIEVAL_DOCUMENT" -> use when embedding content going INTO the database
      - "RETRIEVAL_QUERY"    -> use when embedding the user's search question
    (Gemini optimizes the vector slightly differently depending on which side
    of the search you're on - this small detail improves retrieval quality.)
    """
    result = _embed_with_retry(text, task_type)
    return result.embeddings[0].values


def get_embeddings_batch(
    texts: list[str],
    task_type: str = "RETRIEVAL_DOCUMENT",
    batch_size: int = 50,
) -> list[list[float]]:
    """
    Embed many texts in batches (one API call per `batch_size` texts),
    with retries for temporary errors.
    """
    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        result = _embed_with_retry(batch, task_type)
        embeddings.extend(e.values for e in result.embeddings)
        time.sleep(0.5)  # gentle pacing between batches

    if len(embeddings) != len(texts):
        raise RuntimeError(
            f"Embedding count mismatch: expected {len(texts)}, got {len(embeddings)}"
        )

    return embeddings


if __name__ == "__main__":
    # Quick manual test - run this file directly once you've added your API key to .env
    sample = "The learning rate controls how big each optimization step is."
    vec = get_embedding(sample)
    print(f"Embedding length: {len(vec)}")
    print(f"First 5 values: {vec[:5]}")