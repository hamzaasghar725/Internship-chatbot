"""
Backward-compatible facade for the RAG pipeline.

This file used to contain the entire RAG implementation in one place. It has
been split into focused modules for readability:

    rag/tracing.py            Langfuse observability helpers
    rag/embeddings.py         embedding model loading + query embedding
    rag/indexing.py           text extraction, chunking, FAISS index build/search
    rag/gemini_client.py      raw HTTP client for the Gemini API
    rag/answer_formatting.py  markdown polishing, marker parsing, scope enforcement
    rag/qa.py                 answer_question / generate_answer / summarize_document

Everything that used to be importable from `rag.rag_utils` (e.g.
`from rag.rag_utils import build_or_update_index, answer_question,
summarize_document`) is re-exported here unchanged, so no other file in the
project needs to change. Behaviour is 100% identical to before the split --
only the file layout changed.
"""

# ---- Config / constants (unchanged names, unchanged values) ----
from rag.embeddings import EMBED_MODEL_NAME
from rag.gemini_client import (
    GEMINI_MAX_OUTPUT_TOKENS,
    GEMINI_MODEL_CANDIDATES,
    GEMINI_TEMPERATURE,
)
from rag.indexing import CHUNK_OVERLAP, CHUNK_SIZE, VECTORSTORE_DIR

# ---- Tracing helpers ----
from rag.tracing import (
    APP_RELEASE,
    LANGFUSE_ENVIRONMENT,
    TRACE_APP_TAG,
    _attributes,
    _emit_event,
    _flush,
    _get_langfuse_client,
    _observe,
    _tag_trace,
    _trace_warn,
    _update,
)

# ---- Embeddings ----
from rag.embeddings import embed_query, get_embed_model

# ---- Indexing / retrieval ----
from rag.indexing import (
    _paths_for_user,
    _user_has_index,
    build_or_update_index,
    chunk_text,
    extract_text,
    extract_text_ex,
    load_full_document_text,
    retrieve_relevant_chunks,
)

# ---- OCR (scanned PDFs / images) ----
from rag.ocr import IMAGE_EXTENSIONS, OCRError

# ---- Gemini client ----
from rag.gemini_client import (
    GeminiResponseError,
    _call_gemini,
    _post_to_gemini,
    _safe_call_gemini,
)

# ---- Answer formatting / scope enforcement ----
from rag.answer_formatting import (
    _polish_line,
    _split_markers,
    _split_sentences,
    enforce_scope,
    polish_answer,
)

# ---- Top-level RAG pipeline ----
from rag.qa import (
    _summarize_chunks,
    answer_question,
    generate_answer,
    summarize_document,
)

__all__ = [
    "EMBED_MODEL_NAME",
    "GEMINI_MODEL_CANDIDATES",
    "GEMINI_TEMPERATURE",
    "GEMINI_MAX_OUTPUT_TOKENS",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "VECTORSTORE_DIR",
    "APP_RELEASE",
    "LANGFUSE_ENVIRONMENT",
    "TRACE_APP_TAG",
    "get_embed_model",
    "embed_query",
    "extract_text",
    "extract_text_ex",
    "load_full_document_text",
    "OCRError",
    "IMAGE_EXTENSIONS",
    "chunk_text",
    "build_or_update_index",
    "retrieve_relevant_chunks",
    "GeminiResponseError",
    "answer_question",
    "generate_answer",
    "summarize_document",
    "polish_answer",
    "enforce_scope",
]