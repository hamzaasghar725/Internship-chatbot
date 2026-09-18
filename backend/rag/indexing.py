"""Document ingestion: text extraction, chunking, and the per-user FAISS index."""
import os
import pickle

import faiss
from PyPDF2 import PdfReader

from rag.embeddings import EMBED_MODEL_NAME, embed_query, get_embed_model
from rag.tracing import (
    TRACE_APP_TAG,
    _attributes,
    _emit_event,
    _flush,
    _get_langfuse_client,
    _observe,
    _tag_trace,
    _update,
)

CHUNK_SIZE = 500        # characters per chunk
CHUNK_OVERLAP = 50
VECTORSTORE_DIR = os.path.join(os.path.dirname(__file__), "..", "vectorstore")


def _user_has_index(user_id):
    """True agar is user ne koi document upload kar rakha hai."""
    index_path, _ = _paths_for_user(user_id)
    return os.path.exists(index_path)


def extract_text(file_path):
    """Extracts text from a PDF or TXT file."""
    if file_path.lower().endswith(".pdf"):
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += (page.extract_text() or "") + "\n"
        return text
    else:  # .txt and other plain text files
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Splits text into overlapping chunks so context isn't lost at boundaries."""
    chunks = []
    start = 0
    text = text.strip()
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]


def _paths_for_user(user_id):
    user_dir = os.path.join(VECTORSTORE_DIR, str(user_id))
    os.makedirs(user_dir, exist_ok=True)
    index_path = os.path.join(user_dir, "index.faiss")
    meta_path = os.path.join(user_dir, "meta.pkl")
    return index_path, meta_path


def build_or_update_index(user_id, file_path, filename, session_id=None):
    """
    Processes a new document and builds a fresh FAISS index for the user,
    replacing any previously uploaded document. Each user has their own
    separate index (uploads stay private), and only the most recently
    uploaded document is searchable at any given time.

    Langfuse par ye poora pipeline ek "document-ingest" trace banata hai
    jiske andar chaar spans hain: extract-text, chunking, embedding,
    index-write. Is se dashboard par saaf nazar aata hai ke 20-page PDF
    upload karne me waqt kahan lag raha hai -- text nikalne me, embeddings
    banane me, ya index likhne me.
    """
    langfuse = _get_langfuse_client()
    file_type = (os.path.splitext(filename)[1].lstrip(".").lower() or "unknown")

    with _observe(
        langfuse, "document-ingest", as_type="span",
        input={"filename": filename, "fileType": file_type},
    ) as root_span:
        with _attributes(
            langfuse,
            user_id=f"user_{user_id}",
            session_id=session_id,
            metadata={"mode": "ingest"},
        ):
            _emit_event(langfuse, "ingest-request", metadata={
                "filename": filename,
                "fileType": file_type,
            })

            # ---- 1. Text extraction ----
            with _observe(langfuse, "extract-text", as_type="span",
                          input={"filename": filename, "fileType": file_type}) as span:
                text = extract_text(file_path)
                _update(span, output={
                    "characters": len(text),
                    "isEmpty": not text.strip(),
                })

            # ---- 2. Chunking ----
            with _observe(langfuse, "chunking", as_type="span",
                          input={"chunkSize": CHUNK_SIZE, "overlap": CHUNK_OVERLAP}) as span:
                chunks = chunk_text(text)
                _update(span, output={
                    "chunkCount": len(chunks),
                    "avgChunkChars": (
                        round(sum(len(c) for c in chunks) / len(chunks)) if chunks else 0
                    ),
                })

            if not chunks:
                # Scanned PDF ya khaali file -- yahin ruk jate hain.
                _emit_event(langfuse, "ingest-skipped", metadata={
                    "reason": "no extractable text (scanned PDF or empty file?)",
                    "filename": filename,
                })
                _tag_trace(
                    langfuse,
                    tags=[TRACE_APP_TAG, "ingest", "empty-document"],
                    output={"chunkCount": 0, "indexed": False},
                )
                _update(root_span, output={"chunkCount": 0, "indexed": False})
                _flush(langfuse)
                return 0

            # ---- 3. Embedding ----
            with _observe(langfuse, "embedding", as_type="span",
                          input={"model": EMBED_MODEL_NAME, "chunkCount": len(chunks)}) as span:
                model = get_embed_model()
                embeddings = model.encode(
                    chunks, convert_to_numpy=True,
                    show_progress_bar=False, normalize_embeddings=True,
                )
                embeddings = embeddings.astype("float32")
                _update(span, output={
                    "vectors": int(embeddings.shape[0]),
                    "dimensions": int(embeddings.shape[1]),
                })

            # ---- 4. Index write ----
            with _observe(langfuse, "index-write", as_type="span") as span:
                index_path, meta_path = _paths_for_user(user_id)

                # Start a fresh index for every new upload, discarding any
                # previously uploaded document's chunks. This ensures questions
                # are answered only from the most recently uploaded document.
                index = faiss.IndexFlatL2(embeddings.shape[1])
                index.add(embeddings)
                metadata = [{"text": chunk, "source": filename} for chunk in chunks]

                faiss.write_index(index, index_path)
                with open(meta_path, "wb") as f:
                    pickle.dump(metadata, f)

                _update(span, output={
                    "vectorsInIndex": int(index.ntotal),
                    "replacedPreviousIndex": True,
                })

            _emit_event(langfuse, "ingest-completed", metadata={
                "filename": filename,
                "chunkCount": len(chunks),
            })
            _tag_trace(
                langfuse,
                tags=[TRACE_APP_TAG, "ingest", file_type],
                output={"filename": filename, "chunkCount": len(chunks), "indexed": True},
            )

        _update(root_span, output={
            "filename": filename,
            "chunkCount": len(chunks),
            "indexed": True,
        })

    _flush(langfuse)
    return len(chunks)


def retrieve_relevant_chunks(user_id, query, top_k=4, query_vec=None):
    """
    Fetches the top-k chunks most relevant to the query from the user's index.

    `query_vec`: pehle se bana hua query embedding. answer_question() ise
    pass karti hai taake embedding apne alag Langfuse span me ho. Na diya
    jaye to yahin bana liya jata hai (purane callers waise hi chalte rehte
    hain).

    Note: we deliberately do NOT hard-filter these by embedding distance.
    Short/meta-phrased questions (e.g. "what is the candidate's name in the
    document") often score a weak embedding-similarity distance against the
    actual resume text even though they ARE about the document -- and that
    score range overlaps with genuinely unrelated questions. A fixed
    threshold can't reliably tell the two apart. Instead, all top-k chunks
    are always handed to the LLM, and the "rag-answer" prompt itself judges
    whether the content is actually relevant (see generate_answer() and the
    SOURCE_USED marker it parses from the model's answer).
    """
    index_path, meta_path = _paths_for_user(user_id)
    if not os.path.exists(index_path):
        return []

    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    if query_vec is None:
        query_vec = embed_query(query)

    k = min(top_k, index.ntotal)
    if k == 0:
        return []
    distances, indices = index.search(query_vec, k)

    # Har chunk ke sath uski distance aur rank bhi rakh lete hain -- ye
    # Langfuse ke retrieval span me chala jata hai, jahan se andaza hota
    # hai ke match kitna mazboot tha.
    results = []
    for rank, (idx, distance) in enumerate(zip(indices[0], distances[0]), start=1):
        if 0 <= idx < len(metadata):
            chunk = dict(metadata[idx])
            chunk["distance"] = float(distance)
            chunk["rank"] = rank
            results.append(chunk)

    if len(distances[0]) > 0:
        print(f"[RAG] query={query!r} best_distance={distances[0][0]:.3f} (informational only, not filtered)")

    return results
