import os
import pickle
import time
import numpy as np
import faiss
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import requests
import certifi

# Some Windows setups have a stale SSL_CERT_FILE environment variable pointing
# to a certificate file that no longer exists, which crashes any library that
# creates its own HTTPS client (like Langfuse's httpx client). Fix it here by
# falling back to certifi's bundled certificates if the current path is invalid.
if not os.environ.get("SSL_CERT_FILE") or not os.path.exists(os.environ["SSL_CERT_FILE"]):
    os.environ["SSL_CERT_FILE"] = certifi.where()

from langfuse import get_client, propagate_attributes
from rag.prompt_registry import get_prompt

# ---- Config ----
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# Google periodically deprecates/renames Gemini models. Instead of hardcoding
# a single name, we try several candidates and cache whichever one works for
# subsequent calls.
GEMINI_MODEL_CANDIDATES = [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-flash-latest",
    "gemini-pro-latest",
]
_working_model_name = None  # whichever model succeeds gets cached here
CHUNK_SIZE = 500        # characters per chunk
CHUNK_OVERLAP = 50
VECTORSTORE_DIR = os.path.join(os.path.dirname(__file__), "..", "vectorstore")

_embed_model = None
_langfuse_client = None


def _get_langfuse_client():
    """
    Returns the Langfuse client only if keys are configured in .env, else None.
    This keeps Langfuse fully optional: the app works fine without it.
    """
    global _langfuse_client
    if not os.environ.get("LANGFUSE_PUBLIC_KEY"):
        return None
    if _langfuse_client is None:
        _langfuse_client = get_client()
    return _langfuse_client


def get_embed_model():
    """Lazy-loads the embedding model (only loads once)."""
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model


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


def build_or_update_index(user_id, file_path, filename):
    """
    Processes a new document and builds a fresh FAISS index for the user,
    replacing any previously uploaded document. Each user has their own
    separate index (uploads stay private), and only the most recently
    uploaded document is searchable at any given time.
    """
    text = extract_text(file_path)
    chunks = chunk_text(text)
    if not chunks:
        return 0

    model = get_embed_model()
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    embeddings = embeddings.astype("float32")

    index_path, meta_path = _paths_for_user(user_id)

    # Start a fresh index for every new upload, discarding any previously
    # uploaded document's chunks. This ensures questions are answered only
    # from the most recently uploaded document, not older ones.
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    metadata = []

    index.add(embeddings)
    for chunk in chunks:
        metadata.append({"text": chunk, "source": filename})

    faiss.write_index(index, index_path)
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)

    return len(chunks)


def retrieve_relevant_chunks(user_id, query, top_k=4):
    """Fetches the top-k chunks most relevant to the query from the user's index."""
    index_path, meta_path = _paths_for_user(user_id)
    if not os.path.exists(index_path):
        return []

    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    model = get_embed_model()
    query_vec = model.encode([query], convert_to_numpy=True).astype("float32")

    k = min(top_k, index.ntotal)
    if k == 0:
        return []
    distances, indices = index.search(query_vec, k)

    results = []
    for idx in indices[0]:
        if 0 <= idx < len(metadata):
            results.append(metadata[idx])
    return results


GEMINI_TEMPERATURE = 0.3
GEMINI_MAX_OUTPUT_TOKENS = 1024  # generous ceiling so real answers don't get cut off,
                                 # but still a real, reportable model parameter


def _post_to_gemini(model_name, api_key, prompt, max_retries=2):
    """
    Calls Gemini with one specific model name, retrying on temporary errors.
    Returns (answer_text, usage_dict) where usage_dict has input/output/total tokens.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
    last_error = None
    for attempt in range(max_retries):
        try:
            response = requests.post(
                url,
                headers={
                    "x-goog-api-key": api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": GEMINI_TEMPERATURE,
                        "maxOutputTokens": GEMINI_MAX_OUTPUT_TOKENS,
                    },
                },
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            answer_text = data["candidates"][0]["content"]["parts"][0]["text"]
            usage = data.get("usageMetadata", {})
            token_usage = {
                "input_tokens": usage.get("promptTokenCount", 0),
                "output_tokens": usage.get("candidatesTokenCount", 0),
                "total_tokens": usage.get("totalTokenCount", 0),
            }
            return answer_text, token_usage
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            last_error = e
            if status in (429, 500, 503):
                time.sleep(2 * (attempt + 1))
                continue
            raise  # 404, 401, 400 etc. -> raise immediately (retrying won't help)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            # Network slow or connection dropped, wait a bit and retry
            last_error = e
            time.sleep(2 * (attempt + 1))
            continue
    raise last_error


def _call_gemini(prompt, trace_name="gemini-call", metadata=None, langfuse_prompt=None):
    """
    Calls the Gemini REST API directly over HTTP (no SDK, to avoid
    protobuf/tensorflow version conflicts).
    Returns None if GEMINI_API_KEY is not set.
    Model names change over time (Google deprecates them), so we try
    several candidates until one works.
    Logs the call (model, prompt, answer, token usage, temperature,
    max_completion_tokens, and any extra `metadata` passed in -- e.g. mode,
    selected_sources) to Langfuse if LANGFUSE_PUBLIC_KEY is configured in
    .env; otherwise this is skipped. This is what populates the Preview
    panel's metadata table in the Langfuse UI.
    `langfuse_prompt` (optional): the prompt object returned by
    prompt_registry.get_prompt(). When provided, it's linked to this
    generation so the Langfuse UI shows which prompt version produced it.
    """
    global _working_model_name
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None

    langfuse = _get_langfuse_client()

    # Try whichever model worked last time first (if any)
    models_to_try = ([_working_model_name] if _working_model_name else []) + \
        [m for m in GEMINI_MODEL_CANDIDATES if m != _working_model_name]

    last_error = None
    for model_name in models_to_try:
        try:
            if langfuse:
                with langfuse.start_as_current_observation(
                    as_type="generation",
                    name=trace_name,
                    model=model_name,
                    input=prompt,
                    model_parameters={
                        "temperature": GEMINI_TEMPERATURE,
                        "max_completion_tokens": GEMINI_MAX_OUTPUT_TOKENS,
                    },
                    metadata=metadata or {},
                    prompt=langfuse_prompt,
                ) as generation:
                    answer, token_usage = _post_to_gemini(model_name, api_key, prompt)
                    generation.update(output=answer, usage_details=token_usage)
                langfuse.flush()  # send the trace to Langfuse right away
            else:
                answer, token_usage = _post_to_gemini(model_name, api_key, prompt)

            _working_model_name = model_name  # cache this model for next time
            return answer
        except requests.exceptions.HTTPError as e:
            last_error = e
            status = e.response.status_code if e.response is not None else None
            body = e.response.text[:300] if e.response is not None else str(e)
            print(f"[Gemini] Model '{model_name}' failed (status={status}): {body}")
            if status in (401, 403, 400):
                raise  # auth/bad-request issues, trying another model won't help
            continue  # 404 (model not found) or 429/500/503 (overloaded) -> try next model
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_error = e
            print(f"[Gemini] Model '{model_name}' timeout/connection error: {e}")
            continue  # network slow/unstable -> try next model

    # None of the candidate models worked
    raise last_error


def answer_question(user_id, query, session_id=None, top_k=4):
    """
    Full RAG pipeline for one user question: retrieval + generation.

    This is the single entry point /ask should call. When Langfuse is
    configured it wraps the whole thing in ONE parent trace/span
    ("rag-chat") with two children nested under it:
        rag-chat (root span)
          |- retrieval        (span: which chunks were fetched)
          |- rag-answer       (generation: the LLM call, from _call_gemini)
    That's what produces the trace graph (root -> retrieval -> generation)
    in the Langfuse UI, instead of a single disconnected generation.
    Without Langfuse configured, this just runs retrieval + generation with
    no tracing overhead, exactly as before.
    """
    langfuse = _get_langfuse_client()

    if not langfuse:
        chunks = retrieve_relevant_chunks(user_id, query, top_k=top_k)
        answer = generate_answer(query, chunks)
        sources = list({c["source"] for c in chunks})
        return answer, sources

    with langfuse.start_as_current_observation(
        name="rag-chat",
        as_type="span",
        input={"query": query},
    ) as root_span:
        # Trace-level attributes -- these are what let you filter/search by
        # user or conversation in the Langfuse dashboard (Session/User ID
        # pills you saw in the QueryMind screenshot). propagate_attributes()
        # applies them to this span AND every child span opened inside it.
        with propagate_attributes(
            user_id=f"user_{user_id}",
            session_id=session_id,
            metadata={"mode": "chat"},
        ):
            with langfuse.start_as_current_observation(
                name="retrieval",
                as_type="retriever",
                input={"query": query, "top_k": top_k},
            ) as retrieval_span:
                chunks = retrieve_relevant_chunks(user_id, query, top_k=top_k)
                sources = list({c["source"] for c in chunks})
                retrieval_span.update(
                    output={"num_chunks_found": len(chunks), "sources": sources},
                )

            # generate_answer() -> _call_gemini() opens its own "generation"
            # observation internally; because we're still inside this context,
            # Langfuse automatically nests it under rag-chat.
            answer = generate_answer(query, chunks)

        root_span.update(output={"answer": answer, "sources": sources})

    langfuse.flush()
    return answer, sources


def generate_answer(query, context_chunks):
    """
    Builds context from retrieved chunks and asks the LLM to generate an answer.
    If GEMINI_API_KEY is not set, returns the retrieved text directly
    (so RAG retrieval can still be tested without an API key).
    """
    context = "\n\n".join(f"[Source: {c['source']}]\n{c['text']}" for c in context_chunks)

    if not context_chunks:
        return "I couldn't find any content related to this question in your documents. Please upload a document first."

    prompt, langfuse_prompt = get_prompt("rag-answer", context=context, question=query)

    sources = list({c["source"] for c in context_chunks})
    metadata = {
        "mode": "chat",
        "selected_sources": sources,
        "context_chunks": len(context_chunks),
    }
    answer = _call_gemini(prompt, trace_name="rag-answer", metadata=metadata, langfuse_prompt=langfuse_prompt)
    if answer is None:
        # Fallback: no LLM available, just show the retrieved context directly
        return ("(GEMINI_API_KEY is not set, so showing the retrieved context directly)\n\n"
                + context)
    return answer


def summarize_document(user_id, filename=None, session_id=None):
    """
    Generates a summary of the user's uploaded document(s).
    If filename is given, only that file's chunks are summarized; otherwise all of them.
    Wrapped in a Langfuse trace the same way as answer_question(), so it shows
    up as its own "rag-summary" trace with a nested "load-document" span and
    the generation, rather than a disconnected generation.
    """
    langfuse = _get_langfuse_client()

    def _load_chunks():
        index_path, meta_path = _paths_for_user(user_id)
        if not os.path.exists(meta_path):
            return None
        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)
        if filename:
            return [m["text"] for m in metadata if m["source"] == filename]
        return [m["text"] for m in metadata]

    if not langfuse:
        chunks = _load_chunks()
        if chunks is None:
            return "No document has been uploaded yet. Please upload a document first."
        return _summarize_chunks(chunks, filename)

    with langfuse.start_as_current_observation(
        name="rag-summary",
        as_type="span",
        input={"filename": filename or "all documents"},
    ) as root_span:
        with propagate_attributes(
            user_id=f"user_{user_id}",
            session_id=session_id,
            metadata={"mode": "summary"},
        ):
            with langfuse.start_as_current_observation(name="load-document", as_type="span") as load_span:
                chunks = _load_chunks()
                load_span.update(output={"num_chunks": len(chunks) if chunks is not None else 0})

            if chunks is None:
                root_span.update(output={"error": "no document uploaded"})
                return "No document has been uploaded yet. Please upload a document first."

            summary = _summarize_chunks(chunks, filename)

        root_span.update(output={"summary": summary})

    langfuse.flush()
    return summary


def _summarize_chunks(chunks, filename=None):
    if not chunks:
        return "No document with that name was found."

    full_text = "\n\n".join(chunks)
    # Limit text length for very long documents (to save on token budget)
    full_text = full_text[:12000]

    prompt, langfuse_prompt = get_prompt("doc-summary", document_text=full_text)

    metadata = {
        "mode": "summary",
        "selected_sources": [filename] if filename else "all uploaded documents",
        "context_chunks": len(chunks),
    }
    summary = _call_gemini(prompt, trace_name="rag-summary-generation", metadata=metadata, langfuse_prompt=langfuse_prompt)
    if summary is None:
        return ("(GEMINI_API_KEY is not set, so a summary could not be generated. "
                "Showing a portion of the document below instead)\n\n" + full_text[:1000])
    return summary