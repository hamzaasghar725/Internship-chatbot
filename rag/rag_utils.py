import os
import pickle
import re
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
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
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
    """
    Fetches the top-k chunks most relevant to the query from the user's index.

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

    model = get_embed_model()
    query_vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")

    k = min(top_k, index.ntotal)
    if k == 0:
        return []
    distances, indices = index.search(query_vec, k)

    results = [metadata[idx] for idx in indices[0] if 0 <= idx < len(metadata)]

    if len(distances[0]) > 0:
        print(f"[RAG] query={query!r} best_distance={distances[0][0]:.3f} (informational only, not filtered)")

    return results


GEMINI_TEMPERATURE = 0.3
GEMINI_MAX_OUTPUT_TOKENS = 1024  # generous ceiling so real answers don't get cut off,
                                 # but still a real, reportable model parameter


class GeminiResponseError(Exception):
    """Raised when Gemini returns a 200 OK but the response has no usable
    answer text (e.g. blocked by a safety filter, or hit MAX_TOKENS before
    producing any content). Caught in _call_gemini() so it's treated the same
    as a failed model candidate -- the next candidate model is tried instead
    of crashing the request with a raw KeyError."""
    pass


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

            candidates = data.get("candidates") or []
            if not candidates:
                block_reason = data.get("promptFeedback", {}).get("blockReason")
                raise GeminiResponseError(
                    f"no candidates in response (blockReason={block_reason}, raw={str(data)[:300]})"
                )

            candidate = candidates[0]
            parts = candidate.get("content", {}).get("parts")
            if not parts:
                finish_reason = candidate.get("finishReason")
                raise GeminiResponseError(
                    f"no content parts in response (finishReason={finish_reason}, raw={str(candidate)[:300]})"
                )

            answer_text = parts[0].get("text", "")
            if not answer_text:
                raise GeminiResponseError(f"empty text in response part (raw={str(parts[0])[:300]})")

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
        except GeminiResponseError:
            raise  # malformed/blocked response -- retrying the same model won't help
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
        except GeminiResponseError as e:
            last_error = e
            print(f"[Gemini] Model '{model_name}' returned an unusable response: {e}")
            continue  # blocked/empty response -> try next model

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
        answer, source_used = generate_answer(query, chunks)
        sources = list({c["source"] for c in chunks}) if source_used is not False else []
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
                retrieved_sources = list({c["source"] for c in chunks})
                retrieval_span.update(
                    output={"num_chunks_found": len(chunks), "sources": retrieved_sources},
                )

            # generate_answer() -> _call_gemini() opens its own "generation"
            # observation internally; because we're still inside this context,
            # Langfuse automatically nests it under rag-chat.
            answer, source_used = generate_answer(query, chunks)
            sources = retrieved_sources if source_used is not False else []

        root_span.update(output={"answer": answer, "sources": sources, "source_used": source_used})

    langfuse.flush()
    return answer, sources


def _safe_call_gemini(prompt, trace_name, metadata, langfuse_prompt):
    """
    Wraps _call_gemini() so a total failure (all candidate models timed out,
    or all returned unusable/blocked responses) becomes a friendly message
    instead of an unhandled exception crashing the Flask request with a 500.
    Returns None only when GEMINI_API_KEY isn't set at all (same as before).
    """
    try:
        return _call_gemini(prompt, trace_name=trace_name, metadata=metadata, langfuse_prompt=langfuse_prompt)
    except Exception as e:
        print(f"[Gemini] All candidate models failed for '{trace_name}': {e}")
        return ("Sorry, I couldn't reach the AI model right now (it may be slow or "
                "temporarily unavailable). Please try again in a moment.")


_SOURCE_USED_RE = re.compile(r"\n?\[\[SOURCE_USED:\s*(YES|NO)\s*\]\]\s*$", re.IGNORECASE)


def _split_source_marker(answer):
    """
    The 'rag-answer' prompt ends its reply with a hidden marker line like
    "[[SOURCE_USED: YES]]" or "[[SOURCE_USED: NO]]" so the code -- not an
    embedding-distance guess -- knows whether the document context was
    actually used. Strips the marker from the visible text and returns
    (clean_answer, used_document | None). None means the model didn't
    include a parseable marker (older prompt version, or it just forgot).
    """
    if not answer:
        return answer, None
    match = _SOURCE_USED_RE.search(answer)
    if not match:
        return answer, None
    clean = answer[:match.start()].rstrip()
    used = match.group(1).upper() == "YES"
    return clean, used


def generate_answer(query, context_chunks):
    """
    Builds an answer to the user's question.

    RAG is a *feature*, not a requirement: if a document has been uploaded,
    the top-k retrieved chunks are passed in as extra context that the
    answer CAN be grounded in. If no chunks are available (nothing uploaded
    yet), this answers directly as a general-purpose assistant -- exactly
    like a normal chatbot -- instead of refusing.

    Whether the document was actually relevant is judged by the LLM itself
    (via the "rag-answer" prompt's SOURCE_USED marker), not by embedding
    distance -- short/meta-phrased questions about a document often score a
    weak embedding match against the actual document text even when they
    ARE about it, so a hard distance cutoff can't reliably separate that
    from a genuinely unrelated question.

    Returns (answer_text, source_used):
        source_used is True/False when context_chunks was non-empty and the
        model's marker could be parsed; None when there was no document
        context to begin with (general-chat) or the marker couldn't be
        parsed (caller should keep prior behaviour in that case).

    If GEMINI_API_KEY is not set, falls back to showing the retrieved
    context directly (RAG mode) or a short notice (general mode), so the
    app doesn't crash without a key.
    """
    if not context_chunks:
        # Nothing to ground the answer in -- behave as a general chatbot.
        prompt, langfuse_prompt = get_prompt("general-chat", question=query)
        metadata = {"mode": "general-chat", "selected_sources": [], "context_chunks": 0}
        answer = _safe_call_gemini(prompt, "general-chat", metadata, langfuse_prompt)
        if answer is None:
            return ("(GEMINI_API_KEY is not set, so I can't answer general questions right now. "
                    "You can still upload a document to test retrieval.)"), None
        return answer, None

    context = "\n\n".join(f"[Source: {c['source']}]\n{c['text']}" for c in context_chunks)
    prompt, langfuse_prompt = get_prompt("rag-answer", context=context, question=query)

    sources = list({c["source"] for c in context_chunks})
    metadata = {
        "mode": "chat",
        "selected_sources": sources,
        "context_chunks": len(context_chunks),
    }
    answer = _safe_call_gemini(prompt, "rag-answer", metadata, langfuse_prompt)
    if answer is None:
        # Fallback: no LLM available, just show the retrieved context directly
        return ("(GEMINI_API_KEY is not set, so showing the retrieved context directly)\n\n"
                + context), None

    answer, source_used = _split_source_marker(answer)
    return answer, source_used


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
    summary = _safe_call_gemini(prompt, "rag-summary-generation", metadata, langfuse_prompt)
    if summary is None:
        return ("(GEMINI_API_KEY is not set, so a summary could not be generated. "
                "Showing a portion of the document below instead)\n\n" + full_text[:1000])
    return summary