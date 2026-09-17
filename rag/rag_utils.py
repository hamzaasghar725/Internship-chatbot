import os
import pickle
import re
import time
from contextlib import contextmanager, nullcontext
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


# ==========================================================================
# Langfuse tracing helpers
# ==========================================================================
# Ye helpers jo trace banate hain, uski shakal Langfuse dashboard par aisi
# nazar aati hai:
#
#   rag-chat                    (root span)
#     |- chat-request           (event)      sawal aaya: length, top_k
#     |- embedding              (span)       query -> vector  [kitna waqt]
#     |- retrieval              (retriever)  FAISS search     [kitna waqt]
#     |- mode-selected          (event)      rag ya general-chat, aur kyun
#     |- rag-answer             (generation) Gemini call + tokens + cost
#     |- chat-completed         (event)      answerLength, sourceCount
#
#   document-ingest             (root span)
#     |- extract-text           (span)       PDF/TXT se text nikalna
#     |- chunking               (span)       chunkCount, avg chunk size
#     |- embedding              (span)       vectors, dimensions
#     |- index-write            (span)       FAISS index disk par likhna
#
# Pehle sirf retrieval aur generation nazar aate the, is liye ye pata hi
# nahi chalta tha ke waqt kahan ja raha hai -- embedding me, FAISS search
# me, ya Gemini me. Ab har step ka apna span hai.
#
# SAB SE AHEM USOOL: tracing kabhi bhi chat ko nahi tor sakti. Har helper
# try/except me hai aur Langfuse na ho to sab kuch chup chaap normal chalta
# rehta hai -- yehi wajah hai ke ab code ka ek hi raasta hai, pehle ki tarah
# "agar langfuse hai to ye, warna wo" wali do alag copies nahi.

APP_RELEASE = os.environ.get("APP_RELEASE", "1.0.0")
LANGFUSE_ENVIRONMENT = os.environ.get("LANGFUSE_ENVIRONMENT", "production")
TRACE_APP_TAG = "internship-chatbot"  # har trace par lagta hai, filter karne ke liye

# Langfuse ye do values client banate waqt environment se uthata hai, is
# liye inhe get_client() se PEHLE set karna zaroori hai. Isi se dashboard
# par "Env: production" aur "Release: 1.0.0" wale badges aate hain.
os.environ.setdefault("LANGFUSE_TRACING_ENVIRONMENT", LANGFUSE_ENVIRONMENT)
os.environ.setdefault("LANGFUSE_RELEASE", APP_RELEASE)

_trace_warned = set()


def _trace_warn(key, error):
    """Ek hi tracing warning console me baar baar na chhape."""
    if key in _trace_warned:
        return
    _trace_warned.add(key)
    print(f"[Langfuse] '{key}' skipped ({error}) -- tracing degraded, chat unaffected.")


@contextmanager
def _observe(langfuse, name, as_type="span", **kwargs):
    """
    Langfuse observation kholta hai aur uska handle deta hai.

    Langfuse configured na ho -- ya span khulne me koi masla ho -- to `None`
    milta hai aur `with` block bilkul normally chalta rehta hai. Yani call
    karne wale code ko kabhi check nahi karna parta ke tracing on hai ya nahi.

    Note: span banane ki ghalti yahan pakri jati hai, lekin `with` block ke
    ANDAR ki asli ghalti upar jati hai -- warna chat ka error tracing ke
    peeche chhup jata.
    """
    ctx = None
    if langfuse is not None:
        try:
            ctx = langfuse.start_as_current_observation(name=name, as_type=as_type, **kwargs)
        except Exception as e:
            _trace_warn(f"span:{name}", e)
    if ctx is None:
        yield None
        return
    with ctx as observation:
        yield observation


def _attributes(langfuse, **kwargs):
    """propagate_attributes() ka safe version (Langfuse na ho to no-op)."""
    if langfuse is None:
        return nullcontext()
    try:
        return propagate_attributes(**kwargs)
    except Exception as e:
        _trace_warn("propagate_attributes", e)
        return nullcontext()


def _update(observation, **kwargs):
    """span.update() ka safe version."""
    if observation is None:
        return
    try:
        observation.update(**kwargs)
    except Exception as e:
        _trace_warn("observation.update", e)


def _emit_event(langfuse, name, metadata=None):
    """
    Trace par ek point-in-time event lagata hai (span ki tarah duration
    nahi hoti -- sirf "ye hua, is waqt"). Dashboard par ye chhote circle
    wale nodes bante hain: chat-request, mode-selected, chat-completed.

    SDK versions me method ka naam alag ho sakta hai, is liye pehle
    create_event() try karte hain, phir event-type observation.
    """
    if langfuse is None:
        return
    payload = metadata or {}
    try:
        langfuse.create_event(name=name, metadata=payload)
        return
    except Exception as e:
        # Python `except ... as e` block ke baad `e` khud delete kar deta
        # hai, is liye message ko abhi alag variable me mehfooz karte hain.
        first_error = str(e)
    try:
        with langfuse.start_as_current_observation(name=name, as_type="event", metadata=payload):
            pass
    except Exception:
        _trace_warn(f"event:{name}", first_error)


def _tag_trace(langfuse, tags=None, output=None):
    """
    Poore trace par tags aur ek structured output lagata hai.

    Tags hi wo cheez hain jin se dashboard par filter lagta hai -- misaal ke
    taur par sirf wo sawal dekhna jinme document use hi nahi hua
    ("no-context"), ya sirf ingestion traces.
    """
    if langfuse is None:
        return
    kwargs = {}
    if tags:
        kwargs["tags"] = [t for t in tags if t]
    if output is not None:
        kwargs["output"] = output
    if not kwargs:
        return
    try:
        langfuse.update_current_trace(**kwargs)
    except Exception as e:
        _trace_warn("update_current_trace", e)


def _flush(langfuse):
    """Trace ko foran Langfuse bhej deta hai (warna batch me atka rehta hai)."""
    if langfuse is None:
        return
    try:
        langfuse.flush()
    except Exception as e:
        _trace_warn("flush", e)


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


def embed_query(query):
    """
    Sawal ko vector me badalta hai.

    Pehle ye kaam retrieve_relevant_chunks() ke andar chhupa hua tha, is
    liye Langfuse par embedding aur FAISS search dono ka waqt ek hi span me
    mila jula nazar aata tha. Ab alag function hai taake "embedding" apna
    span bana sake aur dashboard par pata chale ke asal me dair kis step me
    ho rahi hai.
    """
    model = get_embed_model()
    return model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")


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

    This is the single entry point /ask should call.

    Langfuse par ye trace banti hai:

        rag-chat                       (root span)
          |- chat-request              (event)      sawal ki tafseel
          |- embedding                 (span)       query -> vector
          |- retrieval                 (retriever)  FAISS search
          |- mode-selected             (event)      rag ya general-chat
          |- rag-answer / general-chat (generation) Gemini call + tokens
          |- chat-completed            (event)      answer ki tafseel

    Pehle sirf retrieval aur generation nazar aate the. Ab embedding ka
    apna span hai, aur teen events se ye bhi record hota hai ke bot ne
    kaun sa mode chuna aur kyun -- jo debugging me sab se zyada kaam aata
    hai jab jawab document ke bajaye general knowledge se aa jaye.

    Langfuse configured na ho to ye sab helpers no-op ban jate hain aur
    pipeline bilkul waise hi chalti hai, bas tracing ke baghair.
    """
    langfuse = _get_langfuse_client()

    with _observe(langfuse, "rag-chat", as_type="span", input={"query": query}) as root_span:
        # Trace-level attributes -- inhi se Langfuse dashboard par User /
        # Session wali pills banti hain aur filter lagta hai.
        # propagate_attributes() inhe is span AND andar khulne wale har
        # span par laga deta hai.
        with _attributes(
            langfuse,
            user_id=f"user_{user_id}",
            session_id=session_id,
            metadata={"mode": "chat"},
        ):
            _emit_event(langfuse, "chat-request", metadata={
                "queryLength": len(query),
                "queryWords": len(query.split()),
                "topK": top_k,
                "hasDocument": _user_has_index(user_id),
            })

            # ---- 1. Query embedding ----
            with _observe(langfuse, "embedding", as_type="span",
                          input={"model": EMBED_MODEL_NAME, "query": query}) as span:
                query_vec = embed_query(query)
                _update(span, output={"dimensions": int(query_vec.shape[1])})

            # ---- 2. Retrieval ----
            with _observe(langfuse, "retrieval", as_type="retriever",
                          input={"query": query, "topK": top_k}) as span:
                chunks = retrieve_relevant_chunks(user_id, query, top_k=top_k, query_vec=query_vec)
                retrieved_sources = list({c["source"] for c in chunks})
                _update(span, output={
                    "chunkCount": len(chunks),
                    "sources": retrieved_sources,
                    "bestDistance": round(chunks[0]["distance"], 4) if chunks else None,
                })

            # ---- 3. Mode ----
            mode = "rag" if chunks else "general-chat"
            _emit_event(langfuse, "mode-selected", metadata={
                "mode": mode,
                "reason": (
                    "document chunks retrieved, answering with context"
                    if chunks else
                    "no document indexed for this user, answering from general knowledge"
                ),
            })

            # ---- 4. Generation ----
            # generate_answer() -> _call_gemini() apna "generation"
            # observation khud kholta hai; hum abhi bhi is context ke andar
            # hain, is liye Langfuse usay rag-chat ke neeche nest kar deta hai.
            answer, source_used = generate_answer(query, chunks)
            sources = retrieved_sources if source_used is not False else []

            _emit_event(langfuse, "chat-completed", metadata={
                "answerLength": len(answer or ""),
                "sourceCount": len(sources),
                "sources": sources,
                "contextUsed": source_used,
            })

            _tag_trace(
                langfuse,
                tags=[
                    TRACE_APP_TAG,
                    "chat",
                    mode,
                    "context-used" if sources else "no-context",
                ],
                output={
                    "answerLength": len(answer or ""),
                    "sourceCount": len(sources),
                    "sources": sources,
                },
            )

        _update(root_span, output={
            "answer": answer,
            "sources": sources,
            "answerLength": len(answer or ""),
            "sourceCount": len(sources),
            "sourceUsed": source_used,
        })

    _flush(langfuse)
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


# ==========================================================================
# Answer polishing
# ==========================================================================
# Prompt model se professional markdown maangta hai, lekin LLM kabhi kabhi
# rules todta hai: "* item" bullets, adhoora "**Heading" (closing ** ghayab),
# ya "Certainly!" jaisa filler. Frontend (markdown.js) waise to in sab ko
# handle kar leta hai, magar us se pehle yahan text saaf karne ke 3 faide
# hain:
#   1. Database me bhi saaf jawab save hota hai (history, export, audit).
#   2. Copy aur text-to-speech ko saaf text milta hai.
#   3. Kal koi doosra client (mobile app, API) bane to usay bhi saaf mile.
#
# Ye sirf formatting theek karta hai -- jawab ka matlab kabhi nahi badalta.

_FILLER_OPENERS = re.compile(
    r"^\s*(certainly|sure|of course|absolutely|great question|good question|"
    r"that's a great question|happy to help|no problem)\b[!,.\s]*",
    re.IGNORECASE,
)

# "According to the resume, X" -> "X". Model ko style guide me mana kiya
# gaya hai, lekin ye safety net hai agar wo phir bhi likh de -- khaas taur
# par chhote factual jawabon me ("naam kya hai") ye phrase sirf shor hoti
# hai, koi maani nahi rakhti.
_DOCUMENT_ATTRIBUTION = re.compile(
    r"^\s*(according to (the |his |her |their )?"
    r"(resume|document|cv|file|context|text)s?,?\s*)",
    re.IGNORECASE,
)


def _polish_line(line):
    """Ek line ke markdown markers durust karta hai."""
    # "* item" / "+ item" ko standard "- item" bullet bana dete hain.
    line = re.sub(r"^(\s*)[*+]\s+(?=\S)", r"\1- ", line)

    # "***" kabhi bhi valid emphasis nahi -- bold+italic ka mila jula
    # markup jo aksar toota hua render hota hai. Bold par le aate hain.
    line = re.sub(r"\*{3,}", "**", line)

    # Adhoore bold markers: agar line par "**" ki ginti taaq (odd) hai to
    # aakhri wala orphan hai. Usay hata dete hain -- band karne ke bajaye,
    # kyunke band karne se poori line ghalti se bold ho sakti hai.
    if line.count("**") % 2 == 1:
        idx = line.rfind("**")
        line = line[:idx] + line[idx + 2:]

    # "**Label:**value" -> "**Label:** value" (colon ke baad space)
    line = re.sub(r"(\*\*[^*\n]+\*\*):(?=\S)", r"\1: ", line)
    line = re.sub(r"(\*\*[^*\n]+:\*\*)(?=\S)", r"\1 ", line)

    return line.rstrip()


def polish_answer(text):
    """
    Model ke jawab ki formatting ko normalize karta hai.

    Code blocks (``` ... ```) ko chhu kar bhi nahi dekhte -- unke andar
    asterisks aur indentation asli code ka hissa ho sakte hain.
    """
    if not text or not text.strip():
        return text

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _FILLER_OPENERS.sub("", text, count=1)

    # Har paragraph ke shuru se "According to the document/resume, " hata
    # dete hain -- ye sirf pehli line par nahi, kabhi kabhi model beech me
    # naye paragraph ke saath bhi ye phrase dohra deta hai.
    lines_for_attribution = text.split("\n")
    for idx, line in enumerate(lines_for_attribution):
        stripped = _DOCUMENT_ATTRIBUTION.sub("", line)
        if stripped != line and stripped:
            stripped = stripped[0].upper() + stripped[1:]
        lines_for_attribution[idx] = stripped
    text = "\n".join(lines_for_attribution)

    polished = []
    in_code_block = False
    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            in_code_block = not in_code_block
            polished.append(line.rstrip())
            continue
        polished.append(line if in_code_block else _polish_line(line))

    text = "\n".join(polished)

    # Heading se pehle khaali line -- warna wo upar wale paragraph se chipak
    # jati hai aur markdown parser usay heading maan hi nahi pata.
    text = re.sub(r"(?<!\n)\n(#{1,6}\s)", r"\n\n\1", text)

    # Do se zyada khaali lines kabhi zaroori nahi hotin.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


_TRAILING_MARKER_RE = re.compile(
    r"\n?\s*\[\[(SOURCE_USED|ANSWER_TYPE):\s*([A-Z]+)\s*\]\]\s*$", re.IGNORECASE
)


def _split_markers(answer):
    """
    The 'rag-answer' prompt ends its reply with hidden marker lines like
    "[[SOURCE_USED: YES]]" and "[[ANSWER_TYPE: FACT]]" so the code -- not a
    guess -- knows (a) whether the document context was actually used, and
    (b) whether this answer is a short document fact or a proper
    explanation. Strips both markers (in either order, one or both present)
    and returns (clean_answer, source_used, answer_type).

        source_used: True / False / None (None = no parseable marker --
            older prompt version, or the model just forgot)
        answer_type: "FACT" / "EXPLANATION" / None (None = same as above)

    Looping (instead of matching both at once) means it doesn't matter
    which marker the model wrote first.
    """
    if not answer:
        return answer, None, None

    text = answer
    source_used = None
    answer_type = None

    while True:
        match = _TRAILING_MARKER_RE.search(text)
        if not match:
            break
        key = match.group(1).upper()
        value = match.group(2).upper()
        if key == "SOURCE_USED" and value in ("YES", "NO"):
            source_used = (value == "YES")
        elif key == "ANSWER_TYPE" and value in ("FACT", "EXPLANATION"):
            answer_type = value
        text = text[:match.start()].rstrip()

    return text, source_used, answer_type


# ==========================================================================
# Scope enforcement -- jab prompt kaafi na ho
# ==========================================================================
# STYLE_GUIDE model ko "sirf jo poocha gaya wahi do" keh chuka hai, lekin
# Gemini kabhi kabhi phir bhi extra sentences jorta hai (jaise resume ka
# poora background). Prompt par bharosa karne ke bajaye yahan ek deterministic
# check hai: agar sawal chhota aur seedha factual tha (naam, tareekh, number,
# yes/no), aur jawab me headings/bullets/table nahi hain (matlab model ne
# genuinely ek fact ka jawab paragraph bana diya), to sirf pehla jumla
# rakhte hain, baaki hata dete hain.
#
# Ye sirf plain-paragraph jawabon par lagta hai -- agar jawab me "## heading"
# ya "- bullet" ya table hai, to samajh lete hain ke sawal ne genuinely
# structure maanga tha aur kuch nahi chhedte.

# In lafzon me se koi bhi ho to samajh lo user ne khud tafseel maangi hai --
# aise sawal ka lamba jawab "over-answering" nahi, sahi jawab hai.
_DETAIL_REQUESTED_RE = re.compile(
    r"\b(explain|describe|elaborate|summar(y|ize|ise)|overview|breakdown|"
    r"detail|details|list|compare|discuss|walk me through|tell me (more )?about|"
    r"why|how does|how do|how did|how can|pros and cons|advantages|"
    r"background|full|everything|all about)\b",
    re.IGNORECASE,
)

# Ek chhota factual sawal aam taur par in shuru honay wale lafzon se pehchana
# jata hai (naam/tareekh/number/yes-ya-no poochne wale sawal).
_SHORT_FACT_QUESTION_RE = re.compile(
    r"^\s*(what|who|when|where|which|how (many|much|old)|is|are|does|do|"
    r"can|will|did)\b",
    re.IGNORECASE,
)

_STRUCTURED_ANSWER_RE = re.compile(r"^\s{0,3}(#{1,6}\s|[-*+]\s|\d+[.)]\s|\|)", re.MULTILINE)


def _split_sentences(paragraph):
    """
    Ek paragraph ko jumlon me torta hai. Simple hai (regex-based, koi NLP
    library nahi), lekin is kaam ke liye kaafi hai -- hume sirf "pehla jumla
    kahan khatam hota hai" pata karna hai.
    """
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", paragraph.strip()) if s.strip()]


def enforce_scope(query, answer):
    """
    Chhote factual sawalon (naam, tareekh, number, yes/no) ke jawab ko sirf
    pehle jumle tak mehdood karta hai, agar model ne extra paragraph jor
    diya ho.

    Deterministic hai -- model ke "kam likho" maan lene par bharosa nahi
    karta, khud check kar ke faisla karta hai. Isi liye ye function ka naam
    "enforce" hai, "ask" nahi.
    """
    if not answer or not answer.strip():
        return answer

    # Sawal khud tafseel maang raha tha -- kuch mat chhero.
    if _DETAIL_REQUESTED_RE.search(query):
        return answer

    words = query.strip().split()
    looks_like_short_fact = (
        len(words) <= 10 and bool(_SHORT_FACT_QUESTION_RE.match(query.strip()))
    )
    if not looks_like_short_fact:
        return answer

    # Jawab pehle se hi structured hai (heading/bullet/table) -- matlab is
    # sawal ka jawab genuinely woh structure maangta tha, chhero mat.
    if _STRUCTURED_ANSWER_RE.search(answer):
        return answer

    paragraphs = [p for p in answer.split("\n\n") if p.strip()]
    if not paragraphs:
        return answer

    first_para_sentences = _split_sentences(paragraphs[0])
    extra_paragraphs = len(paragraphs) > 1
    extra_sentences = len(first_para_sentences) > 1

    if not extra_paragraphs and not extra_sentences:
        return answer  # Jawab pehle se hi ek jumle ka hai, kuch karne ki zaroorat nahi

    return first_para_sentences[0] if first_para_sentences else answer


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
        # No document was even involved here, so this is never a candidate
        # for scope-trimming: general questions ("what is machine
        # learning?") always get a proper explanatory answer.
        prompt, langfuse_prompt = get_prompt("general-chat", question=query)
        metadata = {"mode": "general-chat", "selected_sources": [], "context_chunks": 0}
        answer = _safe_call_gemini(prompt, "general-chat", metadata, langfuse_prompt)
        if answer is None:
            return ("(GEMINI_API_KEY is not set, so I can't answer general questions right now. "
                    "You can still upload a document to test retrieval.)"), None
        return polish_answer(answer), None

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

    # Dono hidden markers alag karte hain, phir formatting saaf karte hain.
    answer, source_used, answer_type = _split_markers(answer)
    answer = polish_answer(answer)

    # Scope-trimming SIRF tab lagti hai jab model ne khud kaha ho ke ye
    # jawab document se aaya ek "FACT" hai (naam, tareekh, number). Agar
    # model ne "EXPLANATION" kaha -- matlab term document me sirf naam se
    # tha aur jawab general knowledge se banaya gaya -- to poori tafseel
    # rehti hai, kaati nahi jati.
    #
    # Agar marker parse hi nahi hua (purana/cached prompt jisme marker
    # instruction nahi thi), to purane word-based heuristic par wapas chale
    # jate hain -- yehi safety net hai taake behavior kabhi achanak na
    # bigde agar Langfuse par koi purana prompt version chal raha ho.
    if answer_type == "FACT":
        answer = enforce_scope(query, answer)
    elif answer_type is None:
        answer = enforce_scope(query, answer)
    # answer_type == "EXPLANATION" -> kuch nahi chhedte, poori tafseel rehti hai.

    return answer, source_used


def summarize_document(user_id, filename=None, session_id=None):
    """
    Generates a summary of the user's uploaded document(s).
    If filename is given, only that file's chunks are summarized; otherwise all of them.

    Langfuse par apni alag "rag-summary" trace banti hai, ab events ke sath:

        rag-summary                (root span)
          |- summary-request       (event)
          |- load-document         (span)       kitne chunks mile
          |- rag-summary-generation(generation) Gemini call
          |- summary-completed     (event)      summary ki length

    answer_question() jaisa hi dhancha, taake dashboard par dono ek jaise
    lagen aur tags se filter ho saken.
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

    target = filename or "all documents"

    with _observe(langfuse, "rag-summary", as_type="span",
                  input={"filename": target}) as root_span:
        with _attributes(
            langfuse,
            user_id=f"user_{user_id}",
            session_id=session_id,
            metadata={"mode": "summary"},
        ):
            _emit_event(langfuse, "summary-request", metadata={"filename": target})

            with _observe(langfuse, "load-document", as_type="span") as span:
                chunks = _load_chunks()
                _update(span, output={
                    "chunkCount": len(chunks) if chunks is not None else 0,
                    "documentFound": chunks is not None,
                })

            if chunks is None:
                _emit_event(langfuse, "summary-skipped", metadata={
                    "reason": "no document uploaded yet",
                })
                _tag_trace(
                    langfuse,
                    tags=[TRACE_APP_TAG, "summary", "no-document"],
                    output={"error": "no document uploaded"},
                )
                _update(root_span, output={"error": "no document uploaded"})
                _flush(langfuse)
                return "No document has been uploaded yet. Please upload a document first."

            summary = _summarize_chunks(chunks, filename)

            _emit_event(langfuse, "summary-completed", metadata={
                "summaryLength": len(summary or ""),
                "chunkCount": len(chunks),
            })
            _tag_trace(
                langfuse,
                tags=[TRACE_APP_TAG, "summary", "document"],
                output={
                    "summaryLength": len(summary or ""),
                    "chunkCount": len(chunks),
                },
            )

        _update(root_span, output={
            "summary": summary,
            "summaryLength": len(summary or ""),
        })

    _flush(langfuse)
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
    return polish_answer(summary)