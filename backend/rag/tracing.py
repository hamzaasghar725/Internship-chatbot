"""
Langfuse tracing helpers.

Ye helpers jo trace banate hain, uski shakal Langfuse dashboard par aisi
nazar aati hai:

  rag-chat                    (root span)
    |- chat-request           (event)      sawal aaya: length, top_k
    |- embedding              (span)       query -> vector  [kitna waqt]
    |- retrieval              (retriever)  FAISS search     [kitna waqt]
    |- mode-selected          (event)      rag ya general-chat, aur kyun
    |- rag-answer             (generation) Gemini call + tokens + cost
    |- chat-completed         (event)      answerLength, sourceCount

  document-ingest             (root span)
    |- extract-text           (span)       PDF/TXT se text nikalna
    |- chunking               (span)       chunkCount, avg chunk size
    |- embedding              (span)       vectors, dimensions
    |- index-write            (span)       FAISS index disk par likhna

Pehle sirf retrieval aur generation nazar aate the, is liye ye pata hi
nahi chalta tha ke waqt kahan ja raha hai -- embedding me, FAISS search
me, ya Gemini me. Ab har step ka apna span hai.

SAB SE AHEM USOOL: tracing kabhi bhi chat ko nahi tor sakti. Har helper
try/except me hai aur Langfuse na ho to sab kuch chup chaap normal chalta
rehta hai -- yehi wajah hai ke ab code ka ek hi raasta hai, pehle ki tarah
"agar langfuse hai to ye, warna wo" wali do alag copies nahi.
"""
import os
from contextlib import contextmanager, nullcontext

import certifi

# Some Windows setups have a stale SSL_CERT_FILE environment variable pointing
# to a certificate file that no longer exists, which crashes any library that
# creates its own HTTPS client (like Langfuse's httpx client). Fix it here by
# falling back to certifi's bundled certificates if the current path is invalid.
if not os.environ.get("SSL_CERT_FILE") or not os.path.exists(os.environ["SSL_CERT_FILE"]):
    os.environ["SSL_CERT_FILE"] = certifi.where()

from langfuse import get_client, propagate_attributes

APP_RELEASE = os.environ.get("APP_RELEASE", "1.0.0")
LANGFUSE_ENVIRONMENT = os.environ.get("LANGFUSE_ENVIRONMENT", "production")
TRACE_APP_TAG = "internship-chatbot"  # har trace par lagta hai, filter karne ke liye

# Langfuse ye do values client banate waqt environment se uthata hai, is
# liye inhe get_client() se PEHLE set karna zaroori hai. Isi se dashboard
# par "Env: production" aur "Release: 1.0.0" wale badges aate hain.
os.environ.setdefault("LANGFUSE_TRACING_ENVIRONMENT", LANGFUSE_ENVIRONMENT)
os.environ.setdefault("LANGFUSE_RELEASE", APP_RELEASE)

_trace_warned = set()
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

    Langfuse SDK v4 me `update_current_trace()` hata diya gaya hai (yehi
    wajah thi console warning ki: "'Langfuse' object has no attribute
    'update_current_trace'"). Naye "observations-first" model me tags ab
    trace par nahi, balke observations par lagte hain aur upar trace tak
    khud-ba-khud aggregate ho jate hain -- is liye `propagate_attributes()`
    use karte hain jo current (abhi active) observation par tags laga deta
    hai. Output ke liye `set_current_trace_io()` hai, jo purane
    `update_current_trace(output=...)` ki tarah kaam karta hai (deprecated
    hai lekin abhi bhi supported hai).
    """
    if langfuse is None:
        return
    try:
        if tags:
            with propagate_attributes(tags=[t for t in tags if t]):
                pass
        if output is not None:
            langfuse.set_current_trace_io(output=output)
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