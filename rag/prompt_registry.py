"""
prompt_registry.py
===================
Thin wrapper around Langfuse Prompt Management.

Why this file exists:
  - `langfuse.get_prompt()` is a network call. If Langfuse is down or keys
    aren't set, the chatbot must not crash -- so every prompt has a local
    FALLBACK here that is byte-for-byte the same text that used to be
    hardcoded in rag_utils.py.
  - The returned `prompt` object needs to be linked to the generation
    (`prompt=` kwarg on the Langfuse observation), otherwise the Langfuse UI's
    "Metrics" tab won't show per-version latency/cost/tokens.

Prompts are managed as Langfuse "Text" prompts (not "Chat"), because
_call_gemini() sends one single string to the Gemini REST API -- there's no
separate system/user split in this project, so Text prompts are the exact
match for how the code actually works.

Place this file at chatbot_faceproject/rag/prompt_registry.py.
"""

import os

_langfuse_client = None


def _get_langfuse_client():
    """Same pattern as rag_utils.py: no keys -> None, app still works."""
    global _langfuse_client
    if not os.environ.get("LANGFUSE_PUBLIC_KEY"):
        return None
    if _langfuse_client is None:
        from langfuse import get_client
        _langfuse_client = get_client()
    return _langfuse_client


# --------------------------------------------------------------------------
# Fallbacks: exactly the text that used to be hardcoded in rag_utils.py.
# Only used if Langfuse is unreachable, keys are missing, or the prompt
# hasn't been created in Langfuse yet -- so nothing ever breaks because of
# Langfuse being unavailable.
# --------------------------------------------------------------------------
FALLBACKS = {
    # Used when no document has been uploaded yet, or the question has no
    # relevant chunks at all (retrieve_relevant_chunks returned nothing).
    # This is what makes the bot a general-purpose assistant by default,
    # with RAG as a bonus feature rather than a requirement to get an answer.
    "general-chat": """You are a helpful, knowledgeable general-purpose AI assistant (like a
general chatbot). No document context is available for this question --
either the user hasn't uploaded a document yet, or this question doesn't
relate to one.

Question: {{question}}

Rules for answering:
1. Answer fully, accurately, and helpfully using your own knowledge, exactly
   as a general-purpose assistant would.
2. Never refuse to answer and never mention documents, uploads, or missing
   context -- just answer the question directly.
3. Respond in English.

Answer:""",

    "rag-answer": """You are a helpful, knowledgeable general-purpose AI assistant. The user has
uploaded a document, and some possibly-relevant excerpts from it are
included below as extra context you can draw on.

Context retrieved from the document(s):
{{context}}

Question: {{question}}

Rules for answering:
1. First check whether the topic/term the question is about is actually mentioned
   in the context above (even just by name, without full detail).
2. If it IS mentioned (even briefly) -- give a complete, proper, correct answer
   to the question using your own general knowledge, not just what little the
   document says. The document only needs to establish that the topic is
   relevant; it doesn't need to contain the full explanation. Prefer to also
   weave in whatever the document itself adds (extra detail, the document's
   specific angle, examples, figures) alongside the general explanation.
3. If the topic/term is NOT mentioned anywhere in the context, or the context
   is unrelated to the question, IGNORE the context completely and answer the
   question normally and fully from your own general knowledge -- exactly as
   a general-purpose assistant would. Never refuse to answer and never say
   the question "isn't covered in the document"; the document is a bonus,
   not a requirement.
4. Respond in English.

Answer:""",

    "doc-summary": """Write a concise summary (as bullet points) of the document below.
Respond in English.

{{document_text}}

Summary:""",
}


def _render_fallback(name, variables):
    text = FALLBACKS[name]
    for key, value in variables.items():
        text = text.replace("{{" + key + "}}", str(value))
    return text, None


def get_prompt(name, label="production", **variables):
    """
    Fetches a Langfuse "text" prompt and compiles it with `variables`.

    Returns: (compiled_prompt_text, prompt_object)

    `prompt_object` is the Langfuse prompt to pass into the generation
    observation's `prompt=` kwarg for UI linking (which version produced
    which output). It is None when the local fallback was used instead.
    """
    langfuse = _get_langfuse_client()
    if langfuse is None:
        return _render_fallback(name, variables)

    try:
        prompt_obj = langfuse.get_prompt(
            name,
            type="text",
            label=label,
            cache_ttl_seconds=300,  # 5 min client-side cache, refreshed in background
        )
    except Exception as e:
        print(f"[Langfuse] couldn't fetch prompt '{name}' ({e}) -- using local fallback.")
        return _render_fallback(name, variables)

    return prompt_obj.compile(**variables), prompt_obj