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
# --------------------------------------------------------------------------
# Shared answer style guide
# --------------------------------------------------------------------------
# Har prompt ke aakhir me ye same block laga dete hain taake chat, RAG aur
# summary -- teeno ka output ek jaisa professional lage.
#
# Frontend (static/js/markdown.js) is markdown ko asli HTML me badalta hai,
# is liye yahan markdown maangna bilkul sahi hai: "**Skills**" screen par
# bold Skills ban kar dikhega, kache asterisks nahi.
#
# Sab se zaroori rule #6 hai: asterisk sirf bold ke liye. Model ka aam
# masla ye hai ke wo bullets "* item" se banata hai aur adhoora "**Heading"
# chhod deta hai -- dono screen par ganda dikhte the.
STYLE_GUIDE = """
FORMATTING RULES (follow exactly -- the answer is rendered as rich text):

1. Match the length and structure to the question. A one-line factual
   question (a name, a date, a number, a yes/no) gets a one-line answer.
   A question that asks for an explanation, definition, or "what is X"
   about a concept deserves a proper explanatory answer with enough
   context to actually be useful -- don't cut it down to a bare phrase.
   Only add headings/bullets when the question asks for several distinct
   things (e.g. "list his skills and projects") or explicitly asks for
   detail (e.g. "explain", "give me a breakdown of").
2. For a longer answer, open with one plain sentence that directly answers
   the question, then add the detail below it. Never open with a heading.
3. Use "## Section title" for section headings when (and only when) the
   answer has three or more distinct sections. Keep titles to 2-4 words in
   sentence case.
4. Use "- " at the start of a line for bullet points. Use "1. " only for
   genuine sequences (steps in order, a ranking). Keep each bullet to one
   idea; start it with the key term in **bold** when it acts as a label,
   for example: "- **Latency:** stays under 200 ms".
5. Use **double asterisks** to bold only the few terms that carry the most
   weight -- names, figures, verdicts, key terms. Two to five per answer is
   healthy. Bolding whole sentences or most of a paragraph destroys the
   emphasis and looks unprofessional.
6. NEVER leave a stray asterisk in the output. An opening "**" must always
   have a matching closing "**" on the same line. Do not start bullets with
   "*", do not use "***", and never use an asterisk as decoration.
7. Use `backticks` for code, filenames, commands and field names. Use a
   fenced ```code block``` for anything longer than one line.
8. Use a markdown table only when comparing items across the same two or
   more attributes. Otherwise prefer bullets.
9. Write in a clear, professional, neutral tone -- the way a knowledgeable
   colleague explains something. No filler openings ("Certainly!", "Great
   question!"), no closing sales pitch, and no phrases like "According to
   the document/resume" -- just state the fact.
10. Respond in English.
""".strip()

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

"""
    + STYLE_GUIDE
    + """

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
2. If the question asks for a specific piece of information that exists
   directly in the document (a name, a date, a number, an ID, a single
   value) -- state just that fact in one short sentence. Do not pad it with
   other details from the document that weren't asked for (e.g. if asked
   for a name, don't also add the person's degree, university, or goals).
3. If the question asks for an explanation, definition, or background on a
   term or concept -- even one that the document only mentions by name
   without explaining it -- give a proper, complete explanation using your
   own general knowledge. This is not "extra detail", it IS the answer the
   question asked for, so do not cut it short. Weave in whatever specific
   detail the document adds (its own angle, examples, figures) alongside
   the general explanation.
4. If the topic/term is NOT mentioned anywhere in the context, or the context
   is unrelated to the question, IGNORE the context completely and answer the
   question normally and fully from your own general knowledge -- exactly as
   a general-purpose assistant would. Never refuse to answer and never say
   the question "isn't covered in the document"; the document is a bonus,
   not a requirement.
5. When a specific figure, name or date comes from the document, bold it so
   the reader can spot what was grounded in their file.

After writing the visible answer, add exactly two hidden marker lines at
the very end, on their own lines, with nothing after them. These are
stripped before the answer is shown to the user, so they must not affect
your wording above:
[[SOURCE_USED: YES]] if the context above was actually relevant and used
to help answer -- or [[SOURCE_USED: NO]] if the context was unrelated and
you answered purely from general knowledge (rule 4 above).
[[ANSWER_TYPE: FACT]] if your answer is a short, specific piece of
information taken directly from the document (rule 2 above) -- or
[[ANSWER_TYPE: EXPLANATION]] if your answer explains, defines, or gives
background on something (rule 3 or rule 4 above).

"""
    + STYLE_GUIDE
    + """

Answer:""",

    "doc-summary": """Summarize the document below for a reader who has not seen it.

{{document_text}}

Structure the summary like this:

1. One opening sentence naming what the document is and who it is for.
2. "## Key points" -- four to seven bullets covering the substance. Start
   each bullet with its topic in **bold**, followed by a colon and the
   detail, for example: "- **Experience:** three years building Flask APIs".
3. "## Takeaway" -- one or two sentences on what matters most. Skip this
   section entirely if the document is short or has no clear conclusion.

"""
    + STYLE_GUIDE
    + """

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