"""Top-level RAG pipeline: question answering and document summarization."""
import os
import pickle
import re

from rag.answer_formatting import enforce_scope, polish_answer, _split_markers
from rag.embeddings import EMBED_MODEL_NAME, embed_query
from rag.gemini_client import _safe_call_gemini
from rag.indexing import (
    _paths_for_user,
    _user_has_index,
    load_full_document_text,
    retrieve_relevant_chunks,
)
from rag.prompt_registry import get_prompt
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


# "Poora document / har lafz / word by word / as-is dedo" jaisi requests --
# English aur Roman Urdu dono. In par RAG (top-k chunks + LLM) bypass ho kar
# poora saved text seedha laut jata hai, taake na kuch chhoote aur na model
# apni taraf se paraphrase/summarize kare.
_FULL_DOCUMENT_RE = re.compile(
    r"\b("
    r"(entire|whole|full|complete)\s+(document|text|content|data|doc)"
    r"|word[\s-]?(for|by)[\s-]?word"
    r"|single\s+single\s+word"
    r"|verbatim"
    r"|as[\s-]is"
    r"|jaisa\s+hai\s+waisa"
    r"|(poora|pura|sara|sari|puri)\s+(document|text|data)"
    r"|har\s+(lafz|word)"
    r"|lafz\s+lafz"
    r")\b",
    re.IGNORECASE,
)

# Agar query kisi KHAAS cheez ke baare me poochti hai (headings, titles,
# ek list, summary...) to "poori document" jaise alfaz sirf is context ke
# hisse hote hain -- poora raw text dump nahi chahiye hota, sirf wo khaas
# cheez chahiye hoti hai. Aisi requests ko full-document bypass se bahar
# rakhte hain taake "poori document ki headings do" ulta poora text na de de.
_SUBSET_ASK_RE = re.compile(
    r"\b("
    r"headings?|titles?|sections?|subheadings?"
    r"|table\s+of\s+contents|\btoc\b"
    r"|summary|summarize|summarise"
    r"|keywords?"
    r"|list\s+of|outline"
    r"|sirf|only|just"
    r")\b",
    re.IGNORECASE,
)


def _wants_full_document(query):
    query = query or ""
    if not _FULL_DOCUMENT_RE.search(query):
        return False
    return not _SUBSET_ASK_RE.search(query)


# "Headings/titles/sections/outline do" -- top-k retrieval (sirf 4-6 chunks)
# document ke chunks ko RANDOMLY thori si jagah se uthata hai, is liye 15 me
# se sirf 3-4 headings milti hain (baaki jin chunks me thin wo retrieve hi
# nahi hote). Ye sawal poore document ka structure maangte hain, kisi ek
# jagah ka fact nahi -- is liye inke liye top-k bypass kar ke POORA document
# context me diya jata hai, taake koi heading na chhoote.
_STRUCTURE_ASK_RE = re.compile(
    r"\b(headings?|titles?|sections?|subheadings?|table\s+of\s+contents|\btoc\b|outline)\b",
    re.IGNORECASE,
)


def _wants_document_structure(query):
    return bool(_STRUCTURE_ASK_RE.search(query or ""))


# "Exact text in Product Roadmap", "verbatim text of Finance and Budget"...
# Numbered headings ("8. Product Roadmap") ko dhoondh kar us heading se agli
# heading TAK ka asal text (jaisa document me hai) seedha wapas karte hain --
# model se dobara likhwate nahi, taake wo apni taraf se naye sub-headings ya
# bullets na bana de.
_EXACT_CUE_RE = re.compile(
    r"\b("
    r"exact|verbatim|raw"
    r"|word[\s-]?for[\s-]?word|word[\s-]?by[\s-]?word"
    r"|as[\s-]is|jaisa\s+hai\s+waisa"
    r")\b",
    re.IGNORECASE,
)

# Ek poori line jo sirf "8. Product Roadmap" jaisi ho (number + short title,
# koi aur punctuation nahi) -- document ke numbered section headings.
_HEADING_LINE_RE = re.compile(r"^[ \t]*\d{1,2}\.\s+([A-Z][A-Za-z][A-Za-z &/\-]{1,60})[ \t]*$", re.MULTILINE)


def _extract_named_section(query, full_text):
    """
    Query me kisi heading ka naam (jaise "Product Roadmap") mile aur wo
    document ke numbered headings me se kisi se match ho, to us heading se
    lekar AGLI heading tak ka asal (raw) text return karta hai.
    Match na ho to None.
    """
    headings = list(_HEADING_LINE_RE.finditer(full_text or ""))
    if not headings:
        return None
    query_lower = (query or "").lower()
    for i, m in enumerate(headings):
        title = m.group(1).strip()
        if title.lower() in query_lower:
            start = m.end()
            end = headings[i + 1].start() if i + 1 < len(headings) else len(full_text)
            section_text = full_text[start:end].strip()
            return title, section_text
    return None


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
            has_document = _user_has_index(user_id)
            _emit_event(langfuse, "chat-request", metadata={
                "queryLength": len(query),
                "queryWords": len(query.split()),
                "topK": top_k,
                "hasDocument": has_document,
            })

            # ---- 0a. "Exact text of <section heading>" -> that section, raw ----
            if has_document and _EXACT_CUE_RE.search(query):
                full_text, filename = load_full_document_text(user_id)
                found = _extract_named_section(query, full_text) if full_text else None
                if found:
                    title, section_text = found
                    if section_text:
                        _emit_event(langfuse, "mode-selected", metadata={
                            "mode": "exact-section",
                            "reason": f"query asked for the exact/verbatim text of section '{title}'",
                        })
                        answer = section_text
                        _emit_event(langfuse, "chat-completed", metadata={
                            "answerLength": len(answer),
                            "sourceCount": 1,
                            "sources": [filename] if filename else [],
                            "contextUsed": True,
                        })
                        _tag_trace(
                            langfuse,
                            tags=[TRACE_APP_TAG, "chat", "exact-section", "context-used"],
                            output={"answerLength": len(answer), "sources": [filename] if filename else []},
                        )
                        _update(root_span, output={
                            "answer": answer, "sources": [filename] if filename else [],
                            "answerLength": len(answer), "sourceUsed": True,
                        })
                        _flush(langfuse)
                        return answer, ([filename] if filename else [])

            # ---- 0b. "Give me the whole document" -> skip RAG entirely ----
            # Top-k retrieval sirf 4 chunks (~2000 chars) deta hai, is liye
            # "poora document do" jaisi requests ka jawab hamesha adhoora ya
            # LLM ka paraphrase hota -- yahan seedha poora saved text lautate
            # hain, bilkul jaisa document se nikla tha.
            if has_document and _wants_full_document(query):
                full_text, filename = load_full_document_text(user_id)
                if full_text and full_text.strip():
                    _emit_event(langfuse, "mode-selected", metadata={
                        "mode": "full-document",
                        "reason": "query asked for the entire document verbatim",
                    })
                    answer = full_text.strip()
                    _emit_event(langfuse, "chat-completed", metadata={
                        "answerLength": len(answer),
                        "sourceCount": 1,
                        "sources": [filename] if filename else [],
                        "contextUsed": True,
                    })
                    _tag_trace(
                        langfuse,
                        tags=[TRACE_APP_TAG, "chat", "full-document", "context-used"],
                        output={"answerLength": len(answer), "sources": [filename] if filename else []},
                    )
                    _update(root_span, output={
                        "answer": answer, "sources": [filename] if filename else [],
                        "answerLength": len(answer), "sourceUsed": True,
                    })
                    _flush(langfuse)
                    return answer, ([filename] if filename else [])

            # ---- 1/2. Query embedding + retrieval ----
            # "headings/sections/outline do" jaise sawal document ka poora
            # structure maangte hain -- top-k ki jagah poora document ek
            # context ki tarah use hota hai, taake koi heading na chhoote.
            if has_document and _wants_document_structure(query):
                full_text, filename = load_full_document_text(user_id)
                if full_text and full_text.strip():
                    chunks = [{"source": filename, "text": full_text, "distance": 0.0}]
                    retrieved_sources = [filename] if filename else []
                    _emit_event(langfuse, "retrieval-override", metadata={
                        "reason": "structure question -- using full document instead of top-k chunks",
                        "characters": len(full_text),
                    })
                else:
                    chunks, retrieved_sources = [], []
            else:
                with _observe(langfuse, "embedding", as_type="span",
                              input={"model": EMBED_MODEL_NAME, "query": query}) as span:
                    query_vec = embed_query(query)
                    _update(span, output={"dimensions": int(query_vec.shape[1])})

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