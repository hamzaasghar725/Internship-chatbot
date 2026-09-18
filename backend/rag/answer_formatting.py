"""
Answer polishing and scope enforcement.

Prompt model se professional markdown maangta hai, lekin LLM kabhi kabhi
rules todta hai: "* item" bullets, adhoora "**Heading" (closing ** ghayab),
ya "Certainly!" jaisa filler. Frontend (markdown.js) waise to in sab ko
handle kar leta hai, magar us se pehle yahan text saaf karne ke 3 faide
hain:
  1. Database me bhi saaf jawab save hota hai (history, export, audit).
  2. Copy aur text-to-speech ko saaf text milta hai.
  3. Kal koi doosra client (mobile app, API) bane to usay bhi saaf mile.

Ye sirf formatting theek karta hai -- jawab ka matlab kabhi nahi badalta.
"""
import re

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
