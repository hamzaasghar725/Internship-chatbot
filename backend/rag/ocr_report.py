"""
ocr_report.py
=============
Turns an uploaded scanned PDF / image into a single, self-contained HTML
"proof sheet": every page's ORIGINAL scanned image rendered right next to
the text OCR extracted from it, so a human can check side by side that
nothing was missed -- plus a download-ready .html file and honest
extraction stats (pages read, words extracted, [unclear] markers).

This is a SEPARATE OCR pass from the one used for RAG ingestion
(rag/ocr.py's extract_pdf_text / extract_image_text, called from
rag/indexing.py via build_or_update_index). It does not touch that
pipeline, its heuristics, or its behaviour in any way -- this module only
ADDS a new, independent feature.

Why every page is OCR'd here (unlike the ingestion pipeline, which only
OCRs pages that "look scanned" to save API calls): the whole point of this
report is a full page-image + extracted-text pairing for every page, so a
partially-scanned PDF still gets a complete, honest side-by-side sheet.

Important honesty note (read before changing the stats math): there is no
independent ground truth for "how many words were really in the document"
-- OCR is the only way this app has of reading a scanned page in the first
place. So "success rate" here is defined as the percentage of pages that
produced text on this OCR pass (an extraction-completion rate), never as a
proofread accuracy score. Never rename/repurpose this number into an
"accuracy %" claim -- that would be a lie no matter how the code looks.
"""
import base64
import html
import os
import re
import time

from PIL import Image, ImageSequence

from rag.ocr import (
    IMAGE_EXTENSIONS,
    OCR_MAX_PAGES,
    OCRError,
    _prepare_image,
    _render_page,
    _require_api_key,
    _run_ocr_jobs,
)

try:  # PyMuPDF: naye versions "pymupdf" naam se, purane "fitz" se import hote hain
    import pymupdf
except ImportError:  # pragma: no cover
    import fitz as pymupdf


UNCLEAR_RE = re.compile(r"\[unclear\]", re.IGNORECASE)
_TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")
_SEPARATOR_CELL_RE = re.compile(r"^:?-{1,}:?$")


def _word_count(text):
    return len(text.split()) if text else 0


def _markdown_table_to_html(block_lines):
    """block_lines: consecutive '| ... |' lines (may include a '---' separator
    row as the 2nd line, standard Markdown table style). Returns an HTML
    <table> string. Every cell's text is escaped, never dropped or altered."""
    rows = []
    for i, line in enumerate(block_lines):
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if i == 1 and cells and all(_SEPARATOR_CELL_RE.match(c) for c in cells if c):
            continue  # the "---|---|---" separator row -- not real data
        rows.append(cells)
    if not rows:
        return ""
    out = ['<table class="ocr-table">']
    head, body = rows[0], rows[1:]
    out.append("<thead><tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in head) + "</tr></thead>")
    if body:
        out.append("<tbody>")
        for r in body:
            out.append("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in r) + "</tr>")
        out.append("</tbody>")
    out.append("</table>")
    return "".join(out)


def _text_to_html(text):
    """
    Converts OCR text to HTML while preserving every character of it:
      - contiguous Markdown-table-looking lines become a real <table>
      - everything else becomes escaped, whitespace-preserving text (<pre>)
    This only changes how the text is *displayed* -- it never removes or
    rewrites a word, so nothing from the OCR output is ever lost here.
    """
    if not text or not text.strip():
        return '<p class="ocr-empty">(no text found on this page)</p>'

    lines = text.split("\n")
    out = []
    buf = []

    def flush_buf():
        if buf:
            out.append(f'<pre class="ocr-text">{html.escape(chr(10).join(buf))}</pre>')
            buf.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        if _TABLE_LINE_RE.match(line):
            block = []
            while i < len(lines) and _TABLE_LINE_RE.match(lines[i]):
                block.append(lines[i])
                i += 1
            if len(block) >= 2:
                flush_buf()
                out.append(_markdown_table_to_html(block))
            else:
                buf.extend(block)  # a lone '|...|' line isn't really a table
            continue
        buf.append(line)
        i += 1
    flush_buf()
    return "\n".join(out)


def _page_block(index, total, image_b64, mime_type, text, error):
    words = _word_count(text)
    unclear = len(UNCLEAR_RE.findall(text or ""))
    status = "ok" if text and text.strip() else ("error" if error else "empty")
    status_label = {"ok": "OK", "error": "OCR failed", "empty": "No text detected"}[status]
    body_html = (
        f'<p class="ocr-error">OCR error on this page: {html.escape(error)}</p>'
        if error else _text_to_html(text)
    )
    unclear_html = f'<span class="unclear-count">{unclear} [unclear] marker(s)</span>' if unclear else ""
    return f"""
<section class="page-block">
  <div class="page-head">
    <h2>Page {index} of {total}</h2>
    <span class="badge badge-{status}">{status_label}</span>
    <span class="word-count">{words} words extracted</span>
    {unclear_html}
  </div>
  <div class="page-columns">
    <div class="page-image-col">
      <img src="data:{mime_type};base64,{image_b64}" alt="Original scan of page {index}">
    </div>
    <div class="page-text-col">
      {body_html}
    </div>
  </div>
</section>
"""


_HTML_SHELL = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OCR export -- {title}</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 0; padding: 0 0 60px; background: #f4f5f7; color: #1a1a1a; }}
  header.report-head {{ background: #14181f; color: #fff; padding: 28px 32px; }}
  header.report-head h1 {{ margin: 0 0 6px; font-size: 22px; }}
  header.report-head p {{ margin: 4px 0; color: #c7cdd6; font-size: 14px; }}
  .stats-grid {{ display: flex; flex-wrap: wrap; gap: 14px; margin-top: 16px; }}
  .stat-card {{ background: #1e2430; border-radius: 10px; padding: 12px 16px; min-width: 150px; }}
  .stat-card .num {{ font-size: 20px; font-weight: 700; }}
  .stat-card .label {{ font-size: 12px; color: #9aa4b2; }}
  .note {{ max-width: 900px; margin: 18px auto 0; padding: 12px 16px; background: #fff8e1; border: 1px solid #f0c14b; border-radius: 8px; font-size: 13px; color: #4a3b00; }}
  main {{ max-width: 1200px; margin: 24px auto; padding: 0 20px; }}
  .page-block {{ background: #fff; border: 1px solid #e2e5ea; border-radius: 12px; margin-bottom: 22px; overflow: hidden; }}
  .page-head {{ display: flex; align-items: center; gap: 12px; padding: 12px 18px; border-bottom: 1px solid #eceff3; background: #fafbfc; flex-wrap: wrap; }}
  .page-head h2 {{ font-size: 15px; margin: 0; }}
  .badge {{ font-size: 11px; font-weight: 700; padding: 2px 9px; border-radius: 999px; text-transform: uppercase; letter-spacing: .03em; }}
  .badge-ok {{ background: #dcf5e3; color: #146c2e; }}
  .badge-error {{ background: #fde2e1; color: #9c1c14; }}
  .badge-empty {{ background: #eee; color: #666; }}
  .word-count, .unclear-count {{ font-size: 12px; color: #667; }}
  .unclear-count {{ color: #a15c00; }}
  .page-columns {{ display: flex; flex-wrap: wrap; }}
  .page-image-col, .page-text-col {{ flex: 1 1 480px; padding: 16px; min-width: 320px; }}
  .page-image-col {{ border-right: 1px solid #eceff3; background: #fcfcfd; text-align: center; }}
  .page-image-col img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 6px; }}
  .ocr-text {{ white-space: pre-wrap; word-wrap: break-word; font-family: "Courier New", monospace; font-size: 13.5px; line-height: 1.55; margin: 0 0 10px; }}
  .ocr-empty {{ color: #888; font-style: italic; }}
  .ocr-error {{ color: #9c1c14; }}
  table.ocr-table {{ border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 13px; }}
  table.ocr-table th, table.ocr-table td {{ border: 1px solid #ccc; padding: 5px 8px; text-align: left; }}
  table.ocr-table th {{ background: #f0f2f5; }}
  footer {{ text-align: center; color: #999; font-size: 12px; margin-top: 30px; }}
  @media (prefers-color-scheme: dark) {{
    body {{ background: #14161a; color: #e7e9ec; }}
    .page-block {{ background: #1c1f24; border-color: #2a2e35; }}
    .page-head {{ background: #20242b; border-color: #2a2e35; }}
    .page-image-col {{ background: #191c21; border-color: #2a2e35; }}
    table.ocr-table th {{ background: #262b33; }}
    table.ocr-table th, table.ocr-table td {{ border-color: #383e47; }}
    .note {{ background: #2c260f; border-color: #7a5f00; color: #f0d98c; }}
  }}
</style>
</head>
<body>
<header class="report-head">
  <h1>OCR Export -- {title}</h1>
  <p>Generated by Internship Chatbot &middot; {generated_at}</p>
  <div class="stats-grid">
    <div class="stat-card"><div class="num">{total_pages}</div><div class="label">Pages processed</div></div>
    <div class="stat-card"><div class="num">{ok_pages}/{total_pages}</div><div class="label">Pages OCR'd successfully</div></div>
    <div class="stat-card"><div class="num">{success_pct}%</div><div class="label">Pages successfully read</div></div>
    <div class="stat-card"><div class="num">{total_words}</div><div class="label">Words extracted</div></div>
    <div class="stat-card"><div class="num">{unclear_total}</div><div class="label">[unclear] markers</div></div>
  </div>
</header>
<div class="note">
  <strong>How to read this:</strong> each page below shows the <em>original scanned image</em> next to
  the text our OCR extracted from it &mdash; compare them side by side to confirm nothing was missed.
  The "pages successfully read" percentage reflects how many pages produced text on this pass; it is
  <strong>not</strong> an independent proofreading/accuracy score, since OCR is the only way this tool
  has of reading a scanned page in the first place. Any word the model could not read clearly is
  marked <code>[unclear]</code> above.
</div>
<main>
{pages_html}
</main>
<footer>End of report &middot; {total_pages} page(s) &middot; {filename}</footer>
</body>
</html>
"""


def build_ocr_html_report(file_path, filename):
    """
    file_path: path to the already-uploaded file on disk.
    filename: original filename (used for display + suggested download name).
    Returns (html_string, stats_dict). Raises OCRError for anything the
    caller should show as a clean, user-facing message (missing API key,
    unsupported file type, unreadable file, total OCR failure).
    """
    ext = os.path.splitext(filename)[1].lower().lstrip(".")
    page_jobs = []  # [(label, mime_type, data), ...] in page order
    skipped = 0

    if ext == "pdf":
        doc = None
        try:
            doc = pymupdf.open(file_path)
            if getattr(doc, "needs_pass", False):
                doc.authenticate("")
            page_count = len(doc)
            skipped = max(0, page_count - OCR_MAX_PAGES)
            for i in range(min(page_count, OCR_MAX_PAGES)):
                mime_type, data = _render_page(doc[i])
                page_jobs.append((str(i + 1), mime_type, data))
        except Exception as e:
            raise OCRError(f"Could not open this PDF ({e}).")
        finally:
            if doc is not None:
                doc.close()
    elif ext in IMAGE_EXTENSIONS:
        try:
            img = Image.open(file_path)
            if ext in ("tif", "tiff"):
                frames = [f.copy() for _, f in zip(range(OCR_MAX_PAGES), ImageSequence.Iterator(img))]
            else:
                frames = [img]
            for n, frame in enumerate(frames, start=1):
                mime_type, data = _prepare_image(frame)
                page_jobs.append((str(n), mime_type, data))
        except Exception as e:
            raise OCRError(f"Could not open this image ({e}).")
    else:
        raise OCRError(
            "HTML export (with page images) is only available for PDFs and image files, "
            "since it works by OCR-ing each page. Upload a PDF or an image to use it."
        )

    if not page_jobs:
        raise OCRError("This file has no pages to export.")

    _require_api_key()
    results, errors = _run_ocr_jobs(page_jobs)

    total_pages = len(page_jobs)
    ok_pages = sum(1 for label, _, _ in page_jobs if (results.get(label) or "").strip())
    total_words = sum(_word_count(results.get(label, "")) for label, _, _ in page_jobs)
    unclear_total = sum(len(UNCLEAR_RE.findall(results.get(label) or "")) for label, _, _ in page_jobs)
    success_pct = round(100 * ok_pages / total_pages) if total_pages else 0

    if errors and not results:
        raise OCRError("OCR failed for every page (the AI model may be busy or your quota is used up). Please try again in a moment.")

    pages_html = []
    for idx, (label, mime_type, data) in enumerate(page_jobs, start=1):
        text = results.get(label, "")
        error = errors.get(label)
        image_b64 = base64.b64encode(data).decode("ascii")
        pages_html.append(_page_block(idx, total_pages, image_b64, mime_type, text, error))

    html_out = _HTML_SHELL.format(
        title=html.escape(filename),
        generated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        total_pages=total_pages,
        ok_pages=ok_pages,
        success_pct=success_pct,
        total_words=total_words,
        unclear_total=unclear_total,
        pages_html="\n".join(pages_html),
        filename=html.escape(filename),
    )

    stats = {
        "filename": filename,
        "total_pages": total_pages,
        "ok_pages": ok_pages,
        "failed_pages": len(errors),
        "success_pct": success_pct,
        "total_words": total_words,
        "unclear_markers": unclear_total,
        "skipped_pages": skipped,
    }
    return html_out, stats