/* =========================================================================
   markdown.js  --  Answer Formatting Engine
   =========================================================================
   Kyun banayi gayi:
   Pehle chat.js me bot ka jawab `div.textContent = text` se lagta tha.
   textContent har cheez ko plain text samajhta hai, is liye model ka bheja
   hua "**Skills**" screen par literally asterisks ke saath dikhta tha.

   Ye file us markdown ko asli HTML me badalti hai (bold, headings, lists,
   tables, code) -- bina kisi external library ke, taake project offline
   bhi chale aur koi nayi dependency add na ho.

   SECURITY (sab se ahem):
   Sab se pehle saara input HTML-escape hota hai. Uske BAAD hi humare apne
   safe tags lagte hain. Matlab agar model (ya kisi uploaded document) me
   <script> jaisa kuch aa bhi jaye, wo text ban kar dikhega -- chalega nahi.
   Links me sirf http / https / mailto allow hain, "javascript:" block hai.

   Public API (window.MarkdownRenderer):
       .toHtml(markdownText)   -> HTML string
       .toElement(markdownText)-> ready <div class="msg-content">
       .toPlainText(markdownText) -> saaf text (Copy / Listen ke liye)
   ========================================================================= */

(function (global) {
    "use strict";

    /* ---------------------------------------------------------------
       1. Escaping  --  har cheez se pehle
       --------------------------------------------------------------- */
    function escapeHtml(str) {
        return String(str == null ? "" : str)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#39;");
    }

    // Sirf in schemes ko link banne ki ijazat hai.
    function safeUrl(url) {
        var clean = String(url || "").trim();
        if (/^(https?:\/\/|mailto:|\/|#)/i.test(clean)) return clean;
        return null; // javascript:, data:, vbscript: -- sab reject
    }

    /* ---------------------------------------------------------------
       2. Inline formatting  --  bold, italic, code, links
       --------------------------------------------------------------- */
    function renderInline(text) {
        var html = escapeHtml(text);

        // `inline code` ko placeholder se bachate hain, warna uske andar ke
        // asterisks bhi bold ban jayenge (e.g. `a * b` galat render hota).
        var codeStore = [];
        html = html.replace(/`([^`\n]+)`/g, function (_, code) {
            codeStore.push(code);
            return "\u0000CODE" + (codeStore.length - 1) + "\u0000";
        });

        // [text](url) -- url validate hone par hi <a> banta hai.
        // URL pattern ek level ke nested brackets bhi sambhalta hai, warna
        // "alert(1)" jaise URLs par aakhri ")" bahar reh kar screen par
        // dikhne lagta tha.
        html = html.replace(
            /\[([^\]\n]+)\]\(\s*([^()\s]*(?:\([^()\s]*\)[^()\s]*)*)\s*\)/g,
            function (whole, label, url) {
                var href = safeUrl(url.replace(/&amp;/g, "&"));
                if (!href) return label; // javascript: wagaira -- sirf label rehta hai
                return '<a href="' + escapeHtml(href) + '" target="_blank" rel="noopener noreferrer">' + label + "</a>";
            }
        );

        // Bold sab se pehle (**text** / __text__), phir italic --
        // ulta karne se "**x**" ka andar wala hissa italic ban jata hai.
        html = html.replace(/\*\*([^\s*][\s\S]*?[^\s*]|[^\s*])\*\*/g, "<strong>$1</strong>");
        html = html.replace(/__([^\s_][\s\S]*?[^\s_]|[^\s_])__/g, "<strong>$1</strong>");
        html = html.replace(/(^|[^\w*])\*([^\s*][^*\n]*?)\*(?![\w*])/g, "$1<em>$2</em>");
        html = html.replace(/(^|[^\w_])_([^\s_][^_\n]*?)_(?![\w_])/g, "$1<em>$2</em>");
        html = html.replace(/~~([\s\S]+?)~~/g, "<del>$1</del>");

        // -- Stray asterisk cleanup --
        // Model kabhi kabhi "**Heading" likh kar closing "**" bhool jata hai.
        // Aise bache hue markers hata dete hain taake screen par kabhi bhi
        // kacha asterisk na dikhe. Note: " * " (dono taraf space) chhoot
        // jata hai kyunke wo multiplication ho sakta hai, formatting nahi.
        html = html.replace(/\*{2,}/g, "");
        html = html.replace(/(\S)\*/g, "$1").replace(/\*(\S)/g, "$1");

        // code placeholders wapas
        html = html.replace(/\u0000CODE(\d+)\u0000/g, function (_, i) {
            return "<code>" + escapeHtml(codeStore[Number(i)]) + "</code>";
        });

        return html;
    }

    /* ---------------------------------------------------------------
       3. Line classifiers
       --------------------------------------------------------------- */
    var RE_FENCE = /^\s{0,3}```(.*)$/;
    var RE_HEADING = /^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$/;
    var RE_HR = /^\s{0,3}([-*_])\s*(?:\1\s*){2,}$/;
    var RE_QUOTE = /^\s{0,3}>\s?(.*)$/;
    var RE_UL = /^(\s*)[-*+]\s+(.*)$/;
    var RE_OL = /^(\s*)(\d+)[.)]\s+(.*)$/;
    var RE_TABLE_SEP = /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$/;

    function isListLine(line) {
        return RE_UL.test(line) || RE_OL.test(line);
    }

    function indentWidth(spaces) {
        // Tab ko 4 spaces ginte hain taake nesting sahi detect ho.
        return spaces.replace(/\t/g, "    ").length;
    }

    function splitRow(line) {
        var row = line.trim().replace(/^\|/, "").replace(/\|$/, "");
        return row.split("|").map(function (c) { return c.trim(); });
    }

    /* ---------------------------------------------------------------
       4. Block parser  --  markdown -> HTML
       --------------------------------------------------------------- */
    function toHtml(src) {
        var lines = String(src == null ? "" : src).replace(/\r\n?/g, "\n").split("\n");
        var out = [];
        var i = 0;

        while (i < lines.length) {
            var line = lines[i];

            // ---- khaali line ----
            if (!line.trim()) { i++; continue; }

            // ---- fenced code block ```lang ... ``` ----
            var fence = line.match(RE_FENCE);
            if (fence) {
                var lang = (fence[1] || "").trim().split(/\s+/)[0];
                var buf = [];
                i++;
                while (i < lines.length && !RE_FENCE.test(lines[i])) {
                    buf.push(lines[i]);
                    i++;
                }
                i++; // closing fence
                out.push(
                    '<pre class="md-code"' + (lang ? ' data-lang="' + escapeHtml(lang) + '"' : "") +
                    "><code>" + escapeHtml(buf.join("\n")) + "</code></pre>"
                );
                continue;
            }

            // ---- heading ----
            var head = line.match(RE_HEADING);
            if (head) {
                // h1/h2 ko h3/h4 par map karte hain -- chat bubble ke andar
                // page-title size ka heading bhadda lagta hai.
                var level = Math.min(head[1].length + 2, 6);
                out.push("<h" + level + ">" + renderInline(head[2]) + "</h" + level + ">");
                i++;
                continue;
            }

            // ---- horizontal rule ----
            if (RE_HR.test(line)) {
                out.push("<hr>");
                i++;
                continue;
            }

            // ---- table ----
            if (line.indexOf("|") !== -1 && i + 1 < lines.length && RE_TABLE_SEP.test(lines[i + 1])) {
                var headCells = splitRow(line);
                var aligns = splitRow(lines[i + 1]).map(function (c) {
                    if (/^:-+:$/.test(c)) return "center";
                    if (/-+:$/.test(c)) return "right";
                    return "left";
                });
                i += 2;

                var body = [];
                while (i < lines.length && lines[i].indexOf("|") !== -1 && lines[i].trim()) {
                    body.push(splitRow(lines[i]));
                    i++;
                }

                var t = ['<div class="md-table-wrap"><table class="md-table"><thead><tr>'];
                headCells.forEach(function (c, idx) {
                    t.push('<th style="text-align:' + (aligns[idx] || "left") + '">' + renderInline(c) + "</th>");
                });
                t.push("</tr></thead><tbody>");
                body.forEach(function (row) {
                    t.push("<tr>");
                    headCells.forEach(function (_, idx) {
                        t.push('<td style="text-align:' + (aligns[idx] || "left") + '">' + renderInline(row[idx] || "") + "</td>");
                    });
                    t.push("</tr>");
                });
                t.push("</tbody></table></div>");
                out.push(t.join(""));
                continue;
            }

            // ---- blockquote ----
            if (RE_QUOTE.test(line)) {
                var qbuf = [];
                while (i < lines.length && RE_QUOTE.test(lines[i])) {
                    qbuf.push(lines[i].match(RE_QUOTE)[1]);
                    i++;
                }
                out.push("<blockquote>" + toHtml(qbuf.join("\n")) + "</blockquote>");
                continue;
            }

            // ---- list (nested support) ----
            if (isListLine(line)) {
                var consumed = renderList(lines, i, out);
                i = consumed;
                continue;
            }

            // ---- paragraph ----
            var pbuf = [];
            while (
                i < lines.length && lines[i].trim() &&
                !RE_FENCE.test(lines[i]) && !RE_HEADING.test(lines[i]) &&
                !RE_HR.test(lines[i]) && !RE_QUOTE.test(lines[i]) &&
                !isListLine(lines[i])
            ) {
                pbuf.push(lines[i].trim());
                i++;
            }
            if (pbuf.length) {
                out.push("<p>" + renderInline(pbuf.join(" ")) + "</p>");
            }
        }

        return out.join("");
    }

    /* Ek list block ko (nesting ke saath) render karta hai.
       Return: agli line ka index. */
    function renderList(lines, start, out) {
        var first = lines[start];
        var firstMatch = first.match(RE_UL) || first.match(RE_OL);
        var baseIndent = indentWidth(firstMatch[1]);
        var ordered = RE_OL.test(first);
        var items = [];
        var i = start;

        while (i < lines.length) {
            var line = lines[i];

            if (!line.trim()) {
                // Khaali line ke baad agar list continue nahi ho rahi to band.
                var next = lines[i + 1];
                if (!next || !isListLine(next) || indentWidth((next.match(RE_UL) || next.match(RE_OL))[1]) < baseIndent) break;
                i++;
                continue;
            }

            var m = line.match(RE_UL) || line.match(RE_OL);
            if (!m) {
                // List item ki continuation line (indented paragraph text)
                if (items.length && indentWidth(line.match(/^(\s*)/)[1]) > baseIndent) {
                    items[items.length - 1].text += " " + line.trim();
                    i++;
                    continue;
                }
                break;
            }

            var indent = indentWidth(m[1]);
            if (indent < baseIndent) break;

            if (indent > baseIndent) {
                // Nested list -- recursive
                var sub = [];
                i = renderList(lines, i, sub);
                if (items.length) items[items.length - 1].children += sub.join("");
                continue;
            }

            var isOrderedLine = RE_OL.test(line);
            if (isOrderedLine !== ordered && items.length) break; // list ki type badal gayi

            items.push({ text: isOrderedLine ? m[3] : m[2], children: "" });
            i++;
        }

        var tag = ordered ? "ol" : "ul";
        var html = ["<" + tag + ' class="md-list">'];
        items.forEach(function (it) {
            html.push("<li>" + renderInline(it.text) + it.children + "</li>");
        });
        html.push("</" + tag + ">");
        out.push(html.join(""));

        return i;
    }

    /* ---------------------------------------------------------------
       5. Plain text  --  Copy button aur Text-to-Speech ke liye
       --------------------------------------------------------------- */
    function toPlainText(src) {
        var text = String(src == null ? "" : src).replace(/\r\n?/g, "\n");

        // Code ko placeholders me mehfooz kar lete hain. Warna neeche wala
        // asterisk-cleanup code ke andar ghus kar "a * b" ko "a  b" bana
        // deta hai -- copy kiya hua code toot jata.
        var store = [];
        function keep(code) {
            store.push(code);
            return "\u0000KEEP" + (store.length - 1) + "\u0000";
        }
        text = text.replace(/```[\w-]*\n?([\s\S]*?)```/g, function (_, code) {
            return keep(code.replace(/\n$/, ""));
        });
        text = text.replace(/`([^`\n]+)`/g, function (_, code) { return keep(code); });

        // Tables: separator row (|---|---|) hata kar cells ko " | " se jorte
        // hain, taake copy karne par columns parhne layak rahen.
        text = text.replace(/^\s*\|?\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)+\|?\s*$/gm, "\u0000DROP\u0000");
        text = text.replace(/^\s*\|(.+)\|\s*$/gm, function (_, row) {
            return row.split("|").map(function (c) { return c.trim(); }).join(" | ");
        });
        text = text.replace(/^\u0000DROP\u0000\n?/gm, "");

        text = text
            .replace(/!\[([^\]]*)\]\(\s*[^()\s]*(?:\([^()\s]*\)[^()\s]*)*\s*\)/g, "$1")  // images -> alt text
            .replace(/\[([^\]]+)\]\(\s*[^()\s]*(?:\([^()\s]*\)[^()\s]*)*\s*\)/g, "$1")   // links -> sirf label
            .replace(/^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$/gm, "\n$1\n")  // heading dono taraf se alag
            .replace(/^\s{0,3}>\s?/gm, "")                      // quote markers
            .replace(/^\s{0,3}([-*_])\s*(?:\1\s*){2,}$/gm, "")  // horizontal rule
            .replace(/^(\s*)[-*+]\s+/gm, "$1\u2022 ")           // bullets -> •
            .replace(/\*{1,3}/g, "")                            // bold / italic markers
            .replace(/(\w)__(\w)/g, "$1_$2")                    // snake_case bachao
            .replace(/__/g, "")
            .replace(/~~/g, "")
            .replace(/[ \t]+$/gm, "")
            .replace(/\n{3,}/g, "\n\n")
            .trim();

        return text.replace(/\u0000KEEP(\d+)\u0000/g, function (_, i) { return store[Number(i)]; });
    }

    /* ---------------------------------------------------------------
       6. Ready-to-append element
       --------------------------------------------------------------- */
    function toElement(src) {
        var wrap = document.createElement("div");
        wrap.className = "msg-content";
        wrap.innerHTML = toHtml(src);
        return wrap;
    }

    global.MarkdownRenderer = {
        toHtml: toHtml,
        toElement: toElement,
        toPlainText: toPlainText,
        escapeHtml: escapeHtml
    };
})(window);