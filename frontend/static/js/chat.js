const chatBox = document.getElementById("chatBox");
const chatScroll = document.querySelector(".chat-scroll"); // this is the element that actually scrolls (see style.css)
const queryInput = document.getElementById("queryInput");
const sendBtn = document.getElementById("sendBtn");
const uploadBtn = document.getElementById("uploadBtn");
const summarizeBtn = document.getElementById("summarizeBtn");
const documentInput = document.getElementById("documentInput");
const uploadStatus = document.getElementById("uploadStatus");
const micBtn = document.getElementById("micBtn");

let lastUploadedFilename = null;

// ---- Conversation (chat session) tracking ----
// Each browser keeps the id of the conversation it's currently viewing so that
// /ask and /history know which conversation a message belongs to / should load.
let currentSessionId = localStorage.getItem("chatSessionId") || null;

function setSessionId(id) {
    currentSessionId = id;
    if (id) {
        localStorage.setItem("chatSessionId", id);
    } else {
        localStorage.removeItem("chatSessionId");
    }
}

function formatMetrics(metrics) {
    // metrics = { model, input_tokens, output_tokens, total_tokens, latency_ms, cost_usd }
    if (!metrics) return null;
    const parts = [];
    if (metrics.model) parts.push(metrics.model);
    if (metrics.total_tokens != null) {
        parts.push(`${metrics.total_tokens} tokens (${metrics.input_tokens} in / ${metrics.output_tokens} out)`);
    }
    if (metrics.latency_ms != null) {
        parts.push(`${(metrics.latency_ms / 1000).toFixed(2)}s`);
    }
    if (metrics.cost_usd != null) {
        parts.push(`$${metrics.cost_usd.toFixed(6)}`);
    } else {
        parts.push("cost: n/a");
    }
    return parts.join(" \u00b7 ");
}

function addMessage(text, sender, sources = [], metrics = null) {
    const div = document.createElement("div");
    div.className = `msg ${sender}`;

    // ---- Answer formatting ----
    // Bot ke jawab markdown me aate hain (**bold**, ## heading, bullets,
    // tables). Pehle yahan `div.textContent = text` tha, jis ki wajah se
    // markdown render hone ke bajaye screen par kache asterisks dikhte the.
    // Ab bot ke messages MarkdownRenderer se guzar kar asli formatting
    // (bold, headings, lists, tables) ban jate hain.
    //
    // User ka apna message plain text hi rehta hai -- us par formatting
    // lagane ka koi faida nahi, aur na hi uske asterisks badalne chahiye.
    const plainText = (sender === "bot" && window.MarkdownRenderer)
        ? window.MarkdownRenderer.toPlainText(text)
        : text;

    if (sender === "bot" && window.MarkdownRenderer) {
        div.appendChild(window.MarkdownRenderer.toElement(text));
    } else {
        div.textContent = text;
    }

    // Copy button (chat.html) aur Listen button dono ko saaf text chahiye --
    // markdown symbols ke baghair. Raw markdown bhi rakh lete hain taake
    // baad me "copy as markdown" jaisa feature add karna asaan rahe.
    div.__plainText = plainText;
    div.__rawText = text;

    if (sources.length > 0) {
        const src = document.createElement("div");
        src.className = "sources";
        src.textContent = "Source(s): " + sources.join(", ");
        div.appendChild(src);
    }
    const metricsText = formatMetrics(metrics);
    if (metricsText) {
        const met = document.createElement("div");
        met.className = "metrics";
        met.textContent = metricsText;
        div.appendChild(met);
    }
    if (sender === "bot") {
        // Text-to-speech ko plain text milta hai, warna screen reader
        // "star star Skills star star" jaisa bolta hai.
        div.appendChild(createSpeakButton(plainText));
    }
    chatBox.appendChild(div);
    if (chatScroll) chatScroll.scrollTop = chatScroll.scrollHeight;
}

// ---- Text-to-Speech (a "listen" button under every bot answer) ----
const SPEAK_ICON = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"/></svg><span>Listen</span>`;
const STOP_ICON = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="6" y="6" width="12" height="12" rx="1"/></svg><span>Stop</span>`;

let currentSpeakBtn = null; // the speak button (if any) currently reading aloud

function createSpeakButton(text) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "speak-btn";
    btn.title = "Read this answer aloud";
    btn.innerHTML = SPEAK_ICON;
    btn.addEventListener("click", () => speakText(text, btn));
    return btn;
}

function resetSpeakButton(btn) {
    if (!btn) return;
    btn.innerHTML = SPEAK_ICON;
    btn.title = "Read this answer aloud";
}

function speakText(text, btn = null) {
    if (!("speechSynthesis" in window)) return;

    const clickedActiveButton = btn && currentSpeakBtn === btn;

    window.speechSynthesis.cancel(); // stop whatever was being read, if anything
    if (currentSpeakBtn) resetSpeakButton(currentSpeakBtn);
    currentSpeakBtn = null;

    if (clickedActiveButton) return; // clicking the button that's already reading just stops it

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = "en-US";
    utterance.onend = utterance.onerror = () => {
        resetSpeakButton(btn);
        if (currentSpeakBtn === btn) currentSpeakBtn = null;
    };

    if (btn) {
        btn.innerHTML = STOP_ICON;
        btn.title = "Stop reading";
        currentSpeakBtn = btn;
    }

    // Chrome has a known bug where calling speak() in the same tick right
    // after cancel() silently does nothing -- the engine needs a moment to
    // actually finish cancelling first. A tiny delay avoids that race.
    setTimeout(() => window.speechSynthesis.speak(utterance), 50);
}

// ---- Speech-to-Text (mic button fills the question box) ----
const SpeechRecognitionAPI = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognition = null;
let isListening = false;

if (SpeechRecognitionAPI && micBtn) {
    recognition = new SpeechRecognitionAPI();
    recognition.lang = "en-US";
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    recognition.onstart = () => {
        isListening = true;
        micBtn.classList.add("listening");
    };

    recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        queryInput.value = transcript;
        sendQuery(); // auto-send the recognized question, same as pressing Send
    };

    recognition.onerror = (event) => {
        // Log the real reason instead of silently doing nothing, so mic
        // issues (blocked permission, insecure origin, no mic found, etc.)
        // are actually visible instead of just "nothing happens".
        console.error("Speech recognition error:", event.error);
        isListening = false;
        micBtn.classList.remove("listening");

        if (event.error === "not-allowed" || event.error === "service-not-allowed") {
            alert("Microphone access is blocked. Please allow microphone permission for this site in your browser settings.");
        } else if (event.error === "no-speech") {
            // Nothing said -- not a real error, just reset quietly.
        } else if (event.error === "network") {
            alert("Speech recognition needs an internet connection.");
        }
    };

    recognition.onend = () => {
        isListening = false;
        micBtn.classList.remove("listening");
    };

    micBtn.addEventListener("click", () => {
        if (isListening) {
            recognition.stop();
            return;
        }
        try {
            recognition.start();
        } catch (err) {
            // Fires if start() is called while recognition is already active
            // (e.g. state got out of sync after a previous error) -- reset and retry.
            console.error("Could not start speech recognition:", err);
            isListening = false;
            micBtn.classList.remove("listening");
        }
    });
} else if (micBtn) {
    // Browser doesn't support voice input (e.g. Firefox) -- hide the mic button.
    micBtn.style.display = "none";
}

uploadBtn.addEventListener("click", async () => {
    const file = documentInput.files[0];
    if (!file) {
        uploadStatus.textContent = "Please select a file first.";
        return;
    }
    const formData = new FormData();
    formData.append("document", file);
    // Langfuse par upload ka trace usi conversation ke sath group ho jaye.
    if (currentSessionId) formData.append("session_id", currentSessionId);

    uploadStatus.textContent = "Uploading...";
    const res = await fetch("/upload", { method: "POST", body: formData });
    const data = await res.json();

    if (res.ok) {
        uploadStatus.textContent = data.message;
        lastUploadedFilename = data.filename;
        addMessage(`Document uploaded: ${data.filename}`, "bot");
    } else {
        uploadStatus.textContent = data.error || "Upload failed.";
    }
});

summarizeBtn.addEventListener("click", async () => {
    addMessage("Generating summary...", "bot");
    const res = await fetch("/summarize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filename: lastUploadedFilename })
    });
    const data = await res.json();
    addMessage(data.summary || data.error, "bot", [], data.metrics);
});

// ---- Export as HTML: original page images + extracted text side by side ----
const exportHtmlBtn = document.getElementById("exportHtmlBtn");

function addExportResultMessage(stats, htmlContent) {
    const div = document.createElement("div");
    div.className = "msg bot";

    const summaryParts = [
        `HTML export ready for "${stats.filename}":`,
        `${stats.ok_pages}/${stats.total_pages} page(s) OCR'd successfully (${stats.success_pct}%),`,
        `${stats.total_words} word(s) extracted`,
    ];
    if (stats.unclear_markers) summaryParts.push(`, ${stats.unclear_markers} [unclear] marker(s) flagged`);
    if (stats.failed_pages) summaryParts.push(`, ${stats.failed_pages} page(s) failed`);
    if (stats.skipped_pages) summaryParts.push(`, ${stats.skipped_pages} page(s) skipped (page limit)`);
    const summary = document.createElement("div");
    summary.textContent = summaryParts.join(" ");
    div.appendChild(summary);

    const note = document.createElement("div");
    note.className = "sources";
    note.textContent = "The \"success rate\" above is how many pages OCR could read text from, not a proofread accuracy score -- open the file and compare each page's image against its text to check for yourself.";
    div.appendChild(note);

    const blob = new Blob([htmlContent], { type: "text/html" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    const baseName = (stats.filename || "document").replace(/\.[^./\\]+$/, "");
    link.download = `${baseName}-ocr-export.html`;
    link.className = "header-action-btn export-download-link";
    link.textContent = "Download HTML";
    div.appendChild(link);

    chatBox.appendChild(div);
    if (chatScroll) chatScroll.scrollTop = chatScroll.scrollHeight;
}

if (exportHtmlBtn) {
    exportHtmlBtn.addEventListener("click", async () => {
        if (!lastUploadedFilename) {
            addMessage("Please upload a PDF or image first, then export it as HTML.", "bot");
            return;
        }
        addMessage("Generating HTML export (OCR-ing every page -- this can take a little while for longer files)...", "bot");
        try {
            const res = await fetch("/export-html", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ filename: lastUploadedFilename })
            });
            const data = await res.json();
            if (!res.ok) {
                addMessage(data.error || "Could not generate the HTML export.", "bot");
                return;
            }
            addExportResultMessage(data.stats || {}, data.html);
        } catch (err) {
            console.error("Export HTML failed:", err);
            addMessage("Could not generate the HTML export (network error). Please try again.", "bot");
        }
    });
}

async function sendQuery() {
    const query = queryInput.value.trim();
    if (!query) return;
    addMessage(query, "user");
    queryInput.value = "";

    const res = await fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, session_id: currentSessionId })
    });
    const data = await res.json();

    if (data.session_id) {
        const isNewSession = data.session_id !== currentSessionId;
        setSessionId(data.session_id);
        // Tell the sidebar (chat.html) to refresh its conversation list when a
        // brand-new conversation was just created by this message.
        if (isNewSession && window.ChatUI && window.ChatUI.refreshChatList) {
            window.ChatUI.refreshChatList();
        }
    }

    addMessage(data.answer || data.error, "bot", data.sources || [], data.metrics);
}

sendBtn.addEventListener("click", sendQuery);
queryInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") sendQuery();
});


// Load one conversation's history (the currently active session, if any)
async function loadHistory(sessionId) {
    if (!sessionId) return;
    const res = await fetch(`/history?session_id=${encodeURIComponent(sessionId)}`);
    if (!res.ok) return;
    const data = await res.json();
    (data.history || []).forEach(item => {
        addMessage(item.question, "user");
        addMessage(item.response, "bot", item.sources || [], item.metrics);
    });
}

loadHistory(currentSessionId);

// ---- Small API for the sidebar (chat.html inline script) to drive this file ----
window.ChatUI_core = {
    getSessionId: () => currentSessionId,
    // Switch the visible chat to a different saved conversation.
    loadSession: async function (sessionId) {
        setSessionId(sessionId);
        chatBox.innerHTML = "";
        await loadHistory(sessionId);
    },
    // Start a fresh, empty conversation (nothing is deleted -- the old one stays saved).
    startNewChat: function () {
        setSessionId(null);
        chatBox.innerHTML = "";
    },
};