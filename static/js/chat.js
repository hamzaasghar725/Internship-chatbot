const chatBox = document.getElementById("chatBox");
const chatScroll = document.querySelector(".chat-scroll"); // this is the element that actually scrolls (see style.css)
const queryInput = document.getElementById("queryInput");
const sendBtn = document.getElementById("sendBtn");
const uploadBtn = document.getElementById("uploadBtn");
const summarizeBtn = document.getElementById("summarizeBtn");
const documentInput = document.getElementById("documentInput");
const uploadStatus = document.getElementById("uploadStatus");
const micBtn = document.getElementById("micBtn");
const voiceToggle = document.getElementById("voiceToggle");

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

function addMessage(text, sender, sources = [], metrics = null, speakIt = false) {
    const div = document.createElement("div");
    div.className = `msg ${sender}`;
    div.textContent = text;
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
    chatBox.appendChild(div);
    if (chatScroll) chatScroll.scrollTop = chatScroll.scrollHeight;

    // Read the bot's answer aloud if the "Read answers aloud" toggle is on.
    // speakIt is only true for a fresh live answer, never for history loaded on page open.
    if (speakIt && sender === "bot" && voiceToggle && voiceToggle.checked) {
        speakText(text);
    }
}

// ---- Text-to-Speech (bot reads its answer aloud) ----
function speakText(text) {
    if (!("speechSynthesis" in window)) return;
    window.speechSynthesis.cancel(); // stop any answer currently being read
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = "en-US";
    window.speechSynthesis.speak(utterance);
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

    recognition.onerror = () => {
        // Mic access denied, no speech detected, etc. -- just reset the button.
    };

    recognition.onend = () => {
        isListening = false;
        micBtn.classList.remove("listening");
    };

    micBtn.addEventListener("click", () => {
        if (isListening) {
            recognition.stop();
        } else {
            recognition.start();
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
    addMessage(data.summary || data.error, "bot", [], data.metrics, true);
});

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

    addMessage(data.answer || data.error, "bot", data.sources || [], data.metrics, true);
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