const chatBox = document.getElementById("chatBox");
const queryInput = document.getElementById("queryInput");
const sendBtn = document.getElementById("sendBtn");
const uploadBtn = document.getElementById("uploadBtn");
const summarizeBtn = document.getElementById("summarizeBtn");
const documentInput = document.getElementById("documentInput");
const uploadStatus = document.getElementById("uploadStatus");

let lastUploadedFilename = null;

function addMessage(text, sender, sources = []) {
    const div = document.createElement("div");
    div.className = `msg ${sender}`;
    div.textContent = text;
    if (sources.length > 0) {
        const src = document.createElement("div");
        src.className = "sources";
        src.textContent = "Source(s): " + sources.join(", ");
        div.appendChild(src);
    }
    chatBox.appendChild(div);
    chatBox.scrollTop = chatBox.scrollHeight;
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
    addMessage(data.summary || data.error, "bot");
});

async function sendQuery() {
    const query = queryInput.value.trim();
    if (!query) return;
    addMessage(query, "user");
    queryInput.value = "";

    const res = await fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query })
    });
    const data = await res.json();
    addMessage(data.answer || data.error, "bot", data.sources || []);
}

sendBtn.addEventListener("click", sendQuery);
queryInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") sendQuery();
});


// Load previous chat history when the page opens
async function loadHistory() {
    const res = await fetch("/history");
    if (!res.ok) return;
    const data = await res.json();
    (data.history || []).forEach(item => {
        addMessage(item.question, "user");
        addMessage(item.response, "bot", item.sources || []);
    });
}

loadHistory();