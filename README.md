# RAG Chatbot (Login + Document Q&A + Summarization)

Ye ek internship project ke liye full-fledged chatbot hai jisme:
- Sign Up / Sign In (Flask-Login + SQLite)
- Document upload (PDF/TXT)
- RAG (Retrieval-Augmented Generation) se document ke basis par sawalon ke jawab
- Document summarization

## Setup

1. Virtual environment banayein (recommended):
   ```
   python3 -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

2. Dependencies install karein:
   ```
   pip install -r requirements.txt
   ```

3. `.env.example` ko `.env` mein copy karein aur apni values daalein:
   ```
   cp .env.example .env
   ```
   - `SECRET_KEY`: koi bhi random string
   - `OPENAI_API_KEY`: apni OpenAI API key (ye na ho to bhi app chalegi — sirf LLM-generated answers ki jagah retrieved text dikhega)

4. App run karein:
   ```
   python app.py
   ```

5. Browser mein kholein: `http://127.0.0.1:5000`

## Kaise kaam karta hai (flow)

1. **Sign Up / Login** — user account banata hai, phir login karta hai.
2. **Upload** — PDF/TXT document upload karta hai. Backend document ko:
   - chunks mein todta hai (`rag/rag_utils.py -> chunk_text`)
   - har chunk ko embedding (`sentence-transformers`) mein convert karta hai
   - FAISS index mein store karta hai (har user ka apna alag index — `vectorstore/<user_id>/`)
3. **Ask** — user sawal poochta hai. Backend:
   - query ko embed karta hai
   - FAISS se sab se relevant chunks nikalta hai (semantic search)
   - un chunks ko context bana ke LLM (OpenAI) se answer generate karwata hai
4. **Summarize** — uploaded document ke chunks ko combine kar ke LLM se summary banwata hai.

## Project Structure

```
chatbot_project/
├── app.py                 # Flask routes (auth + chat + upload + ask + summarize)
├── models.py               # User database model
├── rag/
│   └── rag_utils.py        # Chunking, embeddings, FAISS, LLM calls
├── templates/               # login.html, signup.html, chat.html
├── static/                  # CSS + JS
├── uploads/                  # Uploaded documents yahan save hote hain
├── vectorstore/               # Har user ka FAISS index yahan store hota hai
└── requirements.txt
```

## Aage Extend Karne Ke Ideas (agar demo ke baad improve karna ho)

- OpenAI ki jagah free local LLM (Ollama + llama3) use karein — sirf `rag_utils.py` mein `generate_answer`/`summarize_document` functions change karne honge.
- Multiple documents ki alag-alag chat history save karna (SQLite mein Chat model add karein).
- Chunking ko better banane ke liye `langchain` ka `RecursiveCharacterTextSplitter` use kar sakte hain.
- File type support badhana (docx, csv, etc).

## Note

Ye project academic/demo purpose ke liye hai. Production mein deploy karne se pehle:
- File upload size/type validation aur security hardening karein
- `.env` file ko kabhi bhi GitHub pe push na karein
- `SECRET_KEY` production mein strong aur secret rakhein

## Face Recognition Login (new)

Signup ab webcam se chehra capture karta hai, aur login page par "Face" tab
se bhi login ho sakta hai — password ki zaroorat nahi agar face match ho jaye.

### Kaise kaam karta hai

1. **Signup**: Browser camera se ek frame capture hota hai, backend
   `DeepFace` (Facenet model) se us chehre ka 128-number "embedding" vector
   nikalta hai, aur `users.db` mein us user ke sath save kar deta hai.
2. **Login (Face tab)**: Naya frame capture hota hai, uska embedding nikala
   jata hai, aur database ke saare stored embeddings ke sath cosine
   distance compare hoti hai. Sabse qareeb match jo threshold (0.40) se
   kam ho, wahi user login ho jata hai.
3. Agar koi match na mile, error dikhta hai aur password tab se login
   kiya ja sakta hai.

### Setup notes

- `pip install -r requirements.txt` ab `deepface`, `opencv-python`, aur
  `Pillow` bhi install karega. Pehli baar chalne par DeepFace apne
  model weights internet se download karega (~90MB) — internet connection
  chahiye hoga is ek dafa ke liye.
- **Agar aapke paas pehle se `users.db` file hai** (purane signup ke sath),
  usko delete kar dein taake naya `face_embedding` column sahi se create
  ho: naya database automatically ban jayega jab app dobara chalayenge.
- Camera access ke liye browser permission dena hoga — pehli baar
  signup/login page kholte waqt browser popup dikhayega.
- `face_utils.py` mein `MATCH_THRESHOLD` value tune ki ja sakti hai agar
  matches bohat strict ya bohat loose lagen (lower = strict, higher = loose).
