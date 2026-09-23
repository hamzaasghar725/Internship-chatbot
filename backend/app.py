import os
import secrets
import uuid
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from dotenv import load_dotenv
from sqlalchemy import text

# Must run before importing clerk_utils (and any other local module that reads
# CLERK_*/other env vars at import time) -- otherwise those modules capture
# empty strings instead of the values from .env.
load_dotenv()

from models import db, User, ChatHistory, ChatSession
from rag.rag_utils import build_or_update_index, answer_question, summarize_document, OCRError, IMAGE_EXTENSIONS
from rag.ocr_report import build_ocr_html_report
from face_utils import decode_base64_image, get_face_embedding, embedding_to_json, find_matching_user, FaceNotDetectedError, MultipleFacesDetectedError
from clerk_utils import (
    is_clerk_configured,
    verify_session_token,
    get_clerk_user_email,
    ClerkNotConfiguredError,
    ClerkVerificationError,
    CLERK_PUBLISHABLE_KEY,
    CLERK_FRONTEND_API,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
# Images (png/jpg/webp/bmp/tiff/gif) aur scanned PDFs OCR se padhe jate hain (rag/ocr.py).
ALLOWED_EXTENSIONS = {"pdf", "txt", "docx", "csv"} | IMAGE_EXTENSIONS

# Project layout: backend/ (this file) and frontend/ (templates + static) are
# now separate sibling folders, so Flask is pointed explicitly at frontend/
# for its templates and static files. This is a path change only -- URLs
# (url_for('static', ...)), template names, and all rendering behavior stay
# exactly the same as before.
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")
app = Flask(
    __name__,
    template_folder=os.path.join(FRONTEND_DIR, "templates"),
    static_folder=os.path.join(FRONTEND_DIR, "static"),
)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-key")
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.join(BASE_DIR, "users.db")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

db.init_app(app)


@app.context_processor
def inject_clerk_config():
    """Makes Clerk's publishable key / frontend API domain available to every
    template as {{ clerk_publishable_key }} / {{ clerk_frontend_api }}, and a
    {{ clerk_enabled }} flag so the templates can show a setup notice instead
    of a broken widget if the .env keys haven't been added yet."""
    return {
        "clerk_publishable_key": CLERK_PUBLISHABLE_KEY,
        "clerk_frontend_api": CLERK_FRONTEND_API,
        "clerk_enabled": is_clerk_configured(),
    }


def unique_username_from_email(email):
    """Turns 'hamza@gmail.com' into a free username: 'hamza', or 'hamza2',
    'hamza3', ... if that's already taken."""
    base = (email.split("@")[0] or "user").strip().lower()
    base = "".join(ch for ch in base if ch.isalnum() or ch in ("_", ".", "-")) or "user"
    candidate = base
    suffix = 1
    while User.query.filter_by(username=candidate).first() is not None:
        suffix += 1
        candidate = f"{base}{suffix}"
    return candidate

login_manager = LoginManager()
login_manager.login_view = "login"
login_manager.init_app(app)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------- Auth Routes ----------------

@app.route("/")
def home():
    if current_user.is_authenticated:
        return redirect(url_for("chat"))
    return redirect(url_for("login"))


@app.route("/signup", methods=["GET"])
@app.route("/signup/<path:_subpath>", methods=["GET"])
def signup(_subpath=None):
    return render_template("signup.html")


@app.route("/signup-complete", methods=["POST"])
def signup_complete():
    """Called by signup.html *after* Clerk's own <SignUp> widget has finished
    (account created, email verified). Verifies the Clerk session token,
    creates our local User row (linked via clerk_user_id) so the rest of the
    app -- which still runs on Flask-Login -- keeps working unchanged, and
    optionally stores a face embedding so face-login also works right away."""
    data = request.get_json(force=True, silent=True) or {}
    token = (data.get("token") or "").strip()
    face_image = (data.get("face_image") or "").strip()

    if not token:
        return jsonify({"success": False, "error": "Missing Clerk session token."}), 400

    try:
        payload = verify_session_token(request)
    except ClerkNotConfiguredError as e:
        return jsonify({"success": False, "error": str(e)}), 500
    except ClerkVerificationError as e:
        return jsonify({"success": False, "error": f"Could not verify your account: {e}"}), 401

    clerk_user_id = payload.get("sub") if isinstance(payload, dict) else getattr(payload, "sub", None)
    if not clerk_user_id:
        return jsonify({"success": False, "error": "Clerk did not return a user id."}), 400

    email = get_clerk_user_email(clerk_user_id)
    if not email:
        return jsonify({"success": False, "error": "Your email isn't verified yet. Please check your inbox."}), 400

    # Face enrolment is required -- reject before touching the database so a
    # failed capture can't leave behind a face-less account that would then
    # be able to sign in without ever enrolling.
    if not face_image:
        return jsonify({"success": False, "error": "Please capture your face to finish signing up."}), 400

    try:
        image_array = decode_base64_image(face_image)
        face_vector = get_face_embedding(image_array)
    except (MultipleFacesDetectedError, FaceNotDetectedError) as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception:
        return jsonify({"success": False, "error": "Couldn't process that photo. Please retake it."}), 400

    user = User.query.filter_by(clerk_user_id=clerk_user_id).first()
    if user is None:
        user = User.query.filter_by(email=email).first()  # e.g. re-running signup after a partial failure

    if user is None:
        user = User(username=unique_username_from_email(email), email=email, clerk_user_id=clerk_user_id)
        user.set_password(secrets.token_urlsafe(32))  # unusable random password; Clerk owns real auth for this account
    else:
        user.clerk_user_id = clerk_user_id
        user.email = email

    user.face_embedding = embedding_to_json(face_vector)

    db.session.add(user)
    db.session.commit()
    login_user(user)
    return jsonify({"success": True, "username": user.username, "redirect": url_for("chat")})


@app.route("/login", methods=["GET"])
@app.route("/login/<path:_subpath>", methods=["GET"])
def login(_subpath=None):
    # The /login/<subpath> route exists because Clerk's SignIn widget uses
    # path-based routing -- it pushes real URLs like /login/factor-two via
    # the History API. This lets a refresh of that URL still load the page.
    return render_template("login.html")


@app.route("/clerk-sync", methods=["POST"])
def clerk_sync():
    """Called by login.html after Clerk's <SignIn> widget confirms the user
    is signed in. Verifies the token, finds/links the matching local User
    row, and logs them into Flask-Login so the rest of the app works exactly
    as it did with the old username/password login."""
    data = request.get_json(force=True, silent=True) or {}
    token = (data.get("token") or "").strip()
    if not token:
        return jsonify({"success": False, "error": "Missing Clerk session token."}), 400

    try:
        payload = verify_session_token(request)
    except ClerkNotConfiguredError as e:
        return jsonify({"success": False, "error": str(e)}), 500
    except ClerkVerificationError as e:
        return jsonify({"success": False, "error": f"Session could not be verified: {e}"}), 401

    clerk_user_id = payload.get("sub") if isinstance(payload, dict) else getattr(payload, "sub", None)
    if not clerk_user_id:
        return jsonify({"success": False, "error": "Clerk did not return a user id."}), 400

    user = User.query.filter_by(clerk_user_id=clerk_user_id).first()
    if user is None:
        # This Clerk identity has no local account yet, which means the
        # /signup-complete step (where the face is captured) never ran.
        # Do NOT create the account here -- doing so would let anyone skip
        # face enrolment entirely just by landing on /login. Send them to
        # /signup to finish properly.
        email = get_clerk_user_email(clerk_user_id)
        if not email:
            return jsonify({"success": False, "error": "Your email isn't verified yet."}), 400

        existing = User.query.filter_by(email=email).first()
        if existing is not None and existing.face_embedding:
            # Same person, already fully enrolled under this email (e.g. they
            # previously signed up with a different Clerk method). Safe to link.
            existing.clerk_user_id = clerk_user_id
            db.session.add(existing)
            db.session.commit()
            user = existing
        else:
            return jsonify({
                "success": False,
                "needs_signup": True,
                "redirect": url_for("signup"),
                "error": "Please finish signing up by capturing your face.",
            }), 403

    if not user.face_embedding:
        # Account exists but face enrolment never completed -- finish it first.
        return jsonify({
            "success": False,
            "needs_signup": True,
            "redirect": url_for("signup"),
            "error": "Please finish signing up by capturing your face.",
        }), 403

    login_user(user)
    return jsonify({"success": True, "username": user.username, "redirect": url_for("chat")})


@app.route("/login-face", methods=["POST"])
def login_face():
    data = request.get_json(force=True, silent=True) or {}
    face_image = data.get("image", "").strip()

    if not face_image:
        return jsonify({"success": False, "error": "No image received."}), 400

    try:
        image_array = decode_base64_image(face_image)
        candidate_vector = get_face_embedding(image_array)
    except MultipleFacesDetectedError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except FaceNotDetectedError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception:
        return jsonify({"success": False, "error": "Could not process the image. Please try again."}), 400

    users_with_faces = User.query.filter(User.face_embedding.isnot(None)).all()
    matched_user, distance = find_matching_user(candidate_vector, users_with_faces)

    if matched_user is None:
        return jsonify({
            "success": False,
            "error": "Face did not match. Please log in with your password or try again."
        }), 401

    login_user(matched_user)
    return jsonify({
        "success": True,
        "username": matched_user.username,
        "redirect": url_for("chat"),
    })


@app.route("/logout")
@login_required
def logout():
    logout_user()
    # Flask-Login's session is cleared above, but Clerk keeps its own,
    # separate client-side session in the browser. If we redirect straight
    # to /login without also signing out of Clerk, that page sees Clerk is
    # still signed in and immediately logs the user back in. Render a tiny
    # page that signs out of Clerk first, then sends them to /login.
    return render_template("logout.html")


# ---------------- Chat / RAG Routes ----------------

@app.route("/chat")
@login_required
def chat():
    return render_template("chat.html", username=current_user.username)


@app.route("/chats", methods=["GET"])
@login_required
def list_chats():
    """Return one entry per saved conversation (session), most recently active first."""
    records = (
        ChatHistory.query
        .filter_by(user_id=current_user.id)
        .order_by(ChatHistory.timestamp.asc())
        .all()
    )
    # Custom titles set via the rename endpoint, keyed by session_id.
    custom_titles = {
        s.session_id: s.title
        for s in ChatSession.query.filter_by(user_id=current_user.id).all()
        if s.title
    }

    sessions = {}
    for r in records:
        sid = r.session_id or "legacy"
        if sid not in sessions:
            sessions[sid] = {
                "session_id": sid,
                "title": custom_titles.get(sid) or (
                    (r.question[:48] + "...") if len(r.question) > 48 else r.question
                ),
                "last_timestamp": r.timestamp.isoformat(),
            }
        else:
            sessions[sid]["last_timestamp"] = r.timestamp.isoformat()

    chats = sorted(sessions.values(), key=lambda c: c["last_timestamp"], reverse=True)
    return jsonify({"chats": chats})


def _owns_session(session_id):
    """True if the current user has any messages under this session_id (or a
    ChatSession row for it), so rename/delete can't touch another user's chat."""
    sid = None if session_id == "legacy" else session_id
    has_messages = ChatHistory.query.filter_by(user_id=current_user.id, session_id=sid).first() is not None
    has_session_row = ChatSession.query.filter_by(user_id=current_user.id, session_id=session_id).first() is not None
    return has_messages or has_session_row


@app.route("/chats/<session_id>/rename", methods=["POST"])
@login_required
def rename_chat(session_id):
    if not _owns_session(session_id):
        return jsonify({"success": False, "error": "Chat not found."}), 404

    data = request.get_json(force=True, silent=True) or {}
    new_title = (data.get("title") or "").strip()
    if not new_title:
        return jsonify({"success": False, "error": "Title can't be empty."}), 400
    new_title = new_title[:120]

    chat_session = ChatSession.query.filter_by(session_id=session_id).first()
    if chat_session is None:
        chat_session = ChatSession(session_id=session_id, user_id=current_user.id)
        db.session.add(chat_session)
    chat_session.title = new_title
    db.session.commit()

    return jsonify({"success": True, "title": new_title})


@app.route("/chats/<session_id>/delete", methods=["POST"])
@login_required
def delete_chat(session_id):
    if not _owns_session(session_id):
        return jsonify({"success": False, "error": "Chat not found."}), 404

    sid = None if session_id == "legacy" else session_id
    ChatHistory.query.filter_by(user_id=current_user.id, session_id=sid).delete()
    ChatSession.query.filter_by(user_id=current_user.id, session_id=session_id).delete()
    db.session.commit()

    return jsonify({"success": True})


@app.route("/history", methods=["GET"])
@login_required
def history():
    """Return one conversation's questions & answers, oldest first.
    A ?session_id= query param selects which conversation; without it, an empty
    (new, unsaved) chat is returned."""
    session_id = request.args.get("session_id")
    if not session_id:
        return jsonify({"history": []})

    sid = None if session_id == "legacy" else session_id
    records = (
        ChatHistory.query
        .filter_by(user_id=current_user.id, session_id=sid)
        .order_by(ChatHistory.timestamp.asc())
        .all()
    )
    return jsonify({"history": [r.to_dict() for r in records]})


@app.route("/history/clear", methods=["POST"])
@login_required
def clear_history():
    """Delete this user's ENTIRE chat history across all conversations (optional utility)."""
    ChatHistory.query.filter_by(user_id=current_user.id).delete()
    db.session.commit()
    return jsonify({"message": "Chat history cleared."})


@app.route("/upload", methods=["POST"])
@login_required
def upload():
    if "document" not in request.files:
        return jsonify({"error": "No file was received."}), 400

    file = request.files["document"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Only PDF, TXT, DOCX, CSV or image files (PNG, JPG, WEBP, BMP, TIFF, GIF) are allowed."}), 400

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{current_user.id}_{file.filename}")
    file.save(save_path)

    # session_id optional hai. Frontend bhejta hai taake Langfuse par
    # document ka "document-ingest" trace usi conversation ke neeche group
    # ho jaye jisme wo document upload kiya gaya tha -- yani ek hi session
    # me upload aur uske baad wale sawal sath nazar aate hain.
    session_id = request.form.get("session_id") or None
    try:
        num_chunks = build_or_update_index(current_user.id, save_path, file.filename, session_id=session_id)
    except OCRError as e:
        # Scanned file / image ko OCR se padhte waqt masla (key missing, model busy, kharab image)
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        print(f"[Upload] could not process '{file.filename}': {e}")
        return jsonify({"error": "Could not read this file. It may be corrupted or password-protected."}), 422

    if num_chunks == 0:
        return jsonify({"error": "No readable text was found in this file. If it is a scan or photo, please upload a clearer image."}), 422

    return jsonify({
        "message": f"'{file.filename}' uploaded and {num_chunks} chunks added to the index.",
        "filename": file.filename
    })


@app.route("/export-html", methods=["POST"])
@login_required
def export_html():
    """
    Generates a standalone HTML "proof sheet" for a scanned PDF / image
    already uploaded by this user: every page's original scanned image next
    to the text OCR extracted from it (side by side), plus honest extraction
    stats. Runs its OWN OCR pass (rag/rag_utils.py's RAG indexing pipeline is
    untouched by this) -- see rag/ocr_report.py for the full explanation.
    """
    data = request.get_json(force=True, silent=True) or {}
    filename = (data.get("filename") or "").strip()
    if not filename:
        return jsonify({"error": "No document to export yet. Please upload a PDF or image first."}), 400

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{current_user.id}_{filename}")
    if not os.path.exists(file_path):
        return jsonify({"error": "That file could not be found. Please upload it again."}), 404

    try:
        html_out, stats = build_ocr_html_report(file_path, filename)
    except OCRError as e:
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        print(f"[ExportHTML] failed for '{filename}': {e}")
        return jsonify({"error": "Could not generate the HTML export. Please try again."}), 500

    return jsonify({"html": html_out, "stats": stats})


@app.route("/ask", methods=["POST"])
@login_required
def ask():
    data = request.get_json() or {}
    query = (data.get("query") or "").strip()
    session_id = data.get("session_id") or str(uuid.uuid4())
    if not query:
        return jsonify({"error": "Question is empty."}), 400

    answer, sources = answer_question(current_user.id, query, session_id=session_id, top_k=4)

    # Save this Q&A into chat history, tagged with its conversation (session_id)
    record = ChatHistory(
        user_id=current_user.id,
        question=query,
        response=answer,
        sources=",".join(sources) if sources else None,
        session_id=session_id,
    )
    db.session.add(record)
    db.session.commit()

    return jsonify({"answer": answer, "sources": sources, "session_id": session_id})


@app.route("/summarize", methods=["POST"])
@login_required
def summarize():
    data = request.get_json() or {}
    filename = data.get("filename")  # optional; None = all documents
    session_id = data.get("session_id")
    summary = summarize_document(current_user.id, filename, session_id=session_id)
    return jsonify({"summary": summary})


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        # One-time, safe migration: add the session_id column if it doesn't exist yet
        # (needed because chat_history.db was created before this column existed).
        existing_cols = [row[1] for row in db.session.execute(text("PRAGMA table_info(chat_history)")).fetchall()]
        if "session_id" not in existing_cols:
            db.session.execute(text("ALTER TABLE chat_history ADD COLUMN session_id VARCHAR(36)"))
            db.session.commit()

        # Same idea for the "user" table: add the Clerk-linking column if this
        # users.db was created before Clerk was integrated.
        existing_user_cols = [row[1] for row in db.session.execute(text("PRAGMA table_info(user)")).fetchall()]
        if "clerk_user_id" not in existing_user_cols:
            db.session.execute(text("ALTER TABLE user ADD COLUMN clerk_user_id VARCHAR(64)"))
            db.session.commit()
    app.run(debug=True, port=5000, use_reloader=False)