import os
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from dotenv import load_dotenv

from models import db, User, ChatHistory
from rag.rag_utils import build_or_update_index, retrieve_relevant_chunks, generate_answer, summarize_document
from face_utils import decode_base64_image, get_face_embedding, embedding_to_json, find_matching_user, FaceNotDetectedError, MultipleFacesDetectedError

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
ALLOWED_EXTENSIONS = {"pdf", "txt"}

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-key")
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.join(BASE_DIR, "users.db")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

db.init_app(app)

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


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        face_image = request.form.get("face_image", "").strip()

        if not username or not email or not password:
            flash("All fields are required.", "error")
            return redirect(url_for("signup"))

        if not face_image:
            flash("Please capture your face before completing signup.", "error")
            return redirect(url_for("signup"))

        if User.query.filter((User.username == username) | (User.email == email)).first():
            flash("This username or email is already registered.", "error")
            return redirect(url_for("signup"))

        try:
            image_array = decode_base64_image(face_image)
            face_vector = get_face_embedding(image_array)
        except MultipleFacesDetectedError as e:
            flash(str(e), "error")
            return redirect(url_for("signup"))
        except FaceNotDetectedError as e:
            flash(str(e), "error")
            return redirect(url_for("signup"))
        except Exception:
            flash("Could not process the face image. Please try again.", "error")
            return redirect(url_for("signup"))

        user = User(username=username, email=email)
        user.set_password(password)
        user.face_embedding = embedding_to_json(face_vector)
        db.session.add(user)
        db.session.commit()

        flash("Account created! Please log in now.", "success")
        return redirect(url_for("login"))

    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for("chat"))

        flash("Incorrect username or password.", "error")
        return redirect(url_for("login"))

    return render_template("login.html")


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
    return redirect(url_for("login"))


# ---------------- Chat / RAG Routes ----------------

@app.route("/chat")
@login_required
def chat():
    return render_template("chat.html", username=current_user.username)


@app.route("/history", methods=["GET"])
@login_required
def history():
    """Return this user's past questions & answers, oldest first."""
    records = (
        ChatHistory.query
        .filter_by(user_id=current_user.id)
        .order_by(ChatHistory.timestamp.asc())
        .all()
    )
    return jsonify({"history": [r.to_dict() for r in records]})


@app.route("/history/clear", methods=["POST"])
@login_required
def clear_history():
    """Delete this user's chat history (optional utility)."""
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
        return jsonify({"error": "Only .pdf or .txt files are allowed."}), 400

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{current_user.id}_{file.filename}")
    file.save(save_path)

    num_chunks = build_or_update_index(current_user.id, save_path, file.filename)

    return jsonify({
        "message": f"'{file.filename}' uploaded and {num_chunks} chunks added to the index.",
        "filename": file.filename
    })


@app.route("/ask", methods=["POST"])
@login_required
def ask():
    data = request.get_json()
    query = (data or {}).get("query", "").strip()
    if not query:
        return jsonify({"error": "Question is empty."}), 400

    chunks = retrieve_relevant_chunks(current_user.id, query, top_k=4)
    answer = generate_answer(query, chunks)
    sources = list({c["source"] for c in chunks})

    # Save this Q&A into chat history
    record = ChatHistory(
        user_id=current_user.id,
        question=query,
        response=answer,
        sources=",".join(sources) if sources else None,
    )
    db.session.add(record)
    db.session.commit()

    return jsonify({"answer": answer, "sources": sources})


@app.route("/summarize", methods=["POST"])
@login_required
def summarize():
    data = request.get_json() or {}
    filename = data.get("filename")  # optional; None = all documents
    summary = summarize_document(current_user.id, filename)
    return jsonify({"summary": summary})


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True, port=5000, use_reloader=False)