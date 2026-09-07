from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

db = SQLAlchemy()


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    face_embedding = db.Column(db.Text, nullable=True)  # JSON-encoded face vector

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    

class ChatHistory(db.Model):
    """Stores every question the user asked and the bot's response."""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    question = db.Column(db.Text, nullable=False)
    response = db.Column(db.Text, nullable=False)
    sources = db.Column(db.Text, nullable=True)  # comma-separated filenames
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship("User", backref=db.backref("chat_history", lazy=True))

    def to_dict(self):
        return {
            "id": self.id,
            "question": self.question,
            "response": self.response,
            "sources": self.sources.split(",") if self.sources else [],
            "timestamp": self.timestamp.isoformat(),
        }
