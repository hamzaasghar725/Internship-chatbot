"""Embedding model loading and query embedding."""
from sentence_transformers import SentenceTransformer

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

_embed_model = None


def get_embed_model():
    """Lazy-loads the embedding model (only loads once)."""
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model


def embed_query(query):
    """
    Sawal ko vector me badalta hai.

    Pehle ye kaam retrieve_relevant_chunks() ke andar chhupa hua tha, is
    liye Langfuse par embedding aur FAISS search dono ka waqt ek hi span me
    mila jula nazar aata tha. Ab alag function hai taake "embedding" apna
    span bana sake aur dashboard par pata chale ke asal me dair kis step me
    ho rahi hai.
    """
    model = get_embed_model()
    return model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")
