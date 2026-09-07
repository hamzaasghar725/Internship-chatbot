"""
face_utils.py
==============
Handles everything related to face recognition:
  - Decoding a webcam snapshot (base64 data URL) into an image
  - Extracting a face embedding vector (a list of numbers that represents
    the unique features of a face) using DeepFace
  - Comparing embeddings to find a matching user at login time

How the matching works:
  Every face is converted into a 128-number vector by the Facenet model.
  Two photos of the SAME person produce vectors that are close together
  (small "distance"). Two photos of DIFFERENT people produce vectors that
  are far apart. We use cosine distance to measure "far apart" — a lower
  number means more similar. MATCH_THRESHOLD decides how close is close
  enough to count as a match.
"""

import base64
import io
import json

import numpy as np
from PIL import Image
from deepface import DeepFace

MODEL_NAME = "Facenet512"
DETECTOR_BACKEND = "opencv"

# Cosine distance threshold: lower = stricter (harder for two different people
# to match), higher = looser (matches more easily, even with a partially similar face).
# For Facenet512, below 0.30 is recommended for strong security.
MATCH_THRESHOLD = 0.25


class FaceNotDetectedError(Exception):
    """Raised when no face could be found in the submitted image."""
    pass


class MultipleFacesDetectedError(Exception):
    """Raised when more than one face is found in the submitted image."""
    pass


def decode_base64_image(data_url):
    """Convert a 'data:image/jpeg;base64,...' string from the browser into a numpy array."""
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    image_bytes = base64.b64decode(data_url)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.array(image)


def get_face_embedding(image_array):
    """Detect the face in the image and return its embedding vector as a numpy array."""
    try:
        results = DeepFace.represent(
            img_path=image_array,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True,
        )
    except Exception as exc:
        raise FaceNotDetectedError(
            "Could not detect a face. Please improve lighting and look directly at the camera."
        ) from exc

    if not results:
        raise FaceNotDetectedError("Could not detect a face.")

    if len(results) > 1:
        raise MultipleFacesDetectedError(
            f"{len(results)} faces detected in the frame. "
            "Please make sure only one person is visible to the camera."
        )

    return np.array(results[0]["embedding"], dtype=np.float32)


def embedding_to_json(vector):
    return json.dumps(vector.tolist())


def embedding_from_json(text):
    return np.array(json.loads(text), dtype=np.float32)


def cosine_distance(vector_a, vector_b):
    denom = (np.linalg.norm(vector_a) * np.linalg.norm(vector_b)) + 1e-8
    similarity = float(np.dot(vector_a, vector_b) / denom)
    return 1 - similarity


def find_matching_user(candidate_embedding, users):
    """
    Compare the candidate embedding against every user who has a stored face.
    Returns (best_user, distance) if a match is found within MATCH_THRESHOLD,
    otherwise (None, best_distance_found_or_None).
    """
    best_user = None
    best_distance = None

    for user in users:
        if not user.face_embedding:
            continue
        stored_vector = embedding_from_json(user.face_embedding)
        distance = cosine_distance(candidate_embedding, stored_vector)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_user = user

    if best_user is not None and best_distance <= MATCH_THRESHOLD:
        return best_user, best_distance
    return None, best_distance