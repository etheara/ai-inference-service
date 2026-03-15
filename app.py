# Standard library imports
import os

# FastAPI core + security helpers
from fastapi import FastAPI, Header, HTTPException

# Data validation / request & response models
from pydantic import BaseModel

# Numerical computation
import numpy as np

# Model persistence (load trained ML models)
import joblib

# Sentence embedding model
from sentence_transformers import SentenceTransformer

# Typing helpers
from typing import Optional, List


# -------------------------------------------------
# FastAPI application instance
# -------------------------------------------------
app = FastAPI()


# -------------------------------------------------
# Security (API key via Render environment variable)
# -------------------------------------------------
# This value is set in Render → Environment → API_KEY
API_KEY = os.getenv("API_KEY", "").strip()


def require_key(x_api_key: Optional[str]):
    """
    Simple API key validation.
    Every protected endpoint must include:
    Header: X-Api-Key = <secret key>
    """

    # If API_KEY is not set, allow requests (useful for local testing)
    if not API_KEY:
        return

    # Reject request if key is missing or incorrect
    if not x_api_key or x_api_key.strip() != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


# -------------------------------------------------
# Load ML metadata and models once at startup
# -------------------------------------------------
try:
    # Load metadata containing the embedding model name
    META = joblib.load("assessment_meta.joblib")
    EMBED_MODEL_NAME = META["embed_model"]

    # Load sentence embedding model
    embedder = SentenceTransformer(EMBED_MODEL_NAME)

    # Load trained regression model
    model = joblib.load("assessment_model.joblib")

except Exception as ex:
    # Fail fast with a clear error if model files are missing
    # This message will appear in Render logs
    raise RuntimeError(
        "Failed to load assessment model files. Ensure assessment_meta.joblib and "
        "assessment_model.joblib are present in the app root on Render."
    ) from ex


# -------------------------------------------------
# Data Transfer Objects (DTOs)
# -------------------------------------------------
class AssessRequest(BaseModel):
    """Request body for /assess"""
    text: str
    userId: str
    chatSessionId: int


class QuickReply(BaseModel):
    """Represents a UI quick-reply button"""
    label: str
    value: str


class AssessResponse(BaseModel):
    """Response returned by /assess"""
    distress: float
    stress: float
    risk: float
    depression: float
    confidence: float
    explanations: List[str]

    # AI-directed next step
    nextAction: str                    # support | checkin_scored | refer_professional | escalate
    nextKey: Optional[str] = None      # Used for scored check-in questions
    professionalSummary: str
    quickReplies: List[QuickReply]


class EmbedRequest(BaseModel):
    """Request body for /embed"""
    text: str


class EmbedResponse(BaseModel):
    """Response returned by /embed"""
    embedding: List[float]
    model: str


# -------------------------------------------------
# Utility / Helper functions
# -------------------------------------------------
@app.get("/health")
def health():
    """
    Health-check endpoint.
    Used by Render and for quick uptime testing.
    Does NOT require API key.
    """
    return {"status": "ok", "embed_model": EMBED_MODEL_NAME}


def get_embedding(text: str) -> np.ndarray:
    """
    Converts input text into a normalized embedding vector.
    Normalization ensures cosine similarity == dot product.
    """
    v = embedder.encode([text], normalize_embeddings=True)[0]
    return np.asarray(v, dtype=np.float32)


def decide_next_action(
    distress: float,
    stress: float,
    risk: float,
    depression: float,
    confidence: float,
    text: str
):
    """
    Policy layer that decides what the AI should do next
    based on model predictions and simple rule-based logic.
    """
    t = text.lower().strip()

    # High-risk or explicit self-harm language → escalate
    if risk >= 0.75 or "kill myself" in t or "suicide" in t or "hurt myself" in t:
        return "escalate", None

    # Medium risk or low confidence → ask scored check-in
    if risk >= 0.35 or distress >= 0.70 or confidence < 0.55:
        if risk >= 0.45:
            return "checkin_scored", "q9_safety"
        if depression >= 0.60:
            return "checkin_scored", "low_mood"
        if stress >= 0.60 or distress >= 0.70:
            return "checkin_scored", "sleep"
        return "checkin_scored", "low_mood"

    # Sustained depression → professional referral
    if depression >= 0.75:
        return "refer_professional", None

    # Default supportive response
    return "support", None


# -------------------------------------------------
# Embedding endpoint
# -------------------------------------------------
@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest, x_api_key: Optional[str] = Header(default=None)):
    """
    Generates a vector embedding for a given text.
    Used for similarity search or semantic analysis.
    """
    require_key(x_api_key)

    t = (req.text or "").strip()
    if not t:
        return EmbedResponse(embedding=[], model=EMBED_MODEL_NAME)

    vec = get_embedding(t)
    return EmbedResponse(
        embedding=vec.astype(float).tolist(),
        model=EMBED_MODEL_NAME
    )


# -------------------------------------------------
# Assessment endpoint
# -------------------------------------------------
@app.post("/assess", response_model=AssessResponse)
def assess(req: AssessRequest, x_api_key: Optional[str] = Header(default=None)):
    """
    Main inference endpoint.
    Predicts emotional state and determines next action.
    """
    require_key(x_api_key)

    t = (req.text or "").strip()

    # Handle empty input safely
    if not t:
        distress = stress = risk = depression = 0.0
        confidence = 0.30
        explanations = ["empty_text"]
        nextAction, nextKey = "support", None

    else:
        # Convert text to embedding and run regression model
        emb = get_embedding(t).reshape(1, -1)
        preds = model.predict(emb)[0]

        # Clamp predictions to [0,1]
        distress = float(np.clip(preds[0], 0.0, 1.0))
        stress = float(np.clip(preds[1], 0.0, 1.0))
        risk = float(np.clip(preds[2], 0.0, 1.0))
        depression = float(np.clip(preds[3], 0.0, 1.0))

        # Confidence score depends on input length
        confidence = 0.85 if len(t) >= 10 else 0.55

        explanations = ["embedding_regression"]
        nextAction, nextKey = decide_next_action(
            distress, stress, risk, depression, confidence, t
        )

    # Generate UI quick-reply buttons
    if nextAction == "checkin_scored":
        quickReplies = [
            QuickReply(label="0 - Not at all", value="0"),
            QuickReply(label="1 - Several days", value="1"),
            QuickReply(label="2 - More than half the days", value="2"),
            QuickReply(label="3 - Nearly every day", value="3"),
        ]
    elif nextAction == "escalate":
        quickReplies = [
            QuickReply(label="I'm safe right now", value="I'm safe right now"),
            QuickReply(label="I need urgent help", value="I need urgent help"),
        ]
    else:
        quickReplies = [
            QuickReply(label="Tell me more", value="Tell me more"),
            QuickReply(label="Can we do a check-in?", value="Can we do a check-in?"),
        ]

    # Human-readable summary (useful for professionals / logs)
    professionalSummary = (
        f"Signals: distress={distress:.2f}, stress={stress:.2f}, "
        f"risk={risk:.2f}, depression={depression:.2f}, conf={confidence:.2f}. "
        f"NextAction={nextAction}."
    )

    return AssessResponse(
        distress=distress,
        stress=stress,
        risk=risk,
        depression=depression,
        confidence=confidence,
        explanations=explanations,
        nextAction=nextAction,
        nextKey=nextKey,
        professionalSummary=professionalSummary,
        quickReplies=quickReplies
    )
