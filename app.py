from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import os

# ======================================================
# API KEY (set as environment variable)
# ======================================================
# Windows:
#   setx INTENT_AI_KEY "my_secret_key_123"
# Linux/macOS:
#   export INTENT_AI_KEY="my_secret_key_123"

API_KEY = os.getenv("INTENT_AI_KEY")

# ======================================================
# FASTAPI APP
# ======================================================
app = FastAPI(title="Intent Classification AI")

class InputText(BaseModel):
    text: str

# ======================================================
# TRAINING DATA (SMALL REAL AI DATASET)
# ======================================================
sentences = [
    "open chrome",
    "open chatgpt",
    "close chrome",
    "close vscode",
    "search python",
    "search chatgpt",
    "find project folder",
    "find drive",
    "copy text",
    "paste",
    "delete file",
    "save document",
    "minimize window",
    "maximize window",
    "restore window",
    "what is python",
    "how to create google account",
    "hello",
    "thank you"
]

labels = [
    "open",
    "open",
    "close",
    "close",
    "search",
    "search",
    "find",
    "find",
    "copy",
    "paste",
    "delete",
    "save",
    "minimize",
    "maximize",
    "restore",
    "explain",
    "explain",
    "chat",
    "chat"
]

# ======================================================
# TRAIN ML MODEL (THIS IS YOUR AI)
# ======================================================
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(sentences)

model = LogisticRegression()
model.fit(X, labels)

# ======================================================
# ENTITY EXTRACTION
# ======================================================
def extract_entities(task, text):
    words = text.split()
    rest = " ".join(words[1:]) if len(words) > 1 else ""

    if task == "search":
        return {"query": rest, "source": "google"}

    if task == "find":
        return {"target": rest, "scope": "local"}

    if task == "explain":
        return {"topic": rest}

    if task in ["open", "close", "minimize", "maximize", "restore"]:
        return {"target": rest}

    if task in ["copy", "delete", "save"]:
        return {"object": rest}

    if task == "paste":
        return {"destination": "current_context"}

    return {}

# ======================================================
# INTENT CLASSIFICATION ENDPOINT (API KEY PROTECTED)
# ======================================================
@app.post("/intent")
def classify_intent(
    data: InputText,
    x_api_key: str = Header(None)
):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    text = data.text.lower().strip()

    task = model.predict(vectorizer.transform([text]))[0]

    if task == "explain":
        intent = "question"
    elif task == "chat":
        intent = "chat"
    else:
        intent = "command"

    return {
        "intent": intent,
        "task": task if intent != "chat" else None,
        "entities": extract_entities(task, text)
    }
