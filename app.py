import re
import os
import spacy
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# CONFIG

API_KEY = os.getenv("INTENT_AI_KEY")  # set in Render / OS env
ACTIONS = ["open", "close", "search", "find", "lookup", "copy", "paste"]


# LOAD NLP MODEL

# LOAD NLP MODEL (safe for Render / cloud)

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")



# FASTAPI APP

app = FastAPI(title="Hybrid NLP Intent AI")

class InputText(BaseModel):
    text: str


# ML FALLBACK MODEL

train_data = [
    ("hello", "chat"),
    ("hi", "chat"),
    ("thank you", "chat"),

    ("what is python", "question"),
    ("why we use python", "question"),
    ("how to install python", "question"),

    ("open chrome", "command"),
    ("close chrome", "command"),
    ("search python", "command"),
]

X = [x for x, _ in train_data]
y = [y for _, y in train_data]

vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

model = LogisticRegression()
model.fit(X_vec, y)

# HELPERS
def normalize(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower()).strip()

def normalize_app(name: str) -> str:
    aliases = {
        "vs studio": "vscode",
        "vs code": "vscode",
        "visual studio": "vscode",
        "visual studio code": "vscode",
        "google chrome": "chrome",
        "chrome browser": "chrome",
        "chat gpt": "chatgpt",
        "whats app": "whatsapp",
        "what's app": "whatsapp"
    }
    return aliases.get(name, name)

def split_tasks(text: str):
    pattern = r"\b(" + "|".join(ACTIONS) + r")\b"
    matches = list(re.finditer(pattern, text))
    chunks = []

    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunks.append(text[start:end].strip())

    return chunks

def parse_task(chunk: str):
    doc = nlp(chunk)
    action = None
    targets = []

    for token in doc:
        if token.pos_ == "VERB" and token.lemma_ in ACTIONS:
            action = token.lemma_
        if token.dep_ in ("dobj", "pobj"):
            targets.append(token.text)

    if not action:
        return []

    if action == "paste":
        return [{"action": "paste"}]

    if not targets or targets[0] in ("it", "app"):
        return [{"action": action, "needs_clarification": True}]

    results = []
    for t in re.split(r"\band\b|,", " ".join(targets)):
        t = t.strip()
        if t:
            results.append({
                "action": action,
                "target": normalize_app(t)
            })

    return results


# API ENDPOINT

@app.post("/intent")
def classify_intent(
    data: InputText,
    x_api_key: str = Header(None)
):
    # ---- API KEY CHECK ----
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    text = normalize(data.text)

    # ---- CHAT ----
    if text in (
        "hi", "hello", "hey",
        "how are you", "how was your day",
        "what are you doing"
    ):
        return {"intent": "chat", "confidence": 1.0}

    # ---- QUESTIONS ----
    if text.startswith(("how", "why", "what")) and not re.search(
        r"\b(" + "|".join(ACTIONS) + r")\b", text
    ):
        return {
            "intent": "question",
            "task": "explain",
            "entities": {"topic": text},
            "confidence": 1.0
        }

    # ---- COMMANDS (MULTI-TASK) ----
    if re.search(r"\b(" + "|".join(ACTIONS) + r")\b", text):
        chunks = split_tasks(text)
        tasks = []

        for chunk in chunks:
            parsed = parse_task(chunk)
            for t in parsed:
                if t.get("needs_clarification"):
                    return {
                        "intent": "clarification",
                        "task": t["action"],
                        "ask": f"What do you want to {t['action']}?",
                        "confidence": 1.0
                    }
                tasks.append(t)

        if tasks:
            return {
                "intent": "command",
                "tasks": tasks,
                "confidence": 1.0
            }

    # ---- ML FALLBACK ----
    vec = vectorizer.transform([text])
    probs = model.predict_proba(vec)[0]
    intent = model.classes_[probs.argmax()]
    confidence = probs.max()

    if confidence < 0.6:
        return {
            "intent": "unknown",
            "ask": "I didn't understand. Can you rephrase?"
        }

    return {
        "intent": intent,
        "confidence": round(confidence, 2)
    }
