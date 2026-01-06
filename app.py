import re
import os
import spacy
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ==================================================
# CONFIG
# ==================================================
API_KEY = os.getenv("INTENT_AI_KEY")

QUESTION_WORDS = {"how", "what", "why", "where", "when", "who", "which"}

COMMAND_VERBS = {
    "open", "close", "search", "find",
    "copy", "paste", "delete", "save",
    "launch", "start", "exit", "quit"
}

SEARCH_PLATFORMS = {
    "youtube": {"youtube", "yt"},
    "google": {"google", "web", "browser"},
    "files": {"file", "files", "folder", "directory"},
}

KNOWN_APPS = {
    "vscode", "chrome", "whatsapp", "chatgpt",
    "notepad", "calculator", "spotify", "edge",
    "word", "excel", "powerpoint", "youtube"
}

FILE_EXTENSIONS = {
    ".txt", ".pdf", ".doc", ".docx", ".xls", ".xlsx",
    ".ppt", ".pptx", ".jpg", ".png", ".mp4", ".zip"
}

# ==================================================
# LOAD NLP
# ==================================================
nlp = spacy.load("en_core_web_sm")

# ==================================================
# FASTAPI APP
# ==================================================
app = FastAPI(
    title="IntentAI – NLP Hybrid API",
    description="Intent detection with robust multi-command & platform-aware search",
    version="1.3.0"
)

class InputText(BaseModel):
    text: str

# ==================================================
# ML FALLBACK MODEL
# ==================================================
train_data = [
    ("hello", "chat"),
    ("hi", "chat"),
    ("thank you", "chat"),
    ("what is python", "question"),
    ("open chrome", "command"),
    ("search python", "command"),
]

X = [x for x, _ in train_data]
y = [y for _, y in train_data]

vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

model = LogisticRegression()
model.fit(X_vec, y)

# ==================================================
# HELPERS
# ==================================================
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

def detect_platform(text: str, query: str = "") -> str:
    text = text.lower()
    query = query.lower()

    # 1️⃣ Explicit platform
    for platform, keywords in SEARCH_PLATFORMS.items():
        for k in keywords:
            if k in text:
                return platform

    # 2️⃣ File / folder
    if any(ext in query for ext in FILE_EXTENSIONS):
        return "files"
    if any(w in query for w in ["file", "folder", "directory"]):
        return "files"

    # 3️⃣ App → Windows search
    for app in KNOWN_APPS:
        if app in query:
            return "windows_search"

    # 4️⃣ Default
    return "google"

def split_tasks(text: str):
    pattern = r"\b(" + "|".join(COMMAND_VERBS) + r")\b"
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

    # 🔥 FIX: POS-based target extraction (NOT dependency-based)
    for token in doc:
        if token.pos_ == "VERB" and token.lemma_ in COMMAND_VERBS:
            action = token.lemma_

        if token.pos_ in ("NOUN", "PROPN") and not token.is_stop:
            targets.append(token.text)

    if not action:
        return []

    if action == "paste":
        return [{"action": "paste"}]

    if not targets:
        return [{"action": action, "needs_clarification": True}]

    joined_targets = " ".join(targets)
    results = []

    # ---------- SEARCH ----------
    if action == "search":
        platform = detect_platform(chunk, joined_targets)

        clean_query = re.sub(
            r"\b(on|in)?\s*(youtube|google|web|browser|files?|folders?)\b",
            "",
            joined_targets
        ).strip()

        results.append({
            "action": "search",
            "query": clean_query,
            "platform": platform
        })

    # ---------- OTHER COMMANDS ----------
    else:
        for t in re.split(r"\band\b|,", joined_targets):
            t = t.strip()
            if t:
                results.append({
                    "action": action,
                    "target": normalize_app(t)
                })

    return results

# ==================================================
# INTENT DETECTION
# ==================================================
def detect_intent(doc):
    for token in doc:
        if token.lower_ in QUESTION_WORDS:
            return "question"

    for token in doc:
        if token.pos_ == "VERB" and token.lemma_ in COMMAND_VERBS:
            return "command"

    return "chat"

# ==================================================
# API ENDPOINT
# ==================================================
@app.post("/intent")
def classify_intent(
    data: InputText,
    x_api_key: str = Header(None)
):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    text = normalize(data.text)
    doc = nlp(text)
    intent = detect_intent(doc)

    # ---------- QUESTION ----------
    if intent == "question":
        return {
            "intent": "question",
            "task": "explain",
            "entities": {"topic": text},
            "confidence": 1.0
        }

    # ---------- COMMAND ----------
    if intent == "command":
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

    # ---------- CHAT ----------
    if intent == "chat":
        return {
            "intent": "chat",
            "confidence": 1.0
        }

    # ---------- ML FALLBACK ----------
    vec = vectorizer.transform([text])
    probs = model.predict_proba(vec)[0]
    ml_intent = model.classes_[probs.argmax()]
    confidence = probs.max()

    if confidence < 0.6:
        return {
            "intent": "unknown",
            "ask": "I didn't understand. Can you rephrase?"
        }

    return {
        "intent": ml_intent,
        "confidence": round(confidence, 2)
    }
