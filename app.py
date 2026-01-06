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
    description="Robust intent detection with multi-command & platform-aware search",
    version="1.4.0"
)

class InputText(BaseModel):
    text: str

# ==================================================
# ML FALLBACK (OPTIONAL)
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

    # explicit platform
    for platform, keywords in SEARCH_PLATFORMS.items():
        if any(k in text for k in keywords):
            return platform

    # files
    if any(ext in query for ext in FILE_EXTENSIONS):
        return "files"
    if any(w in query for w in ["file", "folder", "directory"]):
        return "files"

    # apps → windows search
    if any(app in query for app in KNOWN_APPS):
        return "windows_search"

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

    for token in doc:
        if token.lemma_ in COMMAND_VERBS:
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

    if action == "search":
        platform = detect_platform(chunk, joined_targets)
        clean_query = re.sub(
            r"\b(on|in)?\s*(youtube|google|browser|files?|folders?)\b",
            "",
            joined_targets
        ).strip()

        results.append({
            "action": "search",
            "query": clean_query,
            "platform": platform
        })
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
# 🔥 FIXED INTENT DETECTION (IMPORTANT)
# ==================================================
def detect_intent(text: str):
    text = text.lower()

    # command first (strong rule-based)
    for verb in COMMAND_VERBS:
        if re.search(rf"\b{verb}\b", text):
            return "command"

    # question
    for word in QUESTION_WORDS:
        if re.search(rf"\b{word}\b", text):
            return "question"

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
    intent = detect_intent(text)
    doc = nlp(text)

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

        return {
            "intent": "command",
            "tasks": tasks,
            "confidence": 1.0
        }

    # ---------- CHAT ----------
    return {
        "intent": "chat",
        "confidence": 1.0
    }
