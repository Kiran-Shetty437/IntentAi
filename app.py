import re
import os
import spacy
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

# ==================================================
# CONFIG
# ==================================================
API_KEY = os.getenv("INTENT_AI_KEY")

QUESTION_WORDS = {"how", "what", "why", "where", "when", "who", "which"}

COMMAND_VERBS = {
    "open", "close", "search", "find",
    "copy", "paste", "delete", "save",
    "launch", "start", "exit", "quit",
    "minimize", "maximize", "restore",
    "focus"
}

# kept (not removed)
SEARCH_PLATFORMS = {
    "youtube": {"youtube", "yt"},
    "google": {"google", "web", "browser"},
    "files": {"file", "files", "folder", "directory"},
}

KNOWN_APPS = {
    "vscode", "chrome", "whatsapp", "chatgpt",
    "notepad", "calculator", "spotify", "edge",
    "word", "excel", "powerpoint", "youtube", "google"
}

FILE_EXTENSIONS = {
    ".txt", ".pdf", ".doc", ".docx", ".xls", ".xlsx",
    ".ppt", ".pptx", ".jpg", ".png", ".mp4", ".zip",
    ".py", ".js", ".java", ".cpp"
}

# ==================================================
# LOAD NLP
# ==================================================
nlp = spacy.load("en_core_web_sm")

# ==================================================
# FASTAPI APP
# ==================================================
app = FastAPI(
    title="IntentAI – Explicit Platform Aware API",
    description="Multi-command, multi-task, multi-target intent engine",
    version="3.0.0"
)

class InputText(BaseModel):
    text: str

# ==================================================
# HELPERS
# ==================================================
def normalize(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower()).strip()

def normalize_app(name: str) -> str:
    aliases = {
        "vs code": "vscode",
        "vs studio": "vscode",
        "visual studio": "vscode",
        "visual studio code": "vscode",
        "google chrome": "chrome",
        "chrome browser": "chrome",
        "chat gpt": "chatgpt",
        "whats app": "whatsapp",
        "what's app": "whatsapp"
    }
    return aliases.get(name.strip(), name.strip())

# 🔥 NEW: extract explicit "on <platform>"
def extract_explicit_platform(text: str):
    match = re.search(r"\bon\s+([a-zA-Z0-9_ ]+)", text)
    if not match:
        return None
    return normalize_app(match.group(1).strip())

# ==================================================
# INTENT DETECTION (UNCHANGED)
# ==================================================
def detect_intent(text: str):
    for verb in COMMAND_VERBS:
        if re.search(rf"\b{verb}\b", text):
            return "command"

    for q in QUESTION_WORDS:
        if re.search(rf"\b{q}\b", text):
            return "question"

    return "chat"

# ==================================================
# TASK SPLITTING (UNCHANGED)
# ==================================================
def split_tasks(text: str):
    pattern = r"\b(" + "|".join(COMMAND_VERBS) + r")\b"
    matches = list(re.finditer(pattern, text))

    if not matches:
        return [text]

    chunks = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunks.append(text[start:end].strip())

    return chunks


# TASK PARSER (EXTENDED, NOT REPLACED)
def parse_task(chunk: str):
    doc = nlp(chunk)
    action = None

    for token in doc:
        if token.lemma_ in COMMAND_VERBS:
            action = token.lemma_
            break

    if not action:
        return []

    if action == "paste":
        return [{"action": "paste"}]

    parts = chunk.split(action, 1)
    if len(parts) < 2:
        return [{"action": action, "needs_clarification": True}]

    raw_target_text = parts[1]

    # comma + and logic (unchanged)
    raw_targets = re.split(r"\s*,\s*|\s+and\s+", raw_target_text)
    targets = [t.strip() for t in raw_targets if t.strip()]

    if not targets:
        return [{"action": action, "needs_clarification": True}]

    results = []

    # ---------- SEARCH (UPDATED RULE) ----------
    if action == "search":
        joined_targets = " ".join(targets)

        # 🔥 NEW RULE: explicit platform ONLY
        platform = extract_explicit_platform(chunk)

        clean_query = joined_targets

        if platform:
            clean_query = re.sub(
                rf"\bon\s+{re.escape(platform)}\b",
                "",
                clean_query
            ).strip()

        results.append({
            "action": "search",
            "query": clean_query,
            "platform": platform  # None if not specified
        })

    # ---------- OTHER COMMANDS (UNCHANGED) ----------
    else:
        for t in targets:
            results.append({
                "action": action,
                "target": normalize_app(t)
            })

    return results

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

    raw_text = data.text.lower()
    clean_text = normalize(data.text)

    intent = detect_intent(clean_text)

    # ---------- QUESTION ----------
    if intent == "question":
        return {
            "intent": "question",
            "task": "explain",
            "entities": {"topic": clean_text},
            "confidence": 1.0
        }

    # ---------- COMMAND ----------
    if intent == "command":
        chunks = split_tasks(raw_text)
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

        if not tasks:
            return {
                "intent": "clarification",
                "ask": "What do you want to do?",
                "confidence": 1.0
            }

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
