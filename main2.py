from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pdfplumber
import docx
import sqlite3
import os
import uuid

app = FastAPI()

# Allow frontend to access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For MVP only
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure data folder exists
DATA_FOLDER = "data"
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

# Setup SQLite database
DB_FILE = os.path.join(DATA_FOLDER, "resumes.db")
conn = sqlite3.connect(DB_FILE, check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS resumes (
    id TEXT PRIMARY KEY,
    filename TEXT,
    chunk_index INTEGER,
    chunk_text TEXT
)
""")
conn.commit()

# In-memory sessions
sessions = {}

# --- Models ---
class StartSessionRequest(BaseModel):
    resume_id: str
    role: str

# --- Helpers ---
def extract_text(file: UploadFile):
    ext = file.filename.split('.')[-1].lower()
    text = ""
    if ext == "pdf":
        with pdfplumber.open(file.file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
    elif ext in ["docx", "doc"]:
        doc = docx.Document(file.file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    return text


def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i+chunk_size]))
    return chunks

# --- Endpoints ---
@app.post("/upload_resume")
async def upload_resume(file: UploadFile = File(...)):
    text = extract_text(file)
    chunks = chunk_text(text)
    resume_id = str(uuid.uuid4())

    # Save chunks to DB
    for idx, chunk in enumerate(chunks):
        cursor.execute(
            "INSERT INTO resumes (id, filename, chunk_index, chunk_text) VALUES (?, ?, ?, ?)",
            (resume_id, file.filename, idx, chunk)
        )
    conn.commit()

    return {"status": "success", "resume_id": resume_id, "chunks": len(chunks)}


@app.post("/start_session")
async def start_session(req: StartSessionRequest):
    """
    Start a Q&A session:
    - Store the selected role
    - Retrieve resume chunks
    - Generate first question
    """
    cursor.execute(
        "SELECT chunk_text FROM resumes WHERE id = ? ORDER BY chunk_index",
        (req.resume_id,)
    )
    rows = cursor.fetchall()
    if not rows:
        return {"status": "error", "message": "Resume not found"}

    chunks = [row[0] for row in rows]

    # Create a session
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "resume_id": req.resume_id,
        "role": req.role,
        "chunks": chunks,
        "current_question": 0,
        "answers": []
    }

    # Placeholder first question
    first_question = f"Hi! You want the role of {req.role}. Can you briefly introduce yourself?"

    return {"status": "success", "session_id": session_id, "question": first_question}
