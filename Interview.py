from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pdfplumber
import docx
import sqlite3
import os
import uuid
import json
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, HTTPException

from fastapi.responses import StreamingResponse
import io
import tempfile

from openai import OpenAI
from dotenv import load_dotenv

# -------------------- ENV --------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------------------- APP --------------------
router = APIRouter(
    prefix="/interview",
    tags=["Interview"]
)

router.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://nephele-frontend.vercel.app"
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- STORAGE --------------------
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

DB_PATH = os.path.join(DATA_DIR, "resumes.db")
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

# -------------------- TABLES --------------------
cursor.execute("""
CREATE TABLE IF NOT EXISTS resumes (
    resume_id TEXT,
    filename TEXT,
    chunk_index INTEGER,
    chunk_text TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS interviews (
    interview_id TEXT PRIMARY KEY,
    resume_id TEXT,
    candidate_name TEXT,
    role TEXT,
    score INTEGER,
    hire_recommendation TEXT,
    conversation TEXT,
    created_at TEXT
)
""")

conn.commit()

# -------------------- MEMORY SESSIONS --------------------
sessions = {}

# -------------------- MODELS --------------------
class StartSessionRequest(BaseModel):
    resume_id: str
    role: str
    candidate_name: str


class SubmitAnswerRequest(BaseModel):
    session_id: str
    answer: str

# -------------------- HELPERS --------------------
def extract_text_from_file(filename: str, file_obj) -> str:
    ext = filename.split(".")[-1].lower()
    text = ""

    if ext == "pdf":
        with pdfplumber.open(file_obj) as pdf:
            for page in pdf.pages:
                if page.extract_text():
                    text += page.extract_text() + "\n"

    elif ext in ["docx", "doc"]:
        document = docx.Document(file_obj)
        for p in document.paragraphs:
            text += p.text + "\n"

    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    return text.strip()


def chunk_text(text: str, chunk_size: int = 500):
    words = text.split()
    return [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]


def generate_ai_question(role, resume_chunks, conversation):
    prompt = f"""
You are a professional technical interviewer.

Role: {role}

Resume context:
{resume_chunks}

Conversation so far:
{conversation}

Rules:
- Ask the NEXT interview question
- Do NOT repeat previous questions
- Ask only ONE question
- Keep it concise and role-relevant
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    return response.choices[0].message.content.strip()


def generate_feedback(role, conversation):
    prompt = f"""
You are an ATS interview evaluator.

Role: {role}

Interview conversation:
{conversation}

Return ONLY valid JSON in this format:

{{
  "score": number,
  "strengths": [string, string],
  "weaknesses": [string, string],
  "improvements": [string, string],
  "hire_recommendation": "Yes" | "Maybe" | "No"
}}

Rules:
- No markdown
- No explanation
- Strict JSON only
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message.content.strip()

# =====================================================
# ==================== STAGE 1 ========================
# =====================================================
@router.post("/upload_resume")
async def upload_resume(file: UploadFile = File(...)):
    resume_id = str(uuid.uuid4())

    resume_dir = os.path.join(DATA_DIR, resume_id)
    os.makedirs(resume_dir, exist_ok=True)

    file_path = os.path.join(resume_dir, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    with open(file_path, "rb") as f:
        text = extract_text_from_file(file.filename, f)

    chunks = chunk_text(text)

    for idx, chunk in enumerate(chunks):
        cursor.execute(
            """
            INSERT INTO resumes (resume_id, filename, chunk_index, chunk_text)
            VALUES (?, ?, ?, ?)
            """,
            (resume_id, file.filename, idx, chunk)
        )

    conn.commit()

    return {
        "status": "success",
        "resume_id": resume_id,
        "chunks": len(chunks)
    }

# =====================================================
# ==================== STAGE 2 ========================
# =====================================================
@router.post("/start_session")
async def start_session(req: StartSessionRequest):
    cursor.execute(
        """
        SELECT chunk_text
        FROM resumes
        WHERE resume_id = ?
        ORDER BY chunk_index
        """,
        (req.resume_id,)
    )
    rows = cursor.fetchall()

    if not rows:
        raise HTTPException(status_code=404, detail="Resume not found")

    chunks = [row[0] for row in rows]

    session_id = str(uuid.uuid4())
    first_question = f"Hi! You chose {req.role}. Can you briefly introduce yourself?"

    sessions[session_id] = {
        "resume_id": req.resume_id,
        "candidate_name": req.candidate_name,
        "role": req.role,
        "chunks": chunks,
        "conversation": [],
        "current_question": 0,
        "last_question": first_question
    }

    print("🟢 SESSION CREATED:", session_id)

    return {
        "status": "success",
        "session_id": session_id,
        "question": first_question
    }

# =====================================================
# ==================== STAGE 3–6 ======================
# =====================================================
@router.post("/submit_answer")
async def submit_answer(req: SubmitAnswerRequest):
    session = sessions.get(req.session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    MAX_QUESTIONS = 5

    # Save answer
    session["conversation"].append({
        "question": session["last_question"],
        "answer": req.answer
    })

    session["current_question"] += 1
    print("🧮 QUESTION COUNT:", session["current_question"])

    # ---------- INTERVIEW COMPLETE ----------
    if session["current_question"] >= MAX_QUESTIONS:
        raw_feedback = generate_feedback(
            role=session["role"],
            conversation=session["conversation"]
        )

        try:
            feedback = json.loads(raw_feedback)
        except json.JSONDecodeError:
            feedback = {
                "score": None,
                "strengths": [],
                "weaknesses": [],
                "improvements": [],
                "hire_recommendation": "Unknown",
                "raw": raw_feedback
            }

        # ✅ SAVE INTERVIEW TO DB
        interview_id = str(uuid.uuid4())

        cursor.execute(
            """
            INSERT INTO interviews (
                interview_id,
                resume_id,
                candidate_name,
                role,
                score,
                hire_recommendation,
                conversation,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                interview_id,
                session["resume_id"],
                session["candidate_name"],
                session["role"],
                feedback.get("score"),
                feedback.get("hire_recommendation"),
                json.dumps(session["conversation"]),
                datetime.utcnow().isoformat()
            )
        )

        conn.commit()

        return {
            "status": "completed",
            "interview_id": interview_id,
            "feedback": feedback
        }

    # ---------- NEXT AI QUESTION ----------
    try:
        next_question = generate_ai_question(
            role=session["role"],
            resume_chunks=session["chunks"][:3],
            conversation=session["conversation"]
        )
    except Exception as e:
        print("❌ AI ERROR:", e)
        next_question = "Can you describe a recent project you worked on?"

    session["last_question"] = next_question

    return {
        "status": "success",
        "question": next_question
    }


@router.post("/tts")
async def text_to_speech(payload: dict):
    text = payload.get("text")

    if not text:
        raise HTTPException(status_code=400, detail="Text is required")

    try:
        response = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=text
        )

        audio_bytes = response.read()

        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type="audio/mpeg"
        )

    except Exception as e:
        print("❌ TTS ERROR:", e)
        raise HTTPException(status_code=500, detail="TTS generation failed")


@router.post("/stt")
async def speech_to_text(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="Audio file required")

    try:
        # Save temp audio file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name

        # Transcribe
        with open(tmp_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )

        os.remove(tmp_path)

        return {
            "text": transcript.text
        }

    except Exception as e:
        print("❌ STT ERROR:", e)
        raise HTTPException(status_code=500, detail="Speech-to-text failed")
