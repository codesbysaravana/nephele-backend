import hashlib
import os
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List
from uuid import uuid4

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi import APIRouter

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pydantic_settings import BaseSettings

from PyPDF2 import PdfReader
from docx import Document
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

# ======================================================
# Configuration
# ======================================================
class Settings(BaseSettings):
    DB_PATH: str = "data/documents.db"
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    MAX_FILE_SIZE: int = 5 * 1024 * 1024  # 5 MB
    LESSON_SIZE: int = 5  # chunks per lesson


settings = Settings()
Path(settings.DB_PATH).parent.mkdir(parents=True, exist_ok=True)

# ======================================================
# OpenAI Clients
# ======================================================
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
tts_client = OpenAI()

# ======================================================
# Database
# ======================================================
def get_db():
    return sqlite3.connect(settings.DB_PATH)


def init_db():
    with get_db() as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            filename TEXT,
            file_hash TEXT,
            chunk_index INTEGER,
            content TEXT,
            created_at TEXT
        );

        CREATE TABLE IF NOT EXISTS lessons (
            id TEXT PRIMARY KEY,
            filename TEXT,
            title TEXT,
            lesson_index INTEGER,
            created_at TEXT
        );

        CREATE TABLE IF NOT EXISTS lesson_chunks (
            lesson_id TEXT,
            chunk_id TEXT,
            chunk_order INTEGER
        );

        CREATE TABLE IF NOT EXISTS teaching_content (
            id TEXT PRIMARY KEY,
            lesson_id TEXT,
            teaching_text TEXT,
            created_at TEXT
        );
        """)


init_db()

# ======================================================
# Prompts
# ======================================================
TEACHING_PROMPT = """
You are an expert teacher.

Transform the following lesson into clear teaching content:
- Explain concepts simply
- Use short paragraphs
- Use examples when helpful
- Avoid copying text verbatim
- Keep it suitable for spoken narration

Lesson:
---
{lesson_text}
---
"""

# ======================================================
# Models
# ======================================================
class IngestResponse(BaseModel):
    filename: str
    chunks_created: int

# ======================================================
# Utility Helpers
# ======================================================
def clean_text(text: str) -> str:
    return "\n".join(
        line.strip()
        for line in text.replace("\r", "\n").splitlines()
        if line.strip()
    )


def file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    if size <= overlap:
        raise ValueError("CHUNK_SIZE must be larger than CHUNK_OVERLAP")

    chunks, start = [], 0
    while start < len(text):
        end = start + size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap
    return chunks


def assemble_lesson_text(lesson_id: str) -> str:
    with get_db() as db:
        rows = db.execute(
            """
            SELECT c.content
            FROM lesson_chunks lc
            JOIN chunks c ON c.id = lc.chunk_id
            WHERE lc.lesson_id = ?
            ORDER BY lc.chunk_order
            """,
            (lesson_id,),
        ).fetchall()

    return "\n\n".join(row[0] for row in rows)

# ======================================================
# File Loaders
# ======================================================
def load_txt(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def load_pdf(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_docx(path: str) -> str:
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs)


def load_file_by_type(path: str, suffix: str) -> str:
    if suffix == "txt":
        return load_txt(path)
    if suffix == "pdf":
        return load_pdf(path)
    if suffix == "docx":
        return load_docx(path)
    raise HTTPException(415, "Unsupported file type")

# ======================================================
# Teaching & Audio
# ======================================================
def generate_teaching_content(lesson_text: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful teacher."},
            {"role": "user", "content": TEACHING_PROMPT.format(lesson_text=lesson_text)},
        ],
        temperature=0.4,
    )
    return response.choices[0].message.content.strip()


def audio_stream(text: str):
    with tts_client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=text,
    ) as response:
        for chunk in response.iter_bytes():
            yield chunk

# ======================================================
# Lesson Creation
# ======================================================
def create_lessons(filename: str):
    with get_db() as db:
        chunks = db.execute(
            """
            SELECT id
            FROM chunks
            WHERE filename = ?
            ORDER BY chunk_index
            """,
            (filename,),
        ).fetchall()

        for i in range(0, len(chunks), settings.LESSON_SIZE):
            lesson_id = str(uuid4())
            lesson_index = i // settings.LESSON_SIZE

            db.execute(
                "INSERT INTO lessons VALUES (?, ?, ?, ?, ?)",
                (
                    lesson_id,
                    filename,
                    f"Lesson {lesson_index + 1}",
                    lesson_index,
                    datetime.utcnow().isoformat(),
                ),
            )

            for order, (chunk_id,) in enumerate(chunks[i:i + settings.LESSON_SIZE]):
                db.execute(
                    "INSERT INTO lesson_chunks VALUES (?, ?, ?)",
                    (lesson_id, chunk_id, order),
                )

# ======================================================
# ROUTER
# ======================================================
router = APIRouter()


# ======================================================
# Ingest Endpoint
# ======================================================
@router.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "Empty filename")

    data = await file.read()
    if len(data) > settings.MAX_FILE_SIZE:
        raise HTTPException(413, "File too large")

    digest = file_hash(data)

    with get_db() as db:
        if db.execute(
            "SELECT 1 FROM chunks WHERE file_hash = ? LIMIT 1",
            (digest,),
        ).fetchone():
            raise HTTPException(409, "Document already ingested")

    suffix = file.filename.split(".")[-1].lower()

    fd, path = tempfile.mkstemp(suffix="." + suffix)
    os.close(fd)

    try:
        Path(path).write_bytes(data)
        raw_text = load_file_by_type(path, suffix)
    finally:
        try:
            Path(path).unlink()
        except PermissionError:
            pass

    text = clean_text(raw_text)
    if not text:
        raise HTTPException(400, "No extractable text")

    chunks = chunk_text(text, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)

    with get_db() as db:
        for idx, chunk in enumerate(chunks):
            db.execute(
                "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)",
                (
                    str(uuid4()),
                    file.filename,
                    digest,
                    idx,
                    chunk,
                    datetime.utcnow().isoformat(),
                ),
            )

    create_lessons(file.filename)

    return IngestResponse(
        filename=file.filename,
        chunks_created=len(chunks),
    )

# ======================================================
# Retrieval Endpoints
# ======================================================
@router.get("/documents/{filename}")
def get_document(filename: str):
    with get_db() as db:
        rows = db.execute(
            """
            SELECT content
            FROM chunks
            WHERE filename = ?
            ORDER BY chunk_index
            """,
            (filename,),
        ).fetchall()

    if not rows:
        raise HTTPException(404, "Document not found")

    return {
        "filename": filename,
        "chunks": [r[0] for r in rows],
    }


@router.get("/lessons/{filename}")
def get_lessons(filename: str):
    with get_db() as db:
        lessons = db.execute(
            """
            SELECT id, title, lesson_index
            FROM lessons
            WHERE filename = ?
            ORDER BY lesson_index
            """,
            (filename,),
        ).fetchall()

    if not lessons:
        raise HTTPException(404, "No lessons found")

    response = []
    for lesson_id, title, lesson_index in lessons:
        response.append(
            {
                "lesson_index": lesson_index,
                "title": title,
                "content": assemble_lesson_text(lesson_id),
            }
        )

    return {"filename": filename, "lessons": response}

# ======================================================
# Teaching Generation
# ======================================================
@router.post("/lessons/{filename}/{lesson_index}/generate")
def generate_lesson_teaching(filename: str, lesson_index: int):
    with get_db() as db:
        lesson = db.execute(
            """
            SELECT id
            FROM lessons
            WHERE filename = ? AND lesson_index = ?
            """,
            (filename, lesson_index),
        ).fetchone()

    if not lesson:
        raise HTTPException(404, "Lesson not found")

    lesson_text = assemble_lesson_text(lesson[0])
    teaching_text = generate_teaching_content(lesson_text)

    with get_db() as db:
        db.execute(
            "INSERT INTO teaching_content VALUES (?, ?, ?, ?)",
            (
                str(uuid4()),
                lesson[0],
                teaching_text,
                datetime.utcnow().isoformat(),
            ),
        )

    return {
        "lesson_index": lesson_index,
        "teaching_text": teaching_text,
    }

# ======================================================
# TTS Streaming
# ======================================================
@router.get("/lessons/{filename}/{lesson_index}/tts")
def stream_lesson_tts(filename: str, lesson_index: int):
    with get_db() as db:
        row = db.execute(
            """
            SELECT tc.teaching_text
            FROM teaching_content tc
            JOIN lessons l ON l.id = tc.lesson_id
            WHERE l.filename = ? AND l.lesson_index = ?
            ORDER BY tc.created_at DESC
            LIMIT 1
            """,
            (filename, lesson_index),
        ).fetchone()

    if not row:
        raise HTTPException(404, "Teaching content not generated")

    return StreamingResponse(
        audio_stream(row[0]),
        media_type="audio/mpeg",
    )

# ======================================================
# Health
# ======================================================
@router.get("/health")
def health():
    return {"status": "ok"}
