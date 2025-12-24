from fastapi import FastAPI, File, UploadFile, HTTPException, Body, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response, JSONResponse, FileResponse
from pydantic import BaseModel
from gtts import gTTS
from openai import OpenAI
from dotenv import load_dotenv
import tempfile, subprocess, os, json
import uuid
import asyncio
import os
import logging
import traceback
from helpers.tts_helpers import generate_greeting_tts
from teaching import router as teaching_router
from Interview import router as interview_router

load_dotenv()
client = OpenAI()

app = FastAPI()

# =========================
# CORS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "https://nephele-frontend.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# SYSTEM PROMPT (Nephele)
# =========================
SYSTEM_PROMPT = """
You are Nephele, an AI cloud-based student companion robot.
You assist students with casual conversation, learning, doubt clarification, 
and teaching concepts clearly. 

Your style should be:
- Short, natural, human-like responses.
- Prefer 1–3 sentences.
- Avoid lists unless the student asks.
- Speak conversationally, as if talking to a peer.
- Never read emojis aloud.
- If you are unsure of what the student means, ask a short clarification question.
- Encourage curiosity and provide friendly guidance.

Always respond in a clear, concise, and empathetic tone.
"""


# =========================
# LOGGING AND UTILS
# =========================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

AUDIO_DIR = "audio"
os.makedirs(AUDIO_DIR, exist_ok=True)


# =========================
# CLASS FOR QR ENDPOINT
# =========================

class ScanData(BaseModel):
    name: str
    roll_no: str
    role: str

@app.post("/scan")
async def scan_endpoint(data: ScanData, background_tasks: BackgroundTasks):
    """
    QR scan endpoint that returns a greeting audio TTS.
    """
    return await generate_greeting_tts(
        name=data.name,
        role=data.role,
        background_tasks=background_tasks
    )

# =========================
# AGENT PRIMITIVES
# =========================
async def transcribe_audio(file: UploadFile) -> str:
    """STT: Convert audio file to text using OpenAI."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        tmp.write(await file.read())
        webm_path = tmp.name

    wav_path = webm_path.replace(".webm", ".wav")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", webm_path, wav_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
        with open(wav_path, "rb") as f:
            transcription = client.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=f
            )
        return transcription.text
    finally:
        for path in (webm_path, wav_path):
            if os.path.exists(path):
                os.remove(path)


def think_stream(user_text: str, history=None):
    """Stream GPT response token by token. Optionally include conversation history."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages += history
    messages.append({"role": "user", "content": user_text})

    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        stream=True,
        messages=messages,
    )
    for chunk in stream:
        content = getattr(chunk.choices[0].delta, "content", None)
        if content:
            yield content


def token_buffer(tokens, size=40):
    """Buffer small GPT tokens into larger chunks for TTS efficiency."""
    buf = ""
    for t in tokens:
        buf += t
        if len(buf) >= size:
            yield buf
            buf = ""
    if buf:
        yield buf


def speak_stream(text_chunks):
    """Convert text chunks into streamed audio, stripping WAV headers after the first chunk."""
    first = True
    for text in text_chunks:
        audio_bytes = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=text
        ).read()
        if first:
            first = False
            yield audio_bytes  # full WAV for first chunk
        else:
            yield audio_bytes[44:]  # strip header for subsequent chunks


def speak_text(text: str) -> bytes:
    """TTS for single text input."""
    return client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=text
    ).read()

# =========================
# ENDPOINTS
# =========================
@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Backend is running 🚀"
    }

@app.post("/voice")
async def voice(file: UploadFile = File(...)):
    """STT endpoint: audio → text"""
    if not file:
        raise HTTPException(status_code=400, detail="Audio file required")
    text = await transcribe_audio(file)
    return JSONResponse({"text": text})


@app.post("/tts")
async def tts(payload: dict = Body(...)):
    """TTS endpoint: text → audio"""
    text = payload.get("text")
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")
    audio_bytes = speak_text(text)
    return Response(content=audio_bytes, media_type="audio/wav")


@app.post("/agent")
async def agent(file: UploadFile = File(...)):
    """Full voice agent: audio → GPT → streamed audio"""
    if not file:
        raise HTTPException(status_code=400, detail="Audio required")

    user_text = await transcribe_audio(file)
    if not user_text.strip():
        audio_bytes = speak_text("I didn’t catch that. Could you repeat?")
        return Response(content=audio_bytes, media_type="audio/wav")

    tokens = think_stream(user_text)
    buffered_tokens = token_buffer(tokens)
    audio_stream = speak_stream(buffered_tokens)

    return StreamingResponse(audio_stream, media_type="audio/wav")


@app.post("/agent-memory")
async def agent_memory(file: UploadFile = File(...), history: str = Form(...)):
    """Full voice agent with conversation memory (stateless server)"""
    if not file:
        raise HTTPException(status_code=400, detail="Audio file required")

    # Save audio temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        tmp.write(await file.read())
        webm_path = tmp.name
    wav_path = webm_path.replace(".webm", ".wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", webm_path, wav_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True
    )

    # STT
    user_text = await transcribe_audio(UploadFile(filename=wav_path, file=open(wav_path, "rb")))
    os.remove(webm_path)
    os.remove(wav_path)

    if not user_text.strip():
        audio_bytes = speak_text("I didn’t catch that. Could you repeat?")
        return Response(content=audio_bytes, media_type="audio/wav")

    # Conversation memory from client
    conversation = json.loads(history)
    conversation.append({"role": "user", "content": user_text})

    # GPT response
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    assistant_text = completion.choices[0].message.content
    conversation.append({"role": "assistant", "content": assistant_text})

    # TTS
    audio_bytes = speak_text(assistant_text)
    return Response(content=audio_bytes, media_type="audio/wav")



# =========================
# COMPERE ENDPOINT
# =========================


@app.post("/compere")
async def text_to_speech(file: UploadFile = File(...)):
    logger.info("➡️ /tts request received")

    try:
        logger.info("📄 Reading uploaded file")
        content = await file.read()
        text = content.decode("utf-8").strip()

        if not text:
            raise HTTPException(status_code=400, detail="Text file is empty")

        logger.info(f"📝 Text length: {len(text)} characters")

        audio_id = str(uuid.uuid4())
        audio_path = f"{AUDIO_DIR}/{audio_id}.wav"

        logger.info("🔊 Calling OpenAI TTS API")

        with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=text,
            timeout=30,  # IMPORTANT: prevent hanging forever
        ) as response:
            logger.info("⬇️ Streaming audio to file")
            response.stream_to_file(audio_path)

        logger.info(f"✅ Audio saved: {audio_path}")

        return FileResponse(
            audio_path,
            media_type="audio/wav",
            filename="speech.wav",
        )

    except Exception as e:
        logger.error("🔥 ERROR during /tts processing")
        logger.error(str(e))
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="TTS generation failed")



app.include_router(
    teaching_router,
    prefix="/teaching",
    tags=["Teaching"]
)

app.include_router(
    interview_router,
    prefix="/interview",
    tags=["Interview"]
)
