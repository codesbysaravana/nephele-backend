import uuid
import asyncio
import os
from gtts import gTTS
from fastapi import BackgroundTasks
from fastapi.responses import FileResponse

def remove_file(file_path: str):
    """Delete temporary audio file."""
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            print(f"Deleted temporary file: {file_path}")
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")


async def generate_greeting_tts(
    name: str,
    role: str,
    background_tasks: BackgroundTasks,
    greeting_template: str = None
) -> FileResponse:
    """
    Generate a TTS greeting as an audio file.
    
    Args:
        name: Person's name
        role: Role of person
        background_tasks: FastAPI BackgroundTasks for cleanup
        greeting_template: Optional custom greeting with {name} and {role}
        
    Returns:
        FileResponse containing the generated audio.
    """
    # Default greeting if none provided
    if not greeting_template:
        greeting_template = "Hello {name}! Good morning. You are registered as a {role}. Have a great day ahead!"

    # Fill template
    greeting_text = greeting_template.format(name=name, role=role)

    # Generate TTS
    audio_file = f"audio_{uuid.uuid4().hex}.mp3"
    tts = gTTS(text=greeting_text, lang="en", tld="com")  # female voice
    await asyncio.to_thread(tts.save, audio_file)

    # Schedule deletion
    background_tasks.add_task(remove_file, audio_file)

    # Return as FileResponse
    return FileResponse(audio_file, media_type="audio/mpeg", filename="greeting.mp3")




