# =========================
# Base Image
# =========================
FROM python:3.12-slim

# =========================
# System Dependencies
# =========================
RUN apt-get update && apt-get install -y \
    ffmpeg \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# =========================
# App & Data Directories
# =========================
WORKDIR /app
RUN mkdir -p /data

# =========================
# Copy Project Files
# =========================
COPY . .

# =========================
# Python Dependencies
# =========================
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# =========================
# Environment Variables
# =========================
ENV PYTHONUNBUFFERED=1
ENV RESUMES_DB_PATH=/data/resumes.db
ENV ATTENDANCE_DB_PATH=/data/attendance.db
ENV DOCUMENTS_DB_PATH=/data/documents.db

# =========================
# Expose API Port
# =========================
EXPOSE 8000

# =========================
# SQLite Persistence Volume
# =========================
VOLUME ["/data"]

# =========================
# Run FastAPI (SQLite-safe)
# =========================
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
