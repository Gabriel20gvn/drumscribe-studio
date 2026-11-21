import os
import time
import uuid
import shutil
from typing import Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from processor import process_full_pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JOBS_DIR = os.path.join(BASE_DIR, "jobs")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(JOBS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

app = FastAPI(title="DrumScribe Studio API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

JOBS: Dict[str, Dict[str, Any]] = {}


def cleanup_old_jobs(max_age_seconds: int = 7200):
    now = time.time()
    expired = [jid for jid, info in JOBS.items() if now - info.get("created_at", now) > max_age_seconds]
    for jid in expired:
        JOBS.pop(jid, None)
        jdir = os.path.join(JOBS_DIR, jid)
        if os.path.exists(jdir):
            shutil.rmtree(jdir, ignore_errors=True)


def run_job_worker(
    job_id: str,
    url: str,
    demucs_model: str,
    whisper_model: str,
    language: str,
    cookies_path: Optional[str]
):
    job_dir = os.path.join(JOBS_DIR, job_id)

    def update_progress(pct: int, msg: str):
        if job_id in JOBS:
            JOBS[job_id]["progress"] = pct
            JOBS[job_id]["step"] = msg

    try:
        JOBS[job_id]["status"] = "running"
        result = process_full_pipeline(
            url=url,
            job_dir=job_dir,
            demucs_model=demucs_model,
            whisper_model=whisper_model,
            language=language,
            cookies_path=cookies_path,
            progress_cb=update_progress
        )
        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["progress"] = 100
        JOBS[job_id]["step"] = "Concluído!"
        JOBS[job_id]["result"] = result
    except Exception as exc:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = str(exc)
        JOBS[job_id]["step"] = "Erro durante o processamento."


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    index_path = os.path.join(STATIC_DIR, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/api/process")
async def start_processing(
    background_tasks: BackgroundTasks,
    url: str = Form(...),
    demucs_model: str = Form("htdemucs"),
    whisper_model: str = Form("small"),
    language: str = Form("pt"),
    cookies_file: Optional[UploadFile] = File(None)
):
    cleanup_old_jobs()

    if not url.strip():
        raise HTTPException(status_code=400, detail="Informe uma URL válida do YouTube.")

    job_id = uuid.uuid4().hex[:12]
    job_dir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    cookies_path = None
    if cookies_file and cookies_file.filename:
        cookies_path = os.path.join(job_dir, "cookies.txt")
        with open(cookies_path, "wb") as f:
            f.write(await cookies_file.read())

    JOBS[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 5,
        "step": "Na fila para iniciar...",
        "created_at": time.time(),
        "result": None,
        "error": None
    }

    background_tasks.add_task(
        run_job_worker,
        job_id,
        url.strip(),
        demucs_model,
        whisper_model,
        language,
        cookies_path
    )

    return {"job_id": job_id}


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Tarefa não encontrada.")
    return JOBS[job_id]


@app.get("/api/files/{job_id}/{filename}")
async def get_job_file(job_id: str, filename: str):
    safe_name = os.path.basename(filename)
    file_path = os.path.join(JOBS_DIR, job_id, safe_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Arquivo não encontrado.")

    media_type = None
    if safe_name.endswith(".musicxml") or safe_name.endswith(".xml"):
        media_type = "application/vnd.recordare.musicxml+xml"
    elif safe_name.endswith(".mp3"):
        media_type = "audio/mpeg"
    elif safe_name.endswith(".wav"):
        media_type = "audio/wav"
    elif safe_name.endswith(".zip"):
        media_type = "application/zip"

    return FileResponse(path=file_path, filename=safe_name, media_type=media_type)
