import os
import json
import time
import uuid
import shutil
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, UploadFile, File, Form, Header, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from processor import (
    process_full_pipeline,
    get_library_root,
    delete_item_from_github_library,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JOBS_DIR = os.path.join(BASE_DIR, "jobs")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(JOBS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

app = FastAPI(title="DrumScribe Studio API", version="1.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/vendor", StaticFiles(directory=os.path.join(BASE_DIR, "vendor")), name="vendor")
app.mount("/icons", StaticFiles(directory=os.path.join(BASE_DIR, "icons")), name="icons")

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
    cookies_path: Optional[str],
    uploaded_media_path: Optional[str],
    uploaded_media_name: Optional[str],
    github_token: Optional[str],
    github_repo: str
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
            uploaded_media_path=uploaded_media_path,
            uploaded_media_name=uploaded_media_name,
            base_dir=BASE_DIR,
            github_token=github_token,
            github_repo=github_repo,
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
@app.get("/index.html", response_class=HTMLResponse)
async def serve_frontend():
    return FileResponse(os.path.join(BASE_DIR, "index.html"), media_type="text/html")


@app.get("/studio.css")
async def serve_studio_styles():
    return FileResponse(os.path.join(BASE_DIR, "studio.css"), media_type="text/css")


@app.get("/score-format.js")
async def serve_score_format():
    return FileResponse(os.path.join(BASE_DIR, "score-format.js"), media_type="application/javascript")


@app.get("/drum-grid.js")
async def serve_drum_grid():
    return FileResponse(os.path.join(BASE_DIR, "drum-grid.js"), media_type="application/javascript")


@app.get("/sw.js")
async def serve_service_worker():
    return FileResponse(os.path.join(BASE_DIR, "sw.js"), media_type="application/javascript")


@app.get("/manifest.webmanifest")
async def serve_manifest():
    return FileResponse(os.path.join(BASE_DIR, "manifest.webmanifest"), media_type="application/manifest+json")


@app.post("/api/process")
async def start_processing(
    background_tasks: BackgroundTasks,
    url: str = Form(""),
    demucs_model: str = Form("htdemucs"),
    whisper_model: str = Form("small"),
    language: str = Form("pt"),
    github_token: str = Form(""),
    github_repo: str = Form("Gabriel20gvn/drumscribe-studio"),
    cookies_file: Optional[UploadFile] = File(None),
    media_file: Optional[UploadFile] = File(None)
):
    cleanup_old_jobs()

    has_media = media_file is not None and bool(media_file.filename)
    if not url.strip() and not has_media:
        raise HTTPException(status_code=400, detail="Informe um link ou envie um arquivo de áudio/vídeo.")

    job_id = uuid.uuid4().hex[:12]
    job_dir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    cookies_path = None
    if cookies_file and cookies_file.filename:
        cookies_path = os.path.join(job_dir, "cookies.txt")
        with open(cookies_path, "wb") as f:
            f.write(await cookies_file.read())

    uploaded_media_path = None
    uploaded_media_name = None
    if has_media:
        uploaded_media_name = media_file.filename
        ext = os.path.splitext(uploaded_media_name)[1] or ".mp3"
        uploaded_media_path = os.path.join(job_dir, f"uploaded_input{ext}")
        with open(uploaded_media_path, "wb") as f:
            f.write(await media_file.read())

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
        cookies_path,
        uploaded_media_path,
        uploaded_media_name,
        github_token.strip() or None,
        github_repo.strip() or "Gabriel20gvn/drumscribe-studio"
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
    return _serve_media_file(file_path, safe_name)


@app.get("/api/library")
async def list_library():
    lib_root = get_library_root(BASE_DIR)
    items: List[Dict[str, Any]] = []
    if os.path.isdir(lib_root):
        for entry in os.listdir(lib_root):
            meta_path = os.path.join(lib_root, entry, "meta.json")
            if os.path.isfile(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        items.append(json.load(f))
                except Exception:
                    continue
    items.sort(key=lambda x: x.get("created_at", 0), reverse=True)
    return {
        "storage": "google_drive" if lib_root.startswith("/content/drive") else "local",
        "items": items
    }


@app.get("/api/library/{item_id}/{filename}")
async def get_library_file(item_id: str, filename: str):
    safe_id = os.path.basename(item_id)
    safe_name = os.path.basename(filename)
    file_path = os.path.join(get_library_root(BASE_DIR), safe_id, safe_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Arquivo da biblioteca não encontrado.")
    return _serve_media_file(file_path, safe_name)


@app.delete("/api/library/{item_id}")
async def delete_library_item(
    item_id: str,
    x_github_token: Optional[str] = Header(None),
    x_github_repo: Optional[str] = Header("Gabriel20gvn/drumscribe-studio")
):
    safe_id = os.path.basename(item_id)
    item_dir = os.path.join(get_library_root(BASE_DIR), safe_id)
    if os.path.isdir(item_dir):
        shutil.rmtree(item_dir, ignore_errors=True)
    token = (x_github_token or os.environ.get("GITHUB_TOKEN") or "").strip()
    if token:
        try:
            delete_item_from_github_library(safe_id, token, x_github_repo or "Gabriel20gvn/drumscribe-studio")
        except Exception:
            pass
    return {"deleted": True, "id": safe_id}


def _serve_media_file(file_path: str, safe_name: str):
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
