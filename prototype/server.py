from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
import os
import uuid
import glob
import asyncio
import logging
from typing import Optional

# local import of the refactored pipeline
from prototype.main import generate_plan_from_files

BASE_DIR = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(title="NetSuite Plan Prototype API")
logger = logging.getLogger("prototype.server")


def _save_upload_bytes(content: bytes, filename: str) -> str:
    file_id = uuid.uuid4().hex
    suffix = os.path.splitext(filename)[1] or ""
    dest = os.path.join(UPLOAD_DIR, f"{file_id}{suffix}")
    with open(dest, "wb") as f:
        f.write(content)
    return file_id


def _path_for_file_id(file_id: str) -> Optional[str]:
    matches = glob.glob(os.path.join(UPLOAD_DIR, f"{file_id}*"))
    return matches[0] if matches else None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload/transcript")
async def upload_transcript(file: UploadFile = File(...)):
    try:
        data = await file.read()
        file_id = _save_upload_bytes(data, file.filename)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"file_id": file_id}


@app.post("/upload/pm_export")
async def upload_pm_export(file: UploadFile = File(...)):
    try:
        data = await file.read()
        file_id = _save_upload_bytes(data, file.filename)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"file_id": file_id}


@app.post("/upload/template")
async def upload_template(file: UploadFile = File(...)):
    try:
        data = await file.read()
        file_id = _save_upload_bytes(data, file.filename)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"file_id": file_id}


@app.post("/upload/summary")
async def upload_summary(file: UploadFile = File(...)):
    try:
        data = await file.read()
        file_id = _save_upload_bytes(data, file.filename)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"file_id": file_id}


@app.post("/generate")
async def generate(
    tenant_id: str = Form(...),
    engagement_id: str = Form(...),
    transcript_file_id: str = Form(...),
    pm_export_file_id: str = Form(...),
    template_file_id: str = Form(...),
    summary_file_id: Optional[str] = Form(None),
):
    # resolve file ids to paths
    transcript_path = _path_for_file_id(transcript_file_id)
    pm_path = _path_for_file_id(pm_export_file_id)
    template_path = _path_for_file_id(template_file_id)
    summary_path = _path_for_file_id(summary_file_id) if summary_file_id else None

    missing = []
    if not transcript_path:
        missing.append("transcript_file_id")
    if not pm_path:
        missing.append("pm_export_file_id")
    if not template_path:
        missing.append("template_file_id")
    if missing:
        raise HTTPException(status_code=400, detail={"error": "file_id(s) not found", "missing": missing})

    # If summary not provided, fallback to an empty txt file
    if not summary_path:
        summary_path = os.path.join(UPLOAD_DIR, "__empty_summary.txt")
        if not os.path.exists(summary_path):
            with open(summary_path, "w") as f:
                f.write("")

    try:
        # run the blocking generation in a thread to avoid blocking the event loop
        plan = await asyncio.to_thread(
            generate_plan_from_files,
            transcript_path=transcript_path,
            retro_path=summary_path,
            template_path=template_path,
            pm_export_path=pm_path,
            tenant_id=tenant_id,
            engagement_id=engagement_id,
            output_dir=OUTPUT_DIR,
        )
    except ConnectionResetError as cre:
        logger.warning("Client connection lost during generation: %s", cre)
        # client disconnected; return an error indicating cancellation
        raise HTTPException(status_code=499, detail="client disconnected during processing")
    except asyncio.CancelledError:
        logger.warning("Request cancelled by client")
        raise HTTPException(status_code=499, detail="request cancelled")
    except Exception as exc:
        logger.exception("Error during plan generation: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    return JSONResponse(content={"plan": plan})
