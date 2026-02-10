from fastapi import APIRouter, File, UploadFile, HTTPException
from task_old import extract_fio
import shutil
import tempfile
from pathlib import Path

router = APIRouter()


@router.post("/process-document")
async def process_document(file: UploadFile = File(...)):
    try:
        suffix = Path(file.filename or "image").suffix or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_path = tmp.name

        try:
            fio = extract_fio(temp_path)
            return {"fio": fio}
        finally:
            Path(temp_path).unlink(missing_ok=True)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health():
    return {"status": "ok"}