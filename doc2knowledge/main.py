import os
import shutil
from fastapi import FastAPI, File, UploadFile
from typing import List

app = FastAPI()

UPLOAD_DIR = "uploads"

@app.on_event("startup")
async def startup_event():
    """
    Create the upload directory if it doesn't exist.
    """
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Uploads multiple files and saves them to the uploads directory.
    """
    filenames = []
    for file in files:
        if file.filename:
            # Basic sanitization
            filename = os.path.basename(file.filename)
            file_path = os.path.join(UPLOAD_DIR, filename)
            filenames.append(filename)

            try:
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
            finally:
                file.file.close()


    return {"status": "success", "message": "Files uploaded successfully.", "filenames": filenames}
