import os
import shutil
from fastapi import FastAPI, File, UploadFile
from typing import List
from .utils import extract_text_from_file, chunk_text
from .vector_store import create_and_store_embeddings

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
    Uploads multiple files, processes them, and stores their embeddings.
    """
    for file in files:
        if file.filename:
            # Basic sanitization
            filename = os.path.basename(file.filename)
            file_path = os.path.join(UPLOAD_DIR, filename)

            try:
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
            finally:
                file.file.close()

            # Process the file
            text = extract_text_from_file(file_path)
            chunks = chunk_text(text)
            create_and_store_embeddings(chunks, filename)


    return {"status": "success", "message": "Files successfully processed and indexed."}
