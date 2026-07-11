from __future__ import annotations

from datetime import datetime
from typing import Annotated, cast

from fastapi import (
    APIRouter,
    File,
    HTTPException,
    Request,
    Response,
    UploadFile,
    status,
)
from pydantic import BaseModel, ConfigDict

from doc2knowledge.domain import Document, DocumentStatus
from doc2knowledge.ingestion.extractors import (
    EmptyDocumentError,
    ExtractionError,
    UnsupportedMediaTypeError,
)
from doc2knowledge.ingestion.service import IngestionService, UploadTooLargeError

router = APIRouter(prefix="/documents", tags=["documents"])


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    media_type: str
    sha256: str
    status: DocumentStatus
    created_at: datetime
    chunk_count: int
    error_message: str | None


class UploadResponse(BaseModel):
    document: DocumentResponse
    duplicate: bool


def _service(request: Request) -> IngestionService:
    return cast(IngestionService, request.app.state.ingestion_service)


def _response(document: Document) -> DocumentResponse:
    return DocumentResponse.model_validate(document)


@router.post("", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    request: Request,
    response: Response,
    file: Annotated[UploadFile, File()],
) -> UploadResponse:
    data = await file.read()
    try:
        result = _service(request).ingest(
            filename=file.filename or "document",
            media_type=file.content_type or "application/octet-stream",
            data=data,
        )
    except UploadTooLargeError as error:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={"code": "upload_too_large", "message": str(error)},
        ) from error
    except UnsupportedMediaTypeError as error:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail={"code": "unsupported_media_type", "message": str(error)},
        ) from error
    except EmptyDocumentError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "empty_document", "message": str(error)},
        ) from error
    except ExtractionError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "extraction_failed", "message": str(error)},
        ) from error

    if result.duplicate:
        response.status_code = status.HTTP_200_OK
    return UploadResponse(
        document=_response(result.document),
        duplicate=result.duplicate,
    )


@router.get("", response_model=list[DocumentResponse])
def list_documents(request: Request) -> list[DocumentResponse]:
    return [_response(document) for document in _service(request).list_documents()]


@router.get("/{document_id}", response_model=DocumentResponse)
def get_document(document_id: str, request: Request) -> DocumentResponse:
    document = _service(request).get_document(document_id)
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="document not found"
        )
    return _response(document)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(document_id: str, request: Request) -> Response:
    if not _service(request).delete_document(document_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="document not found"
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)
