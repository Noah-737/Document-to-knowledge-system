from io import BytesIO
from pathlib import Path

import pytest
from anyio import run
from fastapi import UploadFile

from doc2knowledge.api import create_app
from doc2knowledge.config import Settings
from doc2knowledge.ingestion.service import UploadTooLargeError
from doc2knowledge.routes.documents import _read_upload_limited
from tests.fakes import FakeEmbeddingService


def test_upload_reader_stops_at_limit_plus_one_byte() -> None:
    upload = UploadFile(file=BytesIO(b"abcdef"), filename="large.txt")

    with pytest.raises(UploadTooLargeError, match="limit is 4"):
        run(_read_upload_limited, upload, 4)

    assert upload.file.tell() == 5


def test_upload_reader_accepts_content_at_exact_limit() -> None:
    upload = UploadFile(file=BytesIO(b"abcd"), filename="exact.txt")

    assert run(_read_upload_limited, upload, 4) == b"abcd"
    assert upload.file.tell() == 4


def test_app_creates_one_shared_processing_limiter(tmp_path: Path) -> None:
    app = create_app(
        Settings(
            data_dir=tmp_path,
            embedding_model="test-model",
            embedding_dimensions=3,
            processing_workers=3,
        ),
        embedding_service=FakeEmbeddingService(),
    )

    limiter = app.state.processing_limiter
    assert limiter.total_tokens == 3
    assert app.state.processing_limiter is limiter
