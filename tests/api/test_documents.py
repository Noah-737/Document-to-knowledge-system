from pathlib import Path

from fastapi.testclient import TestClient

from doc2knowledge.api import create_app
from doc2knowledge.config import Settings


def make_client(tmp_path: Path, max_upload_bytes: int = 1024) -> TestClient:
    return TestClient(
        create_app(
            Settings(
                data_dir=tmp_path,
                max_upload_bytes=max_upload_bytes,
            )
        )
    )


def test_upload_list_get_duplicate_and_delete_document(tmp_path: Path) -> None:
    client = make_client(tmp_path)

    uploaded = client.post(
        "/documents",
        files={"file": ("notes.txt", b"alpha beta gamma", "text/plain")},
    )

    assert uploaded.status_code == 201
    body = uploaded.json()
    document_id = body["document"]["id"]
    assert body["duplicate"] is False
    assert body["document"]["status"] == "ready"
    assert body["document"]["chunk_count"] == 1

    duplicate = client.post(
        "/documents",
        files={"file": ("renamed.txt", b"alpha beta gamma", "text/plain")},
    )
    assert duplicate.status_code == 200
    assert duplicate.json()["duplicate"] is True
    assert duplicate.json()["document"]["id"] == document_id

    listed = client.get("/documents")
    assert listed.status_code == 200
    assert [item["id"] for item in listed.json()] == [document_id]

    fetched = client.get(f"/documents/{document_id}")
    assert fetched.status_code == 200
    assert fetched.json()["filename"] == "notes.txt"

    deleted = client.delete(f"/documents/{document_id}")
    assert deleted.status_code == 204
    assert client.get(f"/documents/{document_id}").status_code == 404


def test_upload_validation_errors_are_explicit(tmp_path: Path) -> None:
    client = make_client(tmp_path, max_upload_bytes=4)

    too_large = client.post(
        "/documents",
        files={"file": ("large.txt", b"12345", "text/plain")},
    )
    unsupported = client.post(
        "/documents",
        files={"file": ("blob.bin", b"123", "application/octet-stream")},
    )
    empty = client.post(
        "/documents",
        files={"file": ("empty.txt", b"   ", "text/plain")},
    )

    assert too_large.status_code == 413
    assert too_large.json()["detail"]["code"] == "upload_too_large"
    assert unsupported.status_code == 415
    assert unsupported.json()["detail"]["code"] == "unsupported_media_type"
    assert empty.status_code == 422
    assert empty.json()["detail"]["code"] == "empty_document"
