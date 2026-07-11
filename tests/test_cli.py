from typing import Any

from doc2knowledge import __main__


def test_main_starts_uvicorn_factory(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    call: dict[str, Any] = {}

    def fake_run(app: str, **kwargs: Any) -> None:
        call["app"] = app
        call.update(kwargs)

    monkeypatch.setattr(__main__.uvicorn, "run", fake_run)

    __main__.main()

    assert call == {
        "app": "doc2knowledge.api:create_app",
        "factory": True,
        "host": "0.0.0.0",
        "port": 8000,
    }
