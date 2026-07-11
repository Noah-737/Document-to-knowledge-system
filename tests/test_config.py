from pathlib import Path

from doc2knowledge.config import Settings


def test_settings_use_expected_model_defaults(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path)

    assert settings.data_dir == tmp_path
    assert settings.embedding_model == "mixedbread-ai/mxbai-embed-large-v1"
    assert settings.embedding_dimensions == 1024
    assert settings.llm_model == "gemma-4-31b-it"
    assert settings.top_k == 6
    assert settings.gemini_api_key is None


def test_settings_accept_gemini_api_key_alias(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("GEMINI_API_KEY", "secret")

    settings = Settings()

    assert settings.gemini_api_key is not None
    assert settings.gemini_api_key.get_secret_value() == "secret"
