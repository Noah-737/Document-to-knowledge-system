FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN addgroup --system app && adduser --system --ingroup app app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m pip install --upgrade pip && python -m pip install .

RUN mkdir -p /app/data && chown -R app:app /app
USER app

EXPOSE 8000

CMD ["uvicorn", "doc2knowledge.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
