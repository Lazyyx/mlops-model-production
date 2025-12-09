FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml /app/

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN uv sync --frozen

COPY . /app 

ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["/app/.venv/bin/uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]