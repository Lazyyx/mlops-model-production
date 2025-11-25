FROM python:3.12-slim

WORKDIR /app

# Install uv.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY . /app
COPY src ./src

RUN uv sync --frozen --no-cache

ENV PYTHONPATH=/app/src
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

EXPOSE 8501

CMD ["/app/.venv/bin/streamlit", "run", "src/ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]