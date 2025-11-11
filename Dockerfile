FROM python:3.10-slim-bookworm

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV PATH="/app/.venv/bin/:$PATH"

ADD pyproject.toml uv.lock ./

RUN uv sync --locked

ADD model_v20251111.bin serve.py ./

EXPOSE 8000

ENTRYPOINT ["fastapi", "run", "serve.py"]
