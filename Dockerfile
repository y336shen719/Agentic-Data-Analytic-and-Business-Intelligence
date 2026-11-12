# syntax=docker/dockerfile:1
# Multi-platform index verified with imagetools inspect; select AMD64 at build time.
FROM python:3.11.18-slim-bookworm@sha256:a83ba18aef0f865b1a4a611c2cf460fe446f2795673bbb9fe43be570439e3b39

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/home/appuser \
    MPLCONFIGDIR=/home/appuser/.cache/matplotlib

WORKDIR /app

# Install dependencies before copying code to reuse this layer on code changes.
COPY requirements-docker.lock ./requirements-docker.lock
# Do not silently resolve unpinned dependencies or invoke source build toolchains.
RUN python -m pip install --no-cache-dir --only-binary=:all: --no-deps -r requirements-docker.lock \
    && python -m pip check

# Only the user home is writable by the application; code stays root-owned.
RUN groupadd --gid 10001 appuser \
    && useradd --uid 10001 --gid 10001 --create-home appuser

COPY app.py ./app.py
COPY agent/ ./agent/
COPY tools/ ./tools/
COPY views/ ./views/
COPY assets/ ./assets/
COPY data/__init__.py data/prepare_data.py data/schema.md ./data/
COPY .streamlit/config.toml ./.streamlit/config.toml

# Runtime mounts provide /app/data/ecommerce.duckdb and /run/secrets/*.
USER appuser

# Documents the container port; publishing it is a separate runtime decision.
EXPOSE 8501

# Checks the web server, not OpenAI credentials, tracing, or database queries.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; response = urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3); response.close()"]

CMD ["python", "-m", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--browser.gatherUsageStats=false"]
