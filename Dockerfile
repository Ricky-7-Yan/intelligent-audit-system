FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    WEB_HOST=0.0.0.0 \
    SECURITY_MODE=enforced \
    WEB_PORT=8000 \
    PORT=8000

WORKDIR /app

RUN groupadd --system auditpilot \
    && useradd --system --gid auditpilot --home-dir /app --shell /usr/sbin/nologin auditpilot

COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels

COPY --chown=auditpilot:auditpilot . .
RUN mkdir -p /app/data /app/logs \
    && chown -R auditpilot:auditpilot /app/data /app/logs

USER auditpilot

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.getenv(\"PORT\", os.getenv(\"WEB_PORT\", \"8000\"))}/api/health/ready', timeout=5).read()" || exit 1

CMD ["python", "start.py"]
