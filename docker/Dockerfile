FROM pytorch/pytorch:2.12.1-cuda12.6-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONHASHSEED=0

RUN apt-get update \
    && apt-get install -y --no-install-recommends tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md requirements.lock requirements.runtime.lock /app/
COPY src /app/src
COPY data_tools /app/data_tools
COPY configs /app/configs
COPY models /app/models
COPY docker/entrypoint.sh /app/entrypoint.sh

RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.runtime.lock \
    && python -m pip install --no-deps . \
    && chmod 0555 /app/entrypoint.sh

# Fail the image build if the copied lock cannot resolve its model bundle.
# This catches worktree-relative paths and host-specific absolute paths before
# the image is ever deployed.
RUN python -m pcbparse.cli healthcheck --require-ready --config /app/configs/pipeline.yaml --models-lock /app/models/models.lock.json

RUN mkdir -p /data/input /data/output

EXPOSE 8080

ENTRYPOINT ["/usr/bin/tini", "--", "/app/entrypoint.sh"]
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/ready', timeout=3).read()"]
CMD ["python", "-m", "pcbparse.cli", "serve", "--host", "0.0.0.0", "--port", "8080", "--input-root", "/data/input", "--output-root", "/data/output", "--config", "/app/configs/pipeline.yaml", "--profile", "/app/configs/profiles/competition.yaml", "--provider", "production", "--models-lock", "/app/models/models.lock.json"]
