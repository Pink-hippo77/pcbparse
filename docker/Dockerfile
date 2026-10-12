FROM edarace-agent-base:ubuntu24

ARG PCBPARSE_PIP_INDEX_URL=https://pypi.org/simple

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=180 \
    PIP_RETRIES=8 \
    PYTHONHASHSEED=0 \
    PCBPARSE_APP_ROOT=/submission/app \
    LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cuda_nvrtc/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cufft/lib:/usr/local/lib/python3.12/dist-packages/nvidia/curand/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cusolver/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cusparse/lib:/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib:/usr/local/lib/python3.12/dist-packages/nvidia/nvjitlink/lib:/usr/local/lib/python3.12/dist-packages/nvidia/nvtx/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cuda_cupti/lib

WORKDIR /submission/app

COPY pyproject.toml requirements.edarace.txt /submission/app/
COPY src /submission/app/src
COPY configs /submission/app/configs
COPY models/models.lock.json /submission/app/models/models.lock.json
COPY models/releases/v8-portable /submission/app/models/releases/v8-portable
COPY docker/agent_system /submission/agent_system

RUN python3 -m pip install --break-system-packages --no-cache-dir \
        --index-url "${PCBPARSE_PIP_INDEX_URL}" \
        -r /submission/app/requirements.edarace.txt \
    && python3 -m pip install --break-system-packages --no-cache-dir --no-deps \
        /submission/app \
    && chmod 0555 /submission/agent_system

# BuildKit has no GPU device. Verify imports/serialization plus every model and
# lineage byte here; execute the four runtime adapters after image creation.
RUN python3 -m pcbparse.cli healthcheck \
        --config /submission/app/configs/pipeline.edarace.yaml \
        --models-lock /submission/app/models/models.lock.json \
    && python3 -c "from pcbparse.perception.production_provider import composite_provider_config_from_pipeline, required_model_tasks_for_config; from pcbparse.config import load_config; from pcbparse.service.readiness import readiness_report; config=load_config('/submission/app/configs/pipeline.edarace.yaml'); tasks=required_model_tasks_for_config(composite_provider_config_from_pipeline(config)); report=readiness_report(provider_name='production', models_lock='/submission/app/models/models.lock.json', required_tasks=tasks, require_lineage=True); assert report['ready'], report"

WORKDIR /submission
ENTRYPOINT ["/submission/agent_system"]
