# One image, many roles: the command decides whether it is the API, monitor, simulator or dashboard.
FROM python:3.12-slim AS builder
WORKDIR /w
COPY requirements.txt .
RUN pip wheel --no-cache-dir -r requirements.txt -w /wheels

FROM python:3.12-slim AS base
# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 git && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /wheels /wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt && rm -rf /wheels
COPY selfheal ./selfheal
COPY prompts ./prompts
COPY params.yaml dvc.yaml pyproject.toml ./
ENV PYTHONPATH=/app SELFHEAL_DATA_DIR=/data PYTHONUNBUFFERED=1

FROM base AS runtime
ARG GIT_SHA=unknown
ENV GIT_SHA=$GIT_SHA
VOLUME /data
EXPOSE 8000 8501
CMD ["uvicorn", "selfheal.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
