# API image: FastAPI + CPU-only torch. Models are not baked in; the HF cache is mounted
# (see docker-compose.yml) so the image stays smaller and nothing is downloaded at runtime.
FROM python:3.12-slim
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 HF_HOME=/hf-cache
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml ./
COPY rag ./rag
COPY migrations ./migrations
COPY corpus ./corpus
COPY scripts ./scripts
RUN pip install .
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=60s --retries=5 \
  CMD python -c "import urllib.request,sys; sys.exit(urllib.request.urlopen('http://localhost:8000/health').status != 200)"
CMD ["uvicorn", "rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
