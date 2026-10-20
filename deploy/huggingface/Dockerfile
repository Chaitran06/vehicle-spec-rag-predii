# API for the F-150 Spec Finder, built for a Hugging Face Docker Space.
#
# Why here and not Vercel/Netlify: the query has to be embedded by the same
# MiniLM model that embedded the manual, which means PyTorch at request time.
# That is far over the serverless function size limits on both. A Docker Space
# runs it on a free CPU box with the model baked into the image.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/user/.cache/huggingface \
    SPECRAG_DATA_DIR=/tmp/specrag-build \
    SPECRAG_MODE=llm

# Spaces run as uid 1000.
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# CPU-only torch first: the default wheel pulls ~2 GB of CUDA libraries a CPU
# Space can never use.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

USER user
# Bake the embedding model into the image so a cold start does not download it.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY --chown=user specrag ./specrag
COPY --chown=user api ./api
COPY --chown=user frontend ./frontend
# Prebuilt on your machine with `specrag ingest && specrag export-index`.
COPY --chown=user index ./index

EXPOSE 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
