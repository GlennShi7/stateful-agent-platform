# Build context is the repository root (see docker-compose.yml), so db/ and bench/ can be included.
FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
COPY backend/requirements.txt backend/requirements-dev.txt backend/requirements.lock ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt -c requirements.lock
COPY db ./db
COPY bench ./bench
COPY backend/pytest.ini ./
COPY backend/agentplat ./agentplat
COPY backend/evals ./evals
COPY backend/tests ./tests
EXPOSE 8000 8001
CMD ["uvicorn", "agentplat.gateway:app", "--host", "0.0.0.0", "--port", "8000"]
