FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
RUN useradd -m app && chown -R app /app
USER app
EXPOSE 8000 8501 8765
# default: API. docker-compose overrides the command for ui / mcp / ingest.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
