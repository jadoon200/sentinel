FROM python:3.12-slim

WORKDIR /app

# Install pinned dependencies first for layer caching
COPY requirements.lock ./
RUN pip install --no-cache-dir -r requirements.lock

COPY . .
RUN pip install --no-cache-dir .

CMD ["python", "-m", "sentinel.ingest.flows"]
