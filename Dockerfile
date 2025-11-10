FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default for local testing; Render will override PORT env at runtime
ENV PORT=5000
EXPOSE 5000

# Use shell form so $PORT is expanded at runtime.
# Use 1 worker to avoid duplicate background threads / SQLite concurrency issues.
CMD gunicorn app_tx_md5:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120
