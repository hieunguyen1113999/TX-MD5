# --- Base image ---
FROM python:3.10-slim

# --- Set working directory ---
WORKDIR /app

# --- Copy requirements (nếu có) ---
COPY requirements.txt .

# --- Install dependencies ---
RUN pip install --no-cache-dir -r requirements.txt

# --- Copy toàn bộ project ---
COPY . .

# --- Expose port cho Render ---
EXPOSE 10000

# --- Lệnh khởi chạy chính ---
CMD ["python", "app_tx_md5.py"]
