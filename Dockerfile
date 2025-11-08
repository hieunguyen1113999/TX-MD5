# --- Base image ---
FROM python:3.10-slim

# --- Set working directory ---
WORKDIR /app

# --- Copy requirements ---
COPY requirements.txt .

# --- Install dependencies ---
RUN pip install --no-cache-dir -r requirements.txt

# --- Copy toàn bộ project ---
COPY . .

# --- Render yêu cầu expose đúng port ---
EXPOSE 10000

# --- Chạy app, đảm bảo nhận biến PORT từ Render ---
CMD ["bash", "-c", "python app_tx_md5.py"]
