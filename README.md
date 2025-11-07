# TX AI Web — Deploy on Render

## 🚀 Cách triển khai
1. Copy file `app_tx_md5.py` của bạn vào thư mục này.
2. Đưa toàn bộ folder này lên GitHub (repo mới).
3. Vào [https://render.com](https://render.com) → đăng nhập bằng GitHub.
4. Chọn **New + → Web Service**.
5. Kết nối đến repo.
6. Thiết lập:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `python app_tx_md5.py`
   - **Environment:** Python 3
7. Render sẽ tự build và tạo link web dạng:
   ```
   https://<tên-app>.onrender.com
   ```

> ⚠️ Nếu gặp lỗi “port 5000 already in use”, thêm dòng này vào cuối `app_tx_md5.py`:
> ```python
> if __name__ == "__main__":
>     app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
> ```
