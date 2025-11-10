# gunicorn_conf.py
# Post-fork hook: gọi start_background_threads() bên trong mỗi worker
def post_fork(server, worker):
    try:
        # import hàm khởi background threads đã được bạn tách ra trong app_tx_md5.py
        from app_tx_md5 import start_background_threads
        start_background_threads()
        server.log.info("Started background threads in worker")
    except Exception as e:
        server.log.error("Failed to start background threads in worker: %s", e)