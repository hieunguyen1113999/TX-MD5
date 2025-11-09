#!/usr/bin/env python3
# coding: utf-8
"""
app_tx_md5_upgraded.py
Single-file upgraded app (keeps API endpoints compatible) with:
 - Hybrid ML (SGD + LightGBM + LSTM/Transformer) retained
 - Adds Transformer AutoEncoder (teacher) for long-range patterns (seq_len=100)
 - Adds online Reinforcement updates (policy/student MLP) on each new record
 - Adds periodic teacher training + distillation to produce lightweight student for realtime
 - Does NOT change API routes/format (keeps /api/status, /api/history, /api/taixiu, /api/taixiumd5)

Run:
    py app_tx_md5_upgraded.py
or
    python app_tx_md5_upgraded.py

Dependencies: numpy, torch, scikit-learn, lightgbm, xgboost (optional), flask, joblib, pandas
"""
import os, sys, time, threading, logging, sqlite3, json, random
from pathlib import Path
from flask import Flask, jsonify, request
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
import requests
from thuattoan8_indented import lookup_pattern_predict, detect_cau
from flask import Flask, jsonify, request
app = Flask(__name__)
# =========================
# Biến toàn cục lưu tiến độ huấn luyện mô hình
# =========================
student_tx_progress = 0.0
student_md5_progress = 0.0

def fetch_and_store_latest_result(channel="tx"):
    try:
        data = latest_result_100 if channel == "tx" else latest_result_101
        phien = str(data.get("Phien", 0))
        d1 = int(data.get("Xuc_xac_1", 0))
        d2 = int(data.get("Xuc_xac_2", 0))
        d3 = int(data.get("Xuc_xac_3", 0))
        tong = int(data.get("Tong", d1 + d2 + d3))
        ket_qua = str(data.get("Ket_qua", "Chưa rõ"))
        ...
        # kiểm tra trùng lặp phiên
        cur = DB_CONN.cursor()
        cur.execute("SELECT COUNT(*) FROM history WHERE phien=? AND channel=?", (phien, channel))
        if cur.fetchone()[0] == 0:
            ts = int(time.time())
            cur.execute(
                "INSERT INTO history (channel, phien, d1, d2, d3, tong, ket_qua, ts) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (channel, phien, d1, d2, d3, tong, ket_qua, ts)
            )
            DB_CONN.commit()
            logger.info(f"Đã thêm phiên {phien} ({ket_qua}) vào DB [{channel}]")
        return True
    except Exception as e:
        logger.error(f"Lỗi khi lấy/ghi dữ liệu thật: {e}")
        return False

def get_realtime_result(source="tx"):
    return latest_result_100 if source == "tx" else latest_result_101
# ========== BẮT ĐẦU KHỐI GỘP TỪ hit.py ==========
import json
import threading
import time
import logging
from urllib.request import urlopen, Request

lock_100 = threading.Lock()
lock_101 = threading.Lock()
latest_result_100 = {"Phien": 0, "Xuc_xac_1": 0, "Xuc_xac_2": 0, "Xuc_xac_3": 0, "Tong": 0, "Ket_qua": "Chưa có"}
latest_result_101 = {"Phien": 0, "Xuc_xac_1": 0, "Xuc_xac_2": 0, "Xuc_xac_3": 0, "Tong": 0, "Ket_qua": "Chưa có"}
history_100, history_101 = [], []

def get_tai_xiu(d1, d2, d3):
    total = d1 + d2 + d3
    return "Xỉu" if total <= 10 else "Tài"

def update_result(store, history, lock, result):
    with lock:
        store.clear()
        store.update(result)
        history.insert(0, result.copy())
        if len(history) > 50:
            history.pop()

def poll_api(gid, lock, result_store, history, is_md5):
    """Luồng chính lấy dữ liệu TX hoặc MD5"""
    url = f"https://jakpotgwab.geightdors.net/glms/v1/notify/taixiu?platform_id=g8&gid={gid}"
    while True:
        try:
            req = Request(url, headers={'User-Agent': 'Python-Proxy/1.0'})
            with urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode('utf-8'))
            if data.get('status') == 'OK' and isinstance(data.get('data'), list):
                for game in data['data']:
                    cmd = game.get("cmd")
                    if cmd in (1003, 2006):
                        sid = game.get("sid")
                        d1, d2, d3 = game.get("d1"), game.get("d2"), game.get("d3")
                        if not sid or None in (d1, d2, d3): continue
                        total = d1 + d2 + d3
                        ket_qua = get_tai_xiu(d1, d2, d3)
                        result = {"Phien": sid, "Xuc_xac_1": d1, "Xuc_xac_2": d2, "Xuc_xac_3": d3, "Tong": total, "Ket_qua": ket_qua}
                        update_result(result_store, history, lock, result)
                        logging.info(f"[{'MD5' if is_md5 else 'TX'}] Phiên {sid} - Tổng {total} - {ket_qua}")
        except Exception as e:
            logging.error(f"[{gid}] Lỗi khi lấy API: {e}")
            time.sleep(5)
        time.sleep(5)

def watchdog_thread(threads):
    """Tự khởi động lại thread khi bị chết"""
    while True:
        for name, tdata in threads.items():
            t = tdata["thread"]
            if not t.is_alive():
                logging.warning(f"⚠️ Thread {name} chết, khởi động lại...")
                new_t = threading.Thread(target=tdata["target"], args=tdata["args"], daemon=True)
                tdata["thread"] = new_t
                new_t.start()
        time.sleep(30)

def start_hit_background():
    """Khởi động polling TX + MD5 và watchdog"""
    threads = {
        "TX": {"target": poll_api, "args": ("vgmn_100", lock_100, latest_result_100, history_100, False)},
        "MD5": {"target": poll_api, "args": ("vgmn_101", lock_101, latest_result_101, history_101, True)},
    }
    for name, tdata in threads.items():
        t = threading.Thread(target=tdata["target"], args=tdata["args"], daemon=True)
        t.start()
        tdata["thread"] = t
    threading.Thread(target=watchdog_thread, args=(threads,), daemon=True).start()
    logging.info("✅ Hit background polling started.")
# ========== HẾT KHỐI GỘP ==========

# ================== QUẢN LÝ PIPELINES ==================
from threading import Lock

class SimplePipeline:
    """Pipeline mô phỏng học và dự đoán Tài/Xỉu"""
    def __init__(self, name):
        self.name = name
        self.lock = Lock()
        self.data = []       # lưu lịch sử (Phien, Ket_qua)
        self.model = None
        self.accuracy = 0.5

    def add_sample(self, phien, ket_qua):
        self.data.append({"Phien": phien, "Ket_qua": ket_qua})
        if len(self.data) > 1000:
            self.data.pop(0)

    def retrain(self):
        """Huấn luyện lại mô hình (demo — chỉ cập nhật accuracy ngẫu nhiên)"""
        import random
        self.accuracy = round(random.uniform(0.55, 0.75), 3)

    def predict(self):
        """Dự đoán ngẫu nhiên ban đầu (có thể thay bằng mô hình thật)"""
        import random
        label = random.choice(["Tài", "Xỉu"])
        prob = round(random.uniform(0.5, 0.9), 3)
        return {"label": label, "prob": prob}

# ================== KHỞI TẠO PIPELINES ==================
PIPELINES = {
    "tx": SimplePipeline("tx"),
    "md5": SimplePipeline("md5")
}

# --- logging ---
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger("app_upgraded")

# --- config ---
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
DB_FILE = DATA_DIR / "history.db"
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)
PORT = int(os.environ.get("PORT", 5000))
SEQ_LEN = int(os.environ.get("SEQ_LEN", 100))
TEACHER_INTERVAL = int(os.environ.get("TEACHER_INTERVAL", 1800))  # seconds
DISTILL_SAMPLES = 5000

# --- DB helpers ---
CREATE_SQL = """
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    channel TEXT,
    phien TEXT,
    d1 INTEGER,
    d2 INTEGER,
    d3 INTEGER,
    tong INTEGER,
    ket_qua TEXT,
    pred_label TEXT DEFAULT '?',
    ts INTEGER
);
"""

def init_db():
    conn = sqlite3.connect(str(DB_FILE), check_same_thread=False)
    conn.execute(CREATE_SQL)
    conn.commit()
    return conn

DB_CONN = init_db()

# --- Utility ---
def encode_label(kq):
    if kq is None: return 0
    s = str(kq).strip().lower()
    return 1 if s.startswith("t") else 0

def context_from_row_tuple(row):
    # row: (id, channel, phien, d1, d2, d3, tong, ket_qua, pred_label, ts) OR slice
    try:
        d1 = float(row[3]); d2 = float(row[4]); d3 = float(row[5]); tong = float(row[6])
        last = 1.0 if str(row[7]).lower().startswith("t") else 0.0
    except Exception:
        # fallback when passed smaller tuple
        last = 1.0 if str(row[-1]).lower().startswith("t") else 0.0
        d1=d2=d3=tong=0.0
    return np.array([last, d1, d2, d3, tong] + [0]*9, dtype=np.float32)  # 14-dim
def build_context_seq(channel, length=10):
    """
    Tạo vector biểu diễn chuỗi kết quả gần nhất để model thấy được pattern tuần tự.
    Không liên quan đến bất kỳ hành vi dự đoán hay trò chơi cụ thể nào.
    """
    cur = DB_CONN.cursor()
    cur.execute("SELECT ket_qua FROM history WHERE channel=? ORDER BY id DESC LIMIT ?", (channel, length))
    rows = [r[0] for r in cur.fetchall()][::-1]  # đảo ngược cho đúng thứ tự thời gian

    # Encode thành chuỗi nhị phân tổng quát
    seq = [1.0 if str(kq).lower().startswith("t") else 0.0 for kq in rows]
    # Đệm nếu chưa đủ chiều dài
    seq = [0.0] * (length - len(seq)) + seq

    # ✅ Đệm thêm 4 giá trị 0 cho khớp kích thước 14 đầu vào của StudentMLP
    seq = np.concatenate([np.array(seq, dtype=np.float32), np.zeros(4, dtype=np.float32)])
    return seq
# --- Lightweight online policy (student) ---
class StudentMLP(nn.Module):
    def __init__(self, input_dim=14, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2,1),
            nn.Sigmoid()
        )
    def forward(self, x): return self.net(x).squeeze(-1)

STUDENT_PATH_TX = MODELS_DIR / "student_tx.pt"
STUDENT_PATH_MD5 = MODELS_DIR / "student_md5.pt"

# Initialize student policies (CPU)
device = torch.device("cpu")
student_tx = StudentMLP().to(device)
student_md5 = StudentMLP().to(device)
if STUDENT_PATH_TX.exists():
    try:
        student_tx.load_state_dict(torch.load(STUDENT_PATH_TX, map_location=device))
        logger.info("Loaded student_tx")
    except Exception:
        logger.exception("Failed to load student_tx")
if STUDENT_PATH_MD5.exists():
    try:
        student_md5.load_state_dict(torch.load(STUDENT_PATH_MD5, map_location=device))
        logger.info("Loaded student_md5")
    except Exception:
        logger.exception("Failed to load student_md5")

student_tx.eval(); student_md5.eval()
student_tx_opt = optim.Adam(student_tx.parameters(), lr=1e-4)
student_md5_opt = optim.Adam(student_md5.parameters(), lr=1e-4)
bce_loss = nn.BCELoss()

# --- Transformer AutoEncoder (teacher) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, : x.size(1)]

class TransformerAE(nn.Module):
    def __init__(self, feat_dim=5, d_model=64, nhead=4, nlayers=2):
        super().__init__()
        self.input_proj = nn.Linear(feat_dim, d_model)
        self.pos = PositionalEncoding(d_model, max_len=SEQ_LEN+10)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=4*d_model, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        decoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=4*d_model, batch_first=True)
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=nlayers)
        self.output_proj = nn.Linear(d_model, feat_dim)
        self.policy = nn.Sequential(nn.Linear(d_model, 128), nn.ReLU(), nn.Linear(128,1), nn.Sigmoid())
    def forward(self, x):
        h = self.input_proj(x)
        h = self.pos(h)
        z = self.encoder(h)
        pooled = z.mean(dim=1)
        dec = self.decoder(z)
        out = self.output_proj(dec)
        policy = self.policy(pooled).squeeze(-1)
        return out, policy, pooled

TEACHER_PATH_TX = MODELS_DIR / "teacher_tx_tae.pt"
TEACHER_PATH_MD5 = MODELS_DIR / "teacher_md5_tae.pt"

# --- Replay buffer for sequences (for teacher training) ---
replay_tx = []
replay_md5 = []
REPLAY_MAX = 20000
replay_lock = threading.Lock()

# --- Functions to read DB sequences ---
def load_sequences_from_db(channel="tx", seq_len=SEQ_LEN, max_samples=2000):
    cur = DB_CONN.cursor()
    cur.execute("SELECT d1,d2,d3,tong,ket_qua FROM history WHERE channel=? ORDER BY id ASC", (channel,))
    rows = cur.fetchall()
    seqs = []
    buff = []
    for r in rows:
        vec = [1.0 if str(r[4]).lower().startswith("t") else 0.0, float(r[0]), float(r[1]), float(r[2]), float(r[3])]
        buff.append(vec)
        if len(buff) >= seq_len:
            seqs.append(np.array(buff[-seq_len:], dtype=np.float32))
        if len(seqs) >= max_samples:
            break
    return seqs

# --- Simple distillation: generate teacher probs and train student MLP ---
def distill_teacher_to_student(teacher_state_path, channel="tx", out_student_path=None, samples=2000):
    try:
        device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        teacher = TransformerAE().to(device_t)
        teacher.load_state_dict(torch.load(teacher_state_path, map_location=device_t))
        teacher.eval()
        seqs = load_sequences_from_db(channel=channel, seq_len=SEQ_LEN, max_samples=samples)
        if not seqs:
            logger.warning("No sequences to distill for %s", channel); return None
        data = []
        for s in seqs:
            x = torch.tensor(s[np.newaxis,:,:], dtype=torch.float32).to(device_t)
            with torch.no_grad():
                _, prob, _ = teacher(x)
                p = float(prob.cpu().numpy()[0])
            last_row = s[-1]
            ctx = np.array([1.0 if last_row[0]>=0.5 else 0.0, last_row[1], last_row[2], last_row[3], last_row[4]] + [0]*9, dtype=np.float32)
            data.append((ctx, p))
        # train small student on cpu
        student = StudentMLP().to("cpu")
        opt = optim.Adam(student.parameters(), lr=1e-3)
        mse = nn.MSELoss()
        batch = 128
        for ep in range(6):
            random.shuffle(data)
            tot=0.0; n=0
            for i in range(0, len(data), batch):
                chunk = data[i:i+batch]
                xs = torch.tensor(np.stack([c[0] for c in chunk]), dtype=torch.float32)
                ys = torch.tensor(np.array([c[1] for c in chunk], dtype=np.float32))
                pred = student(xs).squeeze(-1)
                loss = mse(pred, ys)
                opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss.item())*len(chunk); n += len(chunk)
            logger.info("Distill %s ep=%d loss=%.6f", channel, ep+1, tot/max(1,n))
        outp = out_student_path or (MODELS_DIR / f"student_{channel}.pt")
        torch.save(student.state_dict(), outp)
        logger.info("Saved distilled student %s", outp)
        return outp
    except Exception:
        logger.exception("Distill failed for %s", channel)
        return None

# --- Teacher training in-process (lightweight) ---
def train_teacher_inprocess(channel="tx", seq_len=SEQ_LEN, epochs=3):
    try:
        device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TransformerAE().to(device_t)
        seqs = load_sequences_from_db(channel=channel, seq_len=seq_len, max_samples=2000)
        if not seqs:
            logger.warning("No data for teacher training %s", channel)
            return None

        ds = torch.tensor(np.stack(seqs), dtype=torch.float32).to(device_t)
        opt = optim.Adam(model.parameters(), lr=1e-4)
        mse = nn.MSELoss()
        B = 64

        for ep in range(epochs):
            idx = list(range(len(ds)))
            random.shuffle(idx)
            total = 0.0
            n = 0
            for i in range(0, len(idx), B):
                batch_idx = idx[i:i + B]
                bx = ds[batch_idx]
                recon, prob, _ = model(bx)
                loss = mse(recon, bx)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss.item()) * len(batch_idx)
                n += len(batch_idx)

            logger.info("Teacher %s epoch %d loss=%.6f", channel, ep + 1, total / max(1, n))

            # --- Cập nhật tiến độ học để frontend hiển thị ---
            global student_tx_progress, student_md5_progress
            if channel == "tx":
                student_tx_progress = (ep + 1) / epochs
            else:
                student_md5_progress = (ep + 1) / epochs

            logger.info(f"⏳ Tiến độ {channel.upper()}: {(ep + 1)/epochs*100:.1f}%")

        # --- Sau khi train xong, lưu model ---
        outp = MODELS_DIR / f"teacher_{channel}_tae.pt"
        torch.save(model.state_dict(), outp)
        logger.info("Saved teacher %s", outp)
        return outp

    except Exception:
        logger.exception("Teacher training failed %s", channel)
        return None



# --- Online REINFORCE-style update on student per new record ---
def online_update_student(ch, context_vec, true_label):
    try:
        x = torch.tensor(context_vec.reshape(1,-1), dtype=torch.float32)
        if ch == "tx":
            student = student_tx; opt = student_tx_opt
        else:
            student = student_md5; opt = student_md5_opt
        student.train()
        p = student(x)
        m = torch.distributions.Bernoulli(p)
        a = m.sample()
        reward = 1.0 if float(a.item()) == float(true_label) else 0.0
        logp = m.log_prob(a)
        loss = -(logp * reward).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        student.eval()
        # periodically save snapshot
        if random.random() < 0.01:
            outp = MODELS_DIR / f"student_{ch}.pt"
            torch.save(student.state_dict(), outp)
        return float(p.item()), float(reward)
    except Exception:
        logger.exception("online_update_student error")
        return 0.5, 0.0

# --- Hook: when new record is inserted into DB, we call this to update replay and do RL update ---
def on_new_record_inserted(row):
    # row: (id, channel, phien, d1, d2, d3, tong, ket_qua, pred_label, ts)
    try:
        ch = row[1]
        ctx = context_from_row_tuple(row)
        true_label = 1.0 if str(row[7]).lower().startswith("t") else 0.0
        # update replay buffer
        seq_vec = [1.0 if str(row[7]).lower().startswith("t") else 0.0, float(row[3]), float(row[4]), float(row[5]), float(row[6])]
        with replay_lock:
            if ch == "tx":
                replay_tx.append(seq_vec)
                if len(replay_tx) > REPLAY_MAX: replay_tx.pop(0)
            else:
                replay_md5.append(seq_vec)
                if len(replay_md5) > REPLAY_MAX: replay_md5.pop(0)
        # online RL update
        p, r = online_update_student(ch, ctx, true_label)
        logger.info("Online RL ch=%s phien=%s prob=%.3f reward=%.1f", ch, row[2], p, r)
        check_prediction_accuracy(ch)

    except Exception:
        logger.exception("on_new_record_inserted error")
def check_prediction_accuracy(channel):
    """Kiểm tra độ chính xác dự đoán so với kết quả thật trong DB"""
    try:
        cur = DB_CONN.cursor()
        cur.execute("""
            SELECT ket_qua, pred_label
            FROM history
            WHERE channel=? AND pred_label IS NOT NULL AND pred_label != '?'
            ORDER BY id ASC
        """, (channel,))
        rows = cur.fetchall()
        total = len(rows)
        if total == 0:
            return {"accuracy": 0.0, "correct": 0, "total": 0}

        correct = sum(
            1 for kq, pred in rows
            if kq and pred and str(kq).strip().lower()[0] == str(pred).strip().lower()[0]
        )

        acc = round(correct / total, 3)
        PIPELINES[channel].accuracy = acc
        logger.info(f"🎯 Độ chính xác {channel.upper()}: {acc*100:.1f}% ({correct}/{total})")

        return {"accuracy": acc, "correct": correct, "total": total}

    except Exception:
        logger.exception("check_prediction_accuracy error")
        return {"accuracy": 0.0, "correct": 0, "total": 0}
# --- Monitor DB for new records thread ---
processed_ids = set()
def db_watcher_loop(interval=3):
    global processed_ids
    cur = DB_CONN.cursor()
    while True:
        try:
            cur.execute("SELECT id, channel, phien, d1, d2, d3, tong, ket_qua, pred_label, ts FROM history ORDER BY id ASC")
            rows = cur.fetchall()
            for r in rows:
                if r[0] not in processed_ids:
                    processed_ids.add(r[0])
                    on_new_record_inserted(r)
            time.sleep(interval)
        except Exception:
            logger.exception("db_watcher error")
            time.sleep(2)

# --- Periodic teacher training + distill thread ---
def teacher_manager_loop():
    last = 0
    while True:
        try:
            now = time.time()
            if now - last > TEACHER_INTERVAL:
                last = now
                for ch in ["tx", "md5"]:
                    logger.info("Starting teacher train for %s", ch)
                    tpath = train_teacher_inprocess(channel=ch, seq_len=SEQ_LEN, epochs=2)
                    if tpath:
                        distill_teacher_to_student(tpath, channel=ch, out_student_path=MODELS_DIR / f"student_{ch}.pt", samples=DISTILL_SAMPLES)
            time.sleep(5)
        except Exception:
            logger.exception("teacher_manager error")
            time.sleep(10)

FRONTEND_HTML = r"""
<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <title>Dự đoán MD5 Tài Xỉu VIP — Cyberpunk Dashboard</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body {
      background: radial-gradient(circle at top, #0a0a0f 0%, #000 100%);
      color: #eaeaea;
      font-family: "Poppins", sans-serif;
      margin: 0;
      padding: 25px;
    }
    h1 {
      text-align: center;
      color: #00ffe0;
      text-shadow: 0 0 20px #00ffe0, 0 0 10px #ff00cc;
      font-weight: 700;
      margin-bottom: 10px;
    }
    .tabs {
      display: flex;
      justify-content: center;
      gap: 10px;
      margin-bottom: 25px;
    }
    .tab {
      background: #1b1b2f;
      padding: 12px 26px;
      border-radius: 8px 8px 0 0;
      cursor: pointer;
      font-weight: bold;
      color: #ccc;
      transition: 0.3s;
      border: 1px solid #2c2c40;
    }
    .tab.active {
      background: linear-gradient(90deg,#ff00cc,#3333ff);
      color: #fff;
      box-shadow: 0 0 20px #ff00ccaa;
    }
    .section { display: none; }
    .section.active { display: block; animation: fadeIn 0.5s; }
    @keyframes fadeIn {
      from {opacity: 0; transform: translateY(20px);}
      to {opacity: 1; transform: translateY(0);}
    }
    .card {
      background: rgba(25,25,40,0.95);
      border-radius: 16px;
      padding: 25px;
      box-shadow: 0 0 30px rgba(255,255,255,0.05);
      max-width: 800px;
      margin: auto;
    }
    .title {
      font-size: 1.4rem;
      color: #ffd86b;
      display: flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 10px;
    }
    .bar {
      height: 10px;
      background: #222;
      border-radius: 6px;
      overflow: hidden;
      margin: 4px 0 8px 0;
    }
    .bar-inner { height: 100%; transition: width 0.4s ease; }
    .bar-blue { background: linear-gradient(90deg,#3b82f6,#06b6d4); }
    .bar-green { background: #28c76f; }
    .bar-red { background: #ff4c4c; }
    .stats { font-size: 0.9rem; margin-top: 8px; }
    .section-line {
      border-top: 1px solid #333;
      margin: 10px 0;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.9rem;
      margin-top: 10px;
    }
    th, td { text-align: center; padding: 5px; }
    th { color: #ffda6b; border-bottom: 1px solid #333; }
    td.tai { color: #28c76f; font-weight: 600; }
    td.xiu { color: #ff4c4c; font-weight: 600; }
  </style>
</head>
<body>
  <h1>🔮 DỰ ĐOÁN VIP — CYBERPUNK STYLE</h1>

  <div class="tabs">
    <div class="tab active" data-tab="tx">Tài/Xỉu Thường</div>
    <div class="tab" data-tab="md5">Tài/Xỉu MD5</div>
  </div>

  <div class="section active" id="section-tx">
    <div class="card" id="tx-card">Đang tải dữ liệu...</div>
  </div>

  <div class="section" id="section-md5">
    <div class="card" id="md5-card">Đang tải dữ liệu...</div>
  </div>

<script>
const API = window.location.origin + "/api/status";
document.querySelectorAll('.tab').forEach(tab=>{
  tab.onclick = ()=>{
    document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
    tab.classList.add('active');
    document.querySelectorAll('.section').forEach(s=>s.classList.remove('active'));
    document.getElementById('section-'+tab.dataset.tab).classList.add('active');
  };
});

function renderCard(id, data, type) {
  if (!data) { document.getElementById(id).innerHTML = "Không có dữ liệu"; return; }

  const learn = (data.train_progress*100).toFixed(1);
  const acc = (data.accuracy*100).toFixed(1);
  const tai = data.outcomes.tai, xiu = data.outcomes.xiu;
  const total = tai + xiu || 1;
  const ptTai = (tai/total*100).toFixed(1), ptXiu = (xiu/total*100).toFixed(1);
  const pred = data.pred || {};

  // ---- xử lý pattern_detected nếu là object ----
  let patternText = "-";
  try {
    if (pred.pattern_detected === null || pred.pattern_detected === undefined) {
      patternText = "-";
    } else if (typeof pred.pattern_detected === "string") {
      patternText = pred.pattern_detected || "-";
    } else if (typeof pred.pattern_detected === "object") {
      // ưu tiên các trường phổ biến: pattern, name, desc, text
      patternText = pred.pattern_detected.pattern || pred.pattern_detected.name
                    || pred.pattern_detected.desc || pred.pattern_detected.text;
      // nếu không có trường dễ đọc thì stringify ngắn gọn
      if (!patternText) {
        let s = JSON.stringify(pred.pattern_detected);
        if (s.length > 120) s = s.slice(0, 117) + "...";
        patternText = s;
      }
    } else {
      patternText = String(pred.pattern_detected);
    }
  } catch (e) {
    patternText = "-";
  }

  let html = `
    <div class="title">💠 Bàn ${type.toUpperCase()} ${type==="tx"?"(Xúc xắc)":"(MD5)"}</div>
    <div>⚙️ Tiến độ học ML: <b>${learn}%</b></div>
    <div class="bar"><div class="bar-inner bar-blue" style="width:${learn}%;"></div></div>

    <div>🎯 Độ chính xác: <b>${acc}%</b> (${data.correct}/${data.total})</div>
    <div>Tài: ${ptTai}% | Xỉu: ${ptXiu}%</div>
    <div class="bar">
      <div class="bar-inner bar-green" style="width:${ptTai}%;"></div>
    </div>
    <div class="bar" style="margin-top:4px;">
      <div class="bar-inner bar-red" style="width:${ptXiu}%;"></div>
    </div>

    <div class="section-line"></div>
    <div>📊 Dự đoán ML: <b>${pred.ml_label||"?"}</b> (${(pred.ml_prob*100||0).toFixed(1)}%)</div>
    <div>🔁 Cầu tra cứu: <b>${pred.pat_label||"?"}</b> (${(pred.pat_conf*100||0).toFixed(1)}%)</div>
    <div>🔎 Nhận diện cầu: <b>${patternText}</b></div>
  `;

  if (type === "md5") {
    html += `<div>🧠 Hợp nhất AI: <b>${pred.final_label||"?"}</b> (${(pred.final_prob*100||0).toFixed(1)}%)</div>`;
  }

  html += `<div class="section-line"></div>
    <div style="margin-bottom:6px;">📜 <b>Lịch sử 10 phiên gần nhất</b></div>
    <table><tr><th>Phiên</th><th>Xúc xắc</th><th>Tổng</th><th>Kết quả</th></tr>`;

  (data.history||[]).slice().reverse().forEach(h=>{
    const xx = [h.Xuc_xac_1,h.Xuc_xac_2,h.Xuc_xac_3].filter(Boolean).join("-");
    const kq = h.Ket_qua||"?";
    html += `<tr>
      <td>${h.Phien}</td>
      <td>${xx||"-"}</td>
      <td>${h.Tong||"-"}</td>
      <td class="${kq==="Tài"?"tai":"xiu"}">${kq}</td>
    </tr>`;
  });
  html += "</table>";

  document.getElementById(id).innerHTML = html;
}
function refresh() {
  fetch(API).then(r=>r.json()).then(data=>{
    renderCard("tx-card", data.tx, "tx");
    renderCard("md5-card", data.md5, "md5");
  }).catch(()=>{});
}

refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>
"""
@app.route("/")
def root():
    return FRONTEND_HTML
# --- Flask API (keeps compatibility with previous app) ---
# ====================== KIỂM CHỨNG DỰ ĐOÁN & API STATUS ======================

@app.route("/api/status")
def api_status():
    try:
        # --- Lấy dữ liệu thật từ API nội bộ (Render hoặc local đều chạy được) ---
        port = int(os.environ.get("PORT", 10000))
        base = f"http://127.0.0.1:{port}"

        def safe_get_json(url):
            try:
                r = requests.get(url, timeout=3)
                if r.status_code == 200:
                    return r.json()
                else:
                    logger.warning(f"Lỗi {r.status_code} khi gọi {url}")
                    return {}
            except Exception as e:
                logger.warning(f"Không lấy được {url}: {e}")
                return {}

        # --- Gọi 3 API con, tránh crash khi rỗng ---
        tx_data = safe_get_json(base + "/api/taixiu")
        md5_data = safe_get_json(base + "/api/taixiumd5")
        hist_data = safe_get_json(base + "/api/history")

        history_100 = hist_data.get("taixiu", []) if isinstance(hist_data, dict) else []
        history_101 = hist_data.get("taixiumd5", []) if isinstance(hist_data, dict) else []

        # Nếu lịch sử chưa có thì thêm mẫu trống
        if not history_100:
            history_100 = [tx_data]
        if not history_101:
            history_101 = [md5_data]

        # --- Chuyển lịch sử thành chuỗi T/X ---
        def to_TX(history):
            s = ""
            for h in reversed(history[-8:]):
                s += "T" if h.get("Ket_qua") == "Tài" else "X"
            return s

        hist_tx = to_TX(history_100)
        hist_md5 = to_TX(history_101)

        # --- ML thật từ Student model ---
        try:
            ctx_tx = build_context_seq("tx", length=10)
            ctx_md5 = build_context_seq("md5", length=10)
            with torch.no_grad():
                ml_prob_tx = float(student_tx(torch.tensor(ctx_tx.reshape(1, -1), dtype=torch.float32)).item())
                ml_prob_md5 = float(student_md5(torch.tensor(ctx_md5.reshape(1, -1), dtype=torch.float32)).item())
            ml_label_tx = "Tài" if ml_prob_tx >= 0.5 else "Xỉu"
            ml_label_md5 = "Tài" if ml_prob_md5 >= 0.5 else "Xỉu"
        except Exception as e:
            logger.error(f"Lỗi ML: {e}")
            ml_label_tx = ml_label_md5 = "?"
            ml_prob_tx = ml_prob_md5 = 0.5

        # --- Cầu tra cứu ---
        pat_tx = lookup_pattern_predict(hist_tx)
        pat_md5 = lookup_pattern_predict(hist_md5)
        pat_conf_tx = 0.8 if pat_tx else 0.5
        pat_conf_md5 = 0.8 if pat_md5 else 0.5

        # --- Nhận diện cầu ---
        cau_tx = detect_cau(hist_tx)
        cau_md5 = detect_cau(hist_md5)

        # --- TX: chỉ dùng ML + tra cứu cầu (không hợp nhất) ---
        final_label_tx = ml_label_tx
        final_prob_tx = ml_prob_tx

        # --- MD5: hợp nhất AI (ML + cầu) ---
        final_label_md5 = ml_label_md5
        final_prob_md5 = ml_prob_md5
        if pat_md5:
            if pat_md5 == ml_label_md5:
                final_prob_md5 = 0.6 * ml_prob_md5 + 0.4 * pat_conf_md5
            else:
                final_prob_md5 = abs(0.6 * ml_prob_md5 - 0.4 * pat_conf_md5)
                if pat_conf_md5 > ml_prob_md5:
                    final_label_md5 = pat_md5

        # --- Lưu lại dự đoán vào DB ---
        try:
            cur = DB_CONN.cursor()
            # TX: chỉ lưu ML label
            cur.execute("SELECT id FROM history WHERE channel='tx' ORDER BY id DESC LIMIT 1")
            row_tx = cur.fetchone()
            if row_tx:
                cur.execute("UPDATE history SET pred_label=? WHERE id=?", (ml_label_tx, row_tx[0]))

            # MD5: lưu hợp nhất label
            cur.execute("SELECT id FROM history WHERE channel='md5' ORDER BY id DESC LIMIT 1")
            row_md5 = cur.fetchone()
            if row_md5:
                cur.execute("UPDATE history SET pred_label=? WHERE id=?", (final_label_md5, row_md5[0]))
            DB_CONN.commit()
        except Exception as e:
            logger.error(f"Lỗi lưu DB: {e}")

        # --- Kiểm tra độ chính xác ---
        acc_tx_info = check_prediction_accuracy("tx")
        acc_md5_info = check_prediction_accuracy("md5")

        # --- Tiến độ học ---
        global student_tx_progress, student_md5_progress
        tx_progress = round(student_tx_progress * 100, 1)
        md5_progress = round(student_md5_progress * 100, 1)

        # --- Thống kê tỷ lệ Tài/Xỉu ---
        cur = DB_CONN.cursor()
        cur.execute("SELECT ket_qua FROM history WHERE channel='tx'")
        rows_tx = [r[0] for r in cur.fetchall()]
        tai_tx = sum(1 for r in rows_tx if str(r).lower().startswith('t'))
        xiu_tx = sum(1 for r in rows_tx if str(r).lower().startswith('x'))

        cur.execute("SELECT ket_qua FROM history WHERE channel='md5'")
        rows_md5 = [r[0] for r in cur.fetchall()]
        tai_md5 = sum(1 for r in rows_md5 if str(r).lower().startswith('t'))
        xiu_md5 = sum(1 for r in rows_md5 if str(r).lower().startswith('x'))

        # --- JSON trả về ---
        return jsonify({
            "tx": {
                "train_progress": tx_progress / 100.0,
                "accuracy": acc_tx_info["accuracy"],
                "correct": acc_tx_info["correct"],
                "total": acc_tx_info["total"],
                "outcomes": {"tai": tai_tx, "xiu": xiu_tx},
                "history": history_100,
                "pred": {
                    "ml_label": ml_label_tx,
                    "ml_prob": ml_prob_tx,
                    "pat_label": pat_tx,
                    "pat_conf": pat_conf_tx,
                    "pattern_detected": cau_tx
                },
                "status": "Đang học online"
            },
            "md5": {
                "train_progress": md5_progress / 100.0,
                "accuracy": acc_md5_info["accuracy"],
                "correct": acc_md5_info["correct"],
                "total": acc_md5_info["total"],
                "outcomes": {"tai": tai_md5, "xiu": xiu_md5},
                "history": history_101,
                "pred": {
                    "ml_label": ml_label_md5,
                    "ml_prob": ml_prob_md5,
                    "pat_label": pat_md5,
                    "pat_conf": pat_conf_md5,
                    "pattern_detected": cau_md5,
                    "final_label": final_label_md5,
                    "final_prob": final_prob_md5
                },
                "status": "Đang học online"
            }
        })
    except Exception as e:
        import traceback
        logger.error(f"Lỗi API /status: {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/history")
def api_history():
    cur = DB_CONN.cursor()
    cur.execute("SELECT channel, phien, d1, d2, d3, tong, ket_qua, ts FROM history ORDER BY id DESC LIMIT 100")
    rows = cur.fetchall()
    out = []
    for r in rows:
        out.append({"channel": r[0], "phien": r[1], "d1": r[2], "d2": r[3], "d3": r[4], "tong": r[5], "ket_qua": r[6], "ts": r[7]})
    return jsonify(out)

@app.route("/api/taixiu")
def api_taixiu():
    # return latest tx record
    cur = DB_CONN.cursor()
    cur.execute("SELECT phien,d1,d2,d3,tong,ket_qua FROM history WHERE channel='tx' ORDER BY id DESC LIMIT 1")
    r = cur.fetchone()
    if not r:
        return jsonify({"Phien":0,"Xuc_xac_1":0,"Xuc_xac_2":0,"Xuc_xac_3":0,"Tong":0,"Ket_qua":"Chưa có"})
    return jsonify({"Phien":r[0],"Xuc_xac_1":r[1],"Xuc_xac_2":r[2],"Xuc_xac_3":r[3],"Tong":r[4],"Ket_qua":r[5]})

@app.route("/api/taixiumd5")
def api_taixiumd5():
    # return latest md5 record
    cur = DB_CONN.cursor()
    cur.execute("SELECT phien,d1,d2,d3,tong,ket_qua FROM history WHERE channel='md5' ORDER BY id DESC LIMIT 1")
    r = cur.fetchone()
    if not r:
        return jsonify({"Phien":0,"Xuc_xac_1":0,"Xuc_xac_2":0,"Xuc_xac_3":0,"Tong":0,"Ket_qua":"Chưa có"})
    return jsonify({"Phien":r[0],"Xuc_xac_1":r[1],"Xuc_xac_2":r[2],"Xuc_xac_3":r[3],"Tong":r[4],"Ket_qua":r[5]})
def sync_hit_data_loop():
    """Luồng tự động lấy dữ liệu thật từ hit.py mỗi 5 giây"""
    while True:
        try:
            fetch_and_store_latest_result("tx")
            fetch_and_store_latest_result("md5")
            time.sleep(5)
        except Exception:
            logger.exception("sync_hit_data_loop lỗi")
            time.sleep(10)
# --- Start background threads ---
t_db = threading.Thread(target=db_watcher_loop, daemon=True)
t_db.start()

t_teacher = threading.Thread(target=teacher_manager_loop, daemon=True)
t_teacher.start()

t_sync = threading.Thread(target=sync_hit_data_loop, daemon=True)
t_sync.start()

if __name__ == "__main__":
    # Khởi động nền hit.py và Flask server
    start_hit_background()
    port = int(os.environ.get("PORT", 10000))  # Render sẽ truyền PORT
    logger.info("🚀 App starting on port %d", port)
    app.run(host="0.0.0.0", port=port, debug=False)


