import os
import requests

# Set environment variables SEBELUM import library lain
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["STREAMLIT_SUPPRESS_CONFIG_WARNINGS"] = "true"

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import torch
import torch.nn as nn
import joblib
import json
import base64
import html
import math
import time

from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# Batasi thread PyTorch (penting untuk server kecil)
torch.set_num_threads(2)

# =================KONFIGURASI HALAMAN AWAL=================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_asset(filename):
    """Baca file dari folder assets/"""
    path = os.path.join(BASE_DIR, "assets", filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# ══════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AutoIN - SIPLah Data Flow Automation",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════
#  GLOBAL CSS — match HTML design exactly
# ══════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap');

/* hidden scrollbar */
html::-webkit-scrollbar { display: none !important; }
body::-webkit-scrollbar { display: none !important; }
* { scrollbar-width: none !important; }
html::-webkit-scrollbar { display: none; }
body::-webkit-scrollbar { display: none; }
* { scrollbar-width: none; }
html,body,[class*="css"]{font-family:'Sora',sans-serif!important}

.stApp{
  background:#EEF2F9!important;
  background-image:
    linear-gradient(rgba(26,107,240,.04) 1px,transparent 1px),
    linear-gradient(90deg,rgba(26,107,240,.04) 1px,transparent 1px)!important;
  background-size:48px 48px!important;
}
.block-container{padding:0!important;padding-top:0!important;margin-top:0!important;max-width:100%!important}
#MainMenu,footer,header,[data-testid="stToolbar"],[data-testid="stDecoration"],
[data-testid="stStatusWidget"]{display:none!important}
.stApp{margin-top:0!important;padding-top:0!important;}         
.stApp > div:first-child {margin-top: 0!important; padding-top: 0!important}
.block-container{padding-top: 0!important;margin-top:0!important;}                           
div[data-testid="stAppViewContainer"]{padding-top:0!important;margin-top:0!important;}            
                        
/* file uploader — light theme */
[data-testid="stFileUploader"]{background:#fff!important;border:1.5px dashed #C3D1E8!important;border-radius:12px!important;padding:16px!important}
[data-testid="stFileUploader"]:hover{border-color:#1A6BF0!important}
[data-testid="stFileDropzoneInstructions"]{color:#5C6E99!important}
[data-testid="baseButton-secondary"]{background:#fff!important;border:1px solid #C3D1E8!important;color:#0D1B3E!important;border-radius:8px!important;font-family:'Sora',sans-serif!important}
[data-testid="baseButton-secondary"]:hover{border-color:#1A6BF0!important;color:#1A6BF0!important}
[data-testid="baseButton-primary"]{background:linear-gradient(135deg,#1A6BF0,#0D3080)!important;border:none!important;border-radius:10px!important;font-family:'Sora',sans-serif!important;font-weight:700!important;font-size:14px!important;box-shadow:0 4px 16px rgba(26,107,240,.3)!important}
[data-testid="baseButton-primary"]:hover{box-shadow:0 6px 24px rgba(26,107,240,.45)!important;transform:translateY(-1px)!important}
[data-testid="stNotification"]{background:#fff!important;border:1px solid #D6E0F0!important;color:#0D1B3E!important}


/* scrollbar */
::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:#C8D5E8;border-radius:4px}
::-webkit-scrollbar-thumb:hover{background:#80B2FA}
            
h1 a{display:none!important}
a.anchor{display:none!important}
.stMarkdown h1 svg{display:none!important}
section.main > div {padding-top: 0 !important;margin-top: 0 !important;}            
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#  MODEL CONFIG
# ══════════════════════════════════════════════════════════
MODEL_FOLDER = "model_artifacts_3"
LEVELS = ["level_1","level_2","level_3","level_4","level_5","level_6"]
device = torch.device("cpu")

# ══════════════════════════════════════════════════════════
#  LOAD MODEL
# ══════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_model():

    with open(f"{MODEL_FOLDER}/config.json") as f:
        cfg = json.load(f)

    MODEL_NAME = cfg["MODEL_NAME"]
    lvls       = cfg["LEVELS"]
    MAX_LEN    = cfg["MAX_LEN"]

    le = joblib.load(f"{MODEL_FOLDER}/label_encoders.pkl")
    tree_maps = joblib.load(f"{MODEL_FOLDER}/tree_maps.pkl")

    nc = {l: len(le[l].classes_) for l in lvls}

    tok = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
        local_files_only=True
    )

    class HMC(nn.Module):
        def __init__(self):
            super().__init__()

            self.encoder = AutoModel.from_pretrained(MODEL_NAME)
            hidden_size = self.encoder.config.hidden_size

            self.classifiers = nn.ModuleDict({
                lvl: nn.Linear(hidden_size, nc[lvl]) for lvl in lvls
            })

        def forward(self, input_ids, attention_mask):
            out = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            cls_token = out.last_hidden_state[:, 0, :]

            return {
                lvl: self.classifiers[lvl](cls_token)
                for lvl in lvls
            }

    # 🔥 INIT MODEL
    model = HMC()

    model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )
    # 🔥 LOAD DULU BARU QUANTIZE (lebih aman & cepat)
    state_dict = torch.load(
        f"{MODEL_FOLDER}/model_HMC.pt_v.3",
        map_location="cpu"
    )
    model.load_state_dict(state_dict)

    # 🔥 QUANTIZATION (lebih optimal urutannya)
    # model = torch.quantization.quantize_dynamic(
    #     model,
    #     {torch.nn.Linear},
    #     dtype=torch.qint8
    # )

    model.eval()

    # 🔥 PRECOMPUTE MASK (hemat waktu inference)
    mask_cache = {}

    for (parent_lvl, child_lvl), mapping in tree_maps.items():
        for parent_id, children in mapping.items():
            key = (child_lvl, parent_id)
            mask = torch.full((nc[child_lvl],), -1e9)
            # 🔥 FIX: pastikan index list
            children = list(children)

            if len(children) > 0:
                mask[children] = 0
            mask_cache[key] = mask

    return model, tok, le, lvls, MAX_LEN, tree_maps, mask_cache


# =========================================
# OPTIMIZED PREDICTION
# =========================================
def run_prediction(df, progress_callback=None):
    
    model, tok, le, lvls, MAX_LEN, tree_maps, mask_cache = load_model()

    work = df.copy()

    # 🔥 lebih hemat daripada pandas string ops
    if "deskripsi" in work.columns:
        texts = [
            (str(a) + " " + str(b)).lower()
            for a, b in zip(
                work["nama item"].fillna(""),
                work["deskripsi"].fillna("")
            )
        ]
    else:
        texts = [
            str(a).lower()
            for a in work["nama item"].fillna("")
        ]

    total = len(texts)

    # 🔥 disesuaikan dengan RAM kecil
    batch_size = 128
    total_batch = math.ceil(total / batch_size)
    all_results = []

    with torch.inference_mode():

        for start_idx in range(0, total, batch_size):

            batch_texts = texts[start_idx:start_idx + batch_size]

            enc = tok(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=MAX_LEN,
                return_tensors="pt"
            )

            logits = model(
                enc["input_ids"],
                enc["attention_mask"]
            )

            # 🔥 langsung batch argmax awal
            preds = {
                lvl: torch.argmax(logits[lvl], dim=1)
                for lvl in lvls
            }

            batch_size_actual = enc["input_ids"].size(0)
            last_reported_pct = 0 
            for j in range(batch_size_actual):

                row = {}
                parent_id = None

                for lvl_idx, lvl in enumerate(lvls):

                    logit = logits[lvl][j]

                    if lvl_idx > 0 and parent_id is not None:

                        key = (lvl, parent_id)

                        if key in mask_cache:
                            logit = logit + mask_cache[key]
                        else:
                            break

                    pred = torch.argmax(logit).item()

                    row[lvl] = le[lvl].classes_[pred]
                    parent_id = pred

                all_results.append(row)

            # 🔥 progress update
            if progress_callback:
                done = min(start_idx + batch_size, total)
                pct = int((done / total) * 100)
                # progress_callback(pct, done, tota)
                current_milestone = (pct // 5) * 5

            if current_milestone > last_reported_pct or pct == 100:
                progress_callback(current_milestone if pct != 100 else 100, done, total)
                last_reported_pct = current_milestone

    pred_df = pd.DataFrame(all_results)

    drop_cols = [c for c in pred_df.columns if c in work.columns]

    result_df = pd.concat(
        [work.drop(columns=drop_cols, errors="ignore"), pred_df],
        axis=1
    )

    return result_df


# ===================== PAGE PERTAMA =====================
ALL_REQUIRED_COLS = [
    "nama item",
    "kawasan",
    "item dilihat",
    "item ditambahkan ke keranjang",
    "item yang dibeli",
    "pendapatan item"
]

def inject_upload_style():

    st.markdown("""
    <style>
    div[data-testid="stFileUploader"] {
        background: transparent !important;
        width: 100% !important;
        max-width: 1400px !important;
        margin: 40px auto !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
    }

    /* 🔥 WRAPPER DALAM */
    div[data-testid="stFileUploader"] > div {
        background: transparent !important;
        border: none !important;
        width: 100% !important;
        height: 320px !important;   /* 🔥 kunci tinggi */
        position: relative !important;
        box-shadow: none !important;
        padding: 0 !important;
                
    }
    /* 🔥 SAMAKAN DROPZONE DENGAN WRAPPER */
    [data-testid="stFileUploaderDropzone"] {
        height: 100% !important;
        min-height: unset !important;
        padding: 0 !important;
    }

    /* 🔥 HAPUS SEMUA ELEMEN BAWAAN YANG BIKIN SPACE */
    [data-testid="stFileUploader"] > div > div:last-child:not(:first-child) {
        display: none !important;
    }

    /* 🔥 HILANGKAN GAP INTERNAL STREAMLIT */
    section[data-testid="stFileUploader"] > div {
        gap: 0 !important;
    }
    
    /* 🔥 HAPUS TOTAL AREA FILE PREVIEW */
    [data-testid="stFileUploaderFile"],
    [data-testid="stFileUploaderFileList"] {
        display: none !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* 🔥 PAKSA WRAPPER NYA IKUT NGE-COLLAPSE */
    [data-testid="stFileUploader"] > div > div {
        height: 100% !important;
    }

    /* 🔥 KILL ELEMEN YANG MASIH NYISA SPACE */
    [data-testid="stFileUploader"] div:has([data-testid="stFileUploaderFile"]) {
        display: none !important;
    }

 
    @keyframes livePulse{0%,100%{opacity:1;box-shadow:0 0 0 0 rgba(14,166,106,.4)}
      50%{opacity:.7;box-shadow:0 0 0 4px rgba(14,166,106,0)}}
    @keyframes uf{0%,100%{transform:translateY(0)}50%{transform:translateY(-7px)}}

    [data-testid="stFileUploader"] label { display:none!important }
    [data-testid="stFileUploaderDropzoneInstructions"] { display:none!important }
    [data-testid="stFileUploaderDropzone"] small { display:none!important }
    section[data-testid="stFileUploader"] > div { gap:0!important }

    [data-testid="stFileUploaderDropzone"] {
      min-height: 320px !important;        /* 🔥 lebih kecil */
      max-width: 1400px !important;         /* 🔥 batasi lebar */
      margin: 40px auto !important;        /* 🔥 turun + center */
                
      display: flex !important;
      align-items: center !important;
      justify-content: center !important;
                
      background: #fff !important;
      border: 2px dashed #B8CCEF !important;
      border-radius: 16px !important;
      box-shadow: 0 1px 4px rgba(11,45,107,.07) !important;
      position: relative !important;
      cursor: pointer !important;
      transition: border-color .2s, background .2s !important;
      display: flex !important;
      align-items: center !important;
      justify-content: center !important;
      padding: 0 !important;
                
      flex: none !important; 
    }
    [data-testid="stFileUploaderDropzone"]:hover {
      border-color: #1A56DB !important;
      background: #E8F0FE !important;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] { display:none !important }
    [data-testid="stFileUploaderDropzone"] small { display:none !important }
    [data-testid="stFileUploaderDropzone"] > div:not(button) {
      display:none !important;
    }

    [data-testid="stFileUploaderDropzone"] button {
      position: absolute !important;
      inset: 0 !important;
      width: 100% !important;
      height: 100% !important;
      opacity: 0 !important;
      cursor: pointer !important;
      z-index: 20 !important;
      border: none !important;
      background: transparent !important;
    }

    [data-testid="stFileUploader"] label { display:none !important }
    [data-testid="stFileUploader"] > div { gap: 0 !important }

    [data-testid="stFileUploader"] > div > div:last-child:not(:first-child) {
      display: none !important;
    }
    </style>

    <div style="position:relative;pointer-events:none">
      <div id="uz-content" style="
        position:absolute;top:37px;left:0;right:0;
        text-align:center;padding:30px 20px 20px;
        pointer-events:none;z-index:1;
      ">
        <div style="width:48px;height:48px;margin:0 auto 10px;
              background:#EFF6FF;border:1px solid #BFDBFE;border-radius:14px;
              display:flex;align-items:center;justify-content:center;font-size:22px;
              animation:uf 3.2s ease-in-out infinite;
              box-shadow:0 4px 16px rgba(11,45,107,.10)">&#128202;</div>
        <div style="font-size:17px;font-weight:800;color:#0A1F4E;
              letter-spacing:-.4px;margin-bottom:6px">Upload File Data Import</div>
        <div style="font-size:12px;color:#5B7299;line-height:1.6;margin-bottom:12px">
          Seret &amp; lepas, atau <b style="color:#1A56DB">klik untuk memilih file</b><br>
          Semua insight &amp; visualisasi muncul otomatis setelah file diproses
        </div>
        <div style="display:inline-flex;align-items:center;gap:7px;padding:4px 14px;
              background:#F7F9FF;border:1px solid #D8E2F8;border-radius:20px;
              font-size:11px;color:#8FA3C8;margin-bottom:10px">
          Format:&nbsp;
          <span style="padding:2px 8px;border-radius:4px;font-size:10px;font-weight:800;
                background:#EFF6FF;color:#1A56DB;border:1px solid #BFDBFE">CSV</span>
          <span style="padding:2px 8px;border-radius:4px;font-size:10px;font-weight:800;
                background:#EFF6FF;color:#1A56DB;border:1px solid #BFDBFE">XLSX</span>
          <span style="padding:2px 8px;border-radius:4px;font-size:10px;font-weight:800;
                background:#EFF6FF;color:#1A56DB;border:1px solid #BFDBFE">XLS</span>
        </div>
        <div style="font-size:11px;color:#9AADC8;margin-bottom:6px">Kolom yang dibutuhkan:</div>
          <div style="display:flex;flex-wrap:wrap;justify-content:center;gap:6px">
            <span style="padding:4px 12px;border-radius:20px;font-size:10px;font-weight:700;
                background:rgba(26,107,240,.12);border:1px solid rgba(26,107,240,.3);
                color:#4D8FF5;font-family:monospace">Nama item</span>
            <span style="padding:4px 12px;border-radius:20px;font-size:10px;font-weight:700;
                background:rgba(26,107,240,.06);border:1px solid rgba(26,107,240,.18);
                color:#80B2FA;font-family:monospace">Kawasan</span>
            <span style="padding:4px 12px;border-radius:20px;font-size:10px;font-weight:700;
                background:rgba(26,107,240,.06);border:1px solid rgba(26,107,240,.18);
                color:#80B2FA;font-family:monospace">Item dilihat</span>
            <span style="padding:4px 12px;border-radius:20px;font-size:10px;font-weight:700;
                background:rgba(26,107,240,.06);border:1px solid rgba(26,107,240,.18);
                color:#80B2FA;font-family:monospace">Item ditambahkan ke keranjang</span>
            <span style="padding:4px 12px;border-radius:20px;font-size:10px;font-weight:700;
                background:rgba(26,107,240,.06);border:1px solid rgba(26,107,240,.18);
                color:#80B2FA;font-family:monospace">Item yang dibeli</span>
            <span style="padding:4px 12px;border-radius:20px;font-size:10px;font-weight:700;
                background:rgba(26,107,240,.06);border:1px solid rgba(26,107,240,.18);
                color:#80B2FA;font-family:monospace">Pendapatan item</span>
          </div>
          <div style="margin-top:10px;display:inline-flex;align-items:center;gap:5px;padding:4px 12px;
            background:#FEF9EC;border:1px solid #FDE68A;border-radius:20px;font-size:10px;color:#92400E;font-weight:600">
            ⚠️ Maks. ukuran file: 100 MB
          </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    
def render_layout():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(BASE_DIR, "template", "template_upload_file.xlsx")
    with open(template_path, "rb") as f:
        file_bytes = f.read()
    b64 = base64.b64encode(file_bytes).decode()

    # ── TOPBAR ──
    st.markdown("""
    <style>
    @keyframes livePulse{0%,100%{opacity:1;box-shadow:0 0 0 0 rgba(14,166,106,.4)}
      50%{opacity:.7;box-shadow:0 0 0 4px rgba(14,166,106,0)}}
    @keyframes uf{0%,100%{transform:translateY(0)}50%{transform:translateY(-7px)}}

    [data-testid="stFileUploader"] label { display:none!important }
    [data-testid="stFileUploaderDropzoneInstructions"] { display:none!important }
    [data-testid="stFileUploaderDropzone"] small { display:none!important }
    section[data-testid="stFileUploader"] > div { gap:0!important }

    [data-testid="stFileUploaderDropzone"] {
      background:#fff!important;
      border:2px dashed #B8CCEF!important;
      border-radius:16px!important;
      min-height:180px!important;
      text-align:center!important;
      transition:all .2s!important;
      cursor:pointer!important;
      box-shadow:0 1px 4px rgba(11,45,107,.07)!important;
      position:relative!important;
      display:flex!important;
      align-items:center!important;
      justify-content:center!important;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
      border-color:#1A56DB!important;
      background:#E8F0FE!important;
      box-shadow:0 4px 16px rgba(11,45,107,.10),0 0 0 4px rgba(26,86,219,.06)!important;
    }
    [data-testid="stFileUploaderDropzone"] button {
      position:absolute!important;
      inset:0!important;
      width:100%!important;
      height:100%!important;
      opacity:0!important;
      cursor:pointer!important;
      border:none!important;
      background:transparent!important;
      z-index:10!important;
    }
    </style>

    <div style="
         position: sticky;
         top: 0;
         z-index: 999;
         display:flex;
         align-items:center;
         justify-content:space-between;
         padding:0 32px;
         height:64px;
         background:rgba(255,255,255,.92);
         backdrop-filter:blur(16px);
         border-bottom:1px solid #D6E0F0;
         box-shadow:0 1px 3px rgba(13,27,62,.06),0 4px 12px rgba(13,27,62,.04);">
      <div>
        <div style="font-size:18px;font-weight:800;color:#0A2463;letter-spacing:-.4px;line-height:1.2">AutoIN</div>
        <div style="font-size:11px;font-weight:500;color:#5C6E99;letter-spacing:.3px">Sheira Devina Kevin</div>
      </div>
    <div style="display:flex;align-items:center;gap:10px;">
      <!-- BUTTON PANDUAN -->
      <a href="https://docs.google.com/document/d/1Mf9-M8Y0q0eEiL9qFSYp9cdHrfRs4q1D__aFZAVQ1Bo/edit?usp=sharing"
        target="_blank"
        style="
          display:inline-flex;
          align-items:center;
          height:34px;
          padding:0 14px;
          font-size:12px;
          font-weight:600;
          border-radius:10px;
          background:#111FA2;
          color:white;
          border:1px solid #111FA2;
          text-decoration:none;
          white-space:nowrap;
        ">
        Panduan Pengguna
      </a>
      <a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,""" + b64 + """"
        download="template_upload_file.xlsx"
        style="
          display:inline-flex;
          align-items:center;
          height:34px;
          padding:0 14px;
          font-size:12px;
          font-weight:600;
          border-radius:10px;
          background:#111FA2;
          color:white;
          border:1px solid #111FA2;
          text-decoration:none;
      ">
        Download Template
      </a>                    

      <!-- STATUS -->
      <div style="
          display:inline-flex;
          align-items:center;
          gap:6px;
          height:34px;
          padding:0 14px;
          background:rgba(156,163,175,.08);
          border:1px solid rgba(156,163,175,.25);
          border-radius:20px;
          font-size:11px;
          font-weight:600;
          color:#6B7280;
      ">
          <div style="width:6px;height:6px;border-radius:50%;background:#9CA3AF"></div>
          Menunggu Data
        </div>
      </div>
    </div>                       
    """, unsafe_allow_html=True)

    # HERO SECTION (INI YANG AKTIF)
    st.markdown("""
    <div class="render-layout-ignore" style="max-width:860px;margin:36px auto 0;padding:0 32px;text-align:center">
      
      <div style="display:inline-flex;align-items:center;gap:6px;padding:5px 14px;
           background:#EEF5FF;border:1px solid rgba(26,107,240,.2);border-radius:20px;
           font-size:10px;font-weight:700;color:#1A6BF0;letter-spacing:2px;
           text-transform:uppercase;margin-bottom:16px">
        <span style="width:5px;height:5px;border-radius:50%;background:#1A6BF0;"></span>
        ONE CLICK ALL DONE
      </div>

      <h1 style="font-size:48px;font-weight:800;color:#0D1B3E;
           letter-spacing:-.8px;line-height:1.2;margin-bottom:2px">
        SIPLah <span style="color:#1A6BF0">Data Flow Automation</span>
      </h1>

      <p style="font-size:15px;color:#5C6E99;margin-bottom:28px">
        " Empowering Data-driven E-Commerce Strategies "
      </p>

    </div>
    """, unsafe_allow_html=True)


def load_file(uploaded):
    try:
        # baca file
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)

        # normalisasi nama kolom
        df.columns = [
            str(c).strip().lower()
            .replace("_", " ")
            .replace("-", " ")
            .replace("/", " ")
            for c in df.columns
        ]

        # 🔥 return juga metadata biar gak tergantung uploaded lagi
        file_info = {
            "name": uploaded.name,
            "size": uploaded.size
        }

        return df, file_info, None

    except Exception as e:
        return None, None, str(e)
    
def validate_df(df):
    file_cols = list(df.columns)

    valid = [c for c in ALL_REQUIRED_COLS if c in file_cols]
    invalid = [c for c in ALL_REQUIRED_COLS if c not in file_cols]

    return {
        "is_valid": len(valid) == len(ALL_REQUIRED_COLS),
        "valid": valid,
        "invalid": invalid,
        "file_cols": file_cols
    }


def reset_state():
    st.session_state.preview_df_raw = None
    st.session_state.processing = False

    # reset file info
    st.session_state.last_file = ""
    st.session_state.last_file_size = 0

    # reset hasil
    st.session_state.result_df = None
    st.session_state.result_fname = ""

    # reset uploader biar benar-benar clear
    st.session_state.uploader_key = st.session_state.get("uploader_key", 0) + 1


def render_stepper(step, substep_label="", error_msg=None, container=None):

        AI_SVG = (
            '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" '
            'xmlns="http://www.w3.org/2000/svg" style="display:block">'
            '<circle cx="12" cy="12" r="3" fill="currentColor"/>'
            '<circle cx="4"  cy="6"  r="2" fill="currentColor" opacity=".7"/>'
            '<circle cx="20" cy="6"  r="2" fill="currentColor" opacity=".7"/>'
            '<circle cx="4"  cy="18" r="2" fill="currentColor" opacity=".7"/>'
            '<circle cx="20" cy="18" r="2" fill="currentColor" opacity=".7"/>'
            '<line x1="6"  y1="7"  x2="10" y2="10" stroke="currentColor" stroke-width="1.5" opacity=".6"/>'
            '<line x1="18" y1="7"  x2="14" y2="10" stroke="currentColor" stroke-width="1.5" opacity=".6"/>'
            '<line x1="6"  y1="17" x2="10" y2="14" stroke="currentColor" stroke-width="1.5" opacity=".6"/>'
            '<line x1="18" y1="17" x2="14" y2="14" stroke="currentColor" stroke-width="1.5" opacity=".6"/>'
            '</svg>'
        )

        STEPS = [
            ("&#128194;", "Baca File",         "#1A6BF0", "#EEF5FF", "#80B2FA"),
            ("&#128269;", "Validasi Kolom",    "#7C3AED", "#F5F3FF", "#C4B5FD"),
            (AI_SVG,      "AI Prediksi Level", "#6366F1", "#EEF2FF", "#A5B4FC"),
            ("&#128202;", "Siapkan Dashboard", "#D97706", "#FFFBEB", "#F59E0B"),
        ]

        DONE_COLOR = "#059669"
        DONE_BG    = "#ECFDF5"
        DONE_BDR   = "#34D399"
        PEND_COLOR = "#9AADC8"
        PEND_BG    = "#F4F7FD"
        PEND_BDR   = "#D6E0F0"

        uid = f"sp{step}"

        parts = []

        for i, (icon, label, col, bg, bdr) in enumerate(STEPS):

            done   = (step > i + 1) or (step == 5 and i <= 3)
            active = (step > 0) and (i == step - 1) and step < 5

            # ---------- ICON ----------
            if done:
                fc = DONE_COLOR
                icon_inner = (
                    f'<div style="width:54px;height:54px;border-radius:50%;background:{DONE_BG};'
                    f'border:2.5px solid {DONE_BDR};display:flex;align-items:center;justify-content:center;'
                    f'font-size:20px">&#9989;</div>'
                )
                spin_html = ""

            elif active:
                fc = col

                is_svg = icon.startswith('<svg')
                icon_content = (
                    f'<div style="color:{col}">{icon}</div>' if is_svg
                    else f'<span style="font-size:20px">{icon}</span>'
                )

                icon_inner = (
                    f'<div style="width:54px;height:54px;border-radius:50%;'
                    f'display:flex;align-items:center;justify-content:center;position:relative;">'
                    f'{icon_content}</div>'
                )

                spin_html = (
                    f'<div style="position:absolute;top:-3px;left:-3px;right:-3px;bottom:-3px;'
                    f'border-radius:50%;border:3px solid transparent;'
                    f'border-top-color:{col};animation:{uid}_spin .8s linear infinite"></div>'
                )

            else:
                fc = PEND_COLOR

                is_svg = icon.startswith('<svg')
                icon_content = (
                    f'<div style="color:{PEND_COLOR};opacity:.5">{icon}</div>' if is_svg
                    else f'<span style="font-size:20px;opacity:.5">{icon}</span>'
                )

                icon_inner = (
                    f'<div style="width:54px;height:54px;border-radius:50%;background:{PEND_BG};'
                    f'border:2.5px solid {PEND_BDR};display:flex;align-items:center;justify-content:center;">'
                    f'{icon_content}</div>'
                )
                spin_html = ""

            # ---------- SUB LABEL (🔥 ini yang penting) ----------
            sub = ""
            if active and substep_label:
                sub = (
                    f'<div style="font-size:11px;color:{fc};margin-top:4px;'
                    f'font-weight:700;font-family:monospace;">'
                    f'{substep_label}</div>'
                )

            parts.append(
                f'<div style="text-align:center;flex:1">'
                f'<div style="position:relative;width:54px;height:54px;margin:0 auto 10px">'
                f'{icon_inner}{spin_html}</div>'
                f'<div style="font-size:11px;font-weight:700;color:{fc}">{label}</div>'
                f'{sub}'
                f'</div>'
            )

        anim_css = f"""
        @keyframes {uid}_spin {{
            to {{ transform: rotate(360deg); }}
        }}
        """

        stepper_html = (
            f'<style>{anim_css}</style>'
            '<div style="display:flex;align-items:center;margin-bottom:30px;">'
            + "".join(parts) +
            '</div>'
        )

        if container:
            with container:
                st.markdown(stepper_html, unsafe_allow_html=True)
        else:
            st.markdown(stepper_html, unsafe_allow_html=True)
        # step_placeholder.markdown(stepper_html, unsafe_allow_html=True)

def render_upload_screen():
    render_layout()
    inject_upload_style()

    #SHE - MENGUBAH KEY AGAR DINAMIS UNTUK MERESET TOMBOL BATAL
    uploaded = st.file_uploader(
    "Upload",
    type=["csv","xlsx","xls"],
    key=f"uploader_{st.session_state.get('uploader_key', 0)}",
    label_visibility="collapsed"
)

    # =====================
    # INIT STATE
    # =====================
    if "step" not in st.session_state:
        st.session_state.step = 0

    if "processing" not in st.session_state:
        st.session_state.processing = False

    # 🔥 LOCK STEP BIAR GA BALIK KE STEP 1/2
    if st.session_state.get("processing", False):
        st.session_state.step = max(st.session_state.step, 3)

    stepper_container = st.empty()

    # helper biar clean
    def update_step(step, label=""):
        st.session_state.step = step
        render_stepper(step, label, container=stepper_container)

    
    # =====================
    # RESET LOGIC
    # =====================
    if (
        uploaded is None
        and st.session_state.get("preview_df_raw") is None
        and not st.session_state.get("processing", False)
    ):
        return None

    if (
        uploaded is None
        and st.session_state.get("preview_df_raw") is not None
        and not st.session_state.get("processing", False)
    ):
        reset_state()
        st.rerun()

    # =====================
    # SIZE VALIDATION
    # =====================
    if uploaded is not None:
        MAX_SIZE_MB = 100
        file_size_mb = uploaded.size / (1024 * 1024)

        if file_size_mb > MAX_SIZE_MB:
            st.markdown(f"""
            <div style="background:#FEF2F2;border:1px solid #FECACA;border-radius:12px;
                 padding:16px 20px;margin-top:12px;margin-bottom:32px;display:flex;align-items:center;gap:12px">
              <span style="font-size:24px">⚠️</span>
              <div>
                <div style="font-size:13px;font-weight:800;color:#991B1B">
                  Ukuran File Melebihi Batas Maksimal 100 MB
                </div>
                <div style="font-size:12px;color:#DC2626;margin-top:3px">
                  File kamu <b>{file_size_mb:.1f} MB</b>, melebihi batas maksimal.
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)
            return None

    # anchor
    st.markdown('<div id="top-anchor"></div>', unsafe_allow_html=True)

    # =====================
    # LOAD FILE
    # =====================
    if st.session_state.get("processing", False):
        df_raw = st.session_state.preview_df_raw
    else:
        if not uploaded and st.session_state.get("preview_df_raw") is not None:
            df_raw = st.session_state.preview_df_raw
        else:
            if uploaded is None:
                return None

            update_step(1, "Membaca file...")

            df_raw, file_info, err = load_file(uploaded)

            if err:
                update_step(0, f"Gagal membaca file: {err}")
                return None

            # simpan ke session_state
            st.session_state.preview_df_raw = df_raw
            st.session_state.last_file = file_info["name"]
            st.session_state.last_file_size = file_info["size"]

    # =====================
    # VALIDATION
    # =====================
    preview_container = st.empty()

    if not st.session_state.get("processing", False):
        update_step(2, "Memeriksa struktur kolom...")

    validation = validate_df(df_raw)

    is_valid = validation["is_valid"]
    valid_cols = validation["valid"]
    invalid_cols = validation["invalid"]
    file_cols = validation["file_cols"]

    valid_count = len(valid_cols)
    total_required = len(ALL_REQUIRED_COLS)

    missing_cols = [c for c in ALL_REQUIRED_COLS if c not in file_cols]

    # =====================
    # PREVIEW
    # =====================
    if not st.session_state.get("processing", False):

        with preview_container.container():
            
            error_html = f"""
            <div style="margin-top:6px;font-size:11px;color:#DC2626">
              Kolom yang belum ada (wajib): {", ".join(missing_cols) if missing_cols else "-"}
            </div>
            """ if not is_valid else ""

            preview_df = df_raw.head(10)
            preview_cols = file_cols

            # 🔹 THEAD
            thead = "".join([
                f"<th style='padding:8px 11px;font-size:10px;font-weight:700;"
                f"color:{'#059669' if c in valid_cols else '#DC2626'};"
                f"border-bottom:2px solid #D6E0F0;background:#F4F7FD'>"
                f"{'✔' if c in valid_cols else '✖'} {c}</th>"
                for c in preview_cols
            ])

            # 🔹 TBODY
            tbody = ""
            for _, row in preview_df.iterrows():
                tbody += "<tr>"
                for c in preview_cols:
                    val = row.get(c, "")
                    # tbody += f"<td style='padding:10px 14px;font-size:11px'>{html.escape(str(val)[:30])}</td>"
                    tbody += f"<td style='padding:10px 14px;font-size:11px;text-align:center;'>{html.escape(str(val)[:30])}</td>"
                tbody += "</tr>"

            # 🔹 FILE INFO
            file_name = st.session_state.get("last_file", "-")
            file_size = st.session_state.get("last_file_size", 0)
            file_size_mb = file_size / (1024 * 1024)

            # 🔹 STATUS
            status_color = "#059669" if is_valid else "#DC2626"
            status_icon = "✔" if is_valid else "⚠"

            # 🔥 FULL HTML RENDER (FIX UTAMA)
            components.html(f"""
            <style>
            @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700;800&display=swap');
            * {{font-family: 'Sora', sans-serif !important;}}
            </style>
            <div>
              <div style="background:#fff;
                          border:1px solid #D6E0F0;
                          border-radius:10px;
                          padding:16px;
                          max-width:1500px;
                          margin:auto;">

                <div style="width:95%;margin:auto;">

                  <div style="text-align:center;
                              font-weight:700;
                              font-size:16px;
                              margin-bottom:15px;
                              font-family:'Sora', sans-serif;">
                    Preview Data
                  </div>


                  <div style="font-size:12px;color:#475569;">
                    <b>File:</b> {file_name} ({file_size_mb:.2f} MB)
                  </div>

                  <div style="font-size:12px;color:#475569;margin-bottom:10px;">
                    <b>Baris:</b> {len(df_raw):,}
                  </div>

                  <div style="height:1px;background:#E2E8F0;margin:10px 0;"></div>

                  <table style="width:100%;
                                font-size:10px;
                                border-collapse:collapse;">
                    <thead>
                      <tr>{thead}</tr>
                    </thead>
                    <tbody>
                      {tbody}
                    </tbody>
                  </table>

                  <div style="margin-top:12px;
                              font-size:12px;
                              color:{status_color};
                              font-weight:600">
                    {status_icon} {valid_count}/{total_required} kolom sesuai format
                  </div>

                  {error_html}

                </div>
              </div>
            </div>
            """, height=600)

            # ======================
            # BUTTON
            # ======================
            st.markdown("""
            <style>
            @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700;800&display=swap');
            button, button * {
              font-family: 'Sora', sans-serif !important;
            }
            div[data-testid="stButton"] button {
              font-family: 'Sora', sans-serif !important;
              font-weight: 700 !important;
              font-size: 14px !important;
              position: relative !important;
              overflow: hidden !important;
              transition: transform 0.1s ease, box-shadow 0.1s ease !important;          
            }

            div[data-testid="stButton"] button * {
              font-size: 14px !important;
              font-weight: 700 !important;
              font-family: 'Sora', sans-serif !important;
            }
                        
            /* Ripple effect */
            div[data-testid="stButton"] button::after {
              content: '' !important;
              position: absolute !important;
              top: 50% !important;
              left: 50% !important;
              width: 0 !important;
              height: 0 !important;
              background: rgba(255, 255, 255, 0.35) !important;
              border-radius: 50% !important;
              transform: translate(-50%, -50%) !important;
              transition: width 0.4s ease, height 0.4s ease, opacity 0.4s ease !important;
              opacity: 0 !important;
              pointer-events: none !important;
            }

            div[data-testid="stButton"] button:active::after {
              width: 300px !important;
              height: 300px !important;
              opacity: 0 !important;
            }  

            /* Press-down saat klik */
            div[data-testid="stButton"] button:active {
              transform: scale(0.95) translateY(2px) !important;
              box-shadow: 0 1px 4px rgba(0,0,0,0.2) !important;
            }

            /* Hover lift */
            div[data-testid="stButton"] button:hover:not(:disabled) {
              transform: translateY(-2px) !important;
              box-shadow: 0 6px 16px rgba(0,0,0,0.18) !important;
            }  
                        
            /* MERAH */
            div[data-testid="stButton"] button[kind="secondary"],
            div[data-testid="stButton"] button[kind="secondary"]:hover,
            div[data-testid="stButton"] button[kind="secondary"]:active,
            div[data-testid="stButton"] button[kind="secondary"]:focus {
              background: linear-gradient(135deg, #FF0000, #FF0000) !important;
              border: none !important;
              color: #fff !important;
            }
            /* HIJAU */
            div[data-testid="stButton"] button[kind="primary"],
            div[data-testid="stButton"] button[kind="primary"]:hover,
            div[data-testid="stButton"] button[kind="primary"]:active,
            div[data-testid="stButton"] button[kind="primary"]:focus {
              background: linear-gradient(135deg, #009C30, #009C30) !important;
              border: none !important;
              color: #fff !important;
            }
            /* DISABLED */
            div[data-testid="stButton"] button[kind="primary"]:disabled {
              background: linear-gradient(135deg, #A7F3D0, #D1FAE5) !important;
              color: #065F46 !important;
              cursor: not-allowed !important;
              box-shadow: none !important;
              opacity: 0.7 !important;
              transform: none !important;
            }
            .button-center {
                display: flex;
                justify-content: center;
                gap: 12px;
                margin-top: 10px;
            }
            .button-center div[data-testid="stButton"] {
                width: auto !important;
                margin: 0 !important;
            }
            .button-center button {
                width: auto !important;
            }            
            </style>
            """, unsafe_allow_html=True)
        
            col_left, col_center, col_right = st.columns([3, 2, 3])
            with col_center:
                col_btn1, col_btn2 = st.columns([1, 1], gap="small")
                # with col_btn1:
                #     if st.button("Batalkan Proses", key="btn_batal", use_container_width=True):
                #         reset_state()
                #         update_step(0, "")
                #         st.rerun()
                with col_btn1:
                    if st.button("Batalkan Proses", key="btn_batal", use_container_width=True):
                        components.html("""
                        <script>window.top.location.reload();</script>
                        """, height=0)

                with col_btn2:
                    lanjut = st.button(
                        "Lanjutkan Proses",
                        key="btn_lanjut",
                        type="primary",
                        use_container_width=True,
                        disabled=not is_valid
                    )
            st.markdown("<div style='margin-bottom:120px'></div>", unsafe_allow_html=True)

            # 🔥 WAJIB ADA
            if lanjut:
                st.session_state.processing = True
                st.rerun()
            if not st.session_state.get("already_scrolled", False):
                components.html(
                    """
                    <script>
                    setTimeout(function() {
                        try {
                            var anchor = window.parent.document.getElementById('top-anchor');
                            if (anchor) {
                                anchor.scrollIntoView({ behavior: 'smooth', block: 'start' });
                            } else {
                                window.parent.scrollTo({ top: 0, behavior: 'smooth' });
                            }
                        } catch(e) {
                            window.parent.scrollTo({ top: 0, behavior: 'smooth' });
                        }
                    }, 300);
                    </script>
                    """,
                    height=0,
                    scrolling=False
                )
                st.session_state.already_scrolled = True
            # stop di sini kalau belum lanjut
            return 
        
    # =====================
    # PROCESS AI
    # =====================
    total_data = len(df_raw)

    def on_progress(pct, done, total_data):
        update_step(3, f"{pct}%")

    # update_step(3, f"0% · 0 / {total_data:,} data")
    update_step(3, f"0%")

    result_df = run_prediction(df_raw, progress_callback=on_progress)

    # =====================
    # FINAL STEP
    # =====================
    update_step(4, "Menyiapkan visualisasi...")

    return result_df, st.session_state.get("last_file", "")
 

# ===================== PAGE KEDUA =====================
PAL  = ["#f5a623","#60a5fa","#4ade80","#a78bfa","#f472b6","#fb923c","#34d399","#e879f9"]
TAGS = ["t0","t1","t2","t3","t4","t5","t6"]

ALL_COLS = [
    "nama item","kawasan","level_1","level_2","level_3","level_4","level_5","level_6","item dilihat","item ditambahkan ke keranjang","item yang dibeli","pendapatan item"]

def safe_n(v):
    try:
        s = str(v).replace("Rp","").replace(" ","").strip()
        if "," in s and "." in s:
            s = s.replace(".","").replace(",",".")
        else:
            s = s.replace(",","")
        return float(s)
    except:
        return 0.0

def fmt_rp(n):
    if n >= 1e9: return f"Rp {n/1e9:.2f} M"
    if n >= 1e6: return f"Rp {n/1e6:.1f} Jt"
    if n >= 1e3: return f"Rp {n/1e3:.1f} rb"
    return f"Rp {int(n):,}"

def fmt_n(n):
    if n >= 1e6: return f"{n/1e6:.1f}M"
    if n >= 1e3: return f"{n/1e3:.1f} rb"
    return f"{int(n):,}"

def df_to_js(df):
    """Convert DataFrame to safe JSON for embedding in JS."""
    import json as _json, math as _math
    cols = [c for c in ALL_COLS if c in df.columns]
    rows = []
    for _, row in df.iterrows():
        obj = {}
        for c in cols:
            val = row.get(c, "")
            if val is None or (isinstance(val, float) and _math.isnan(val)):
                obj[c] = ""
            else:
                obj[c] = str(val).strip()
        rows.append(obj)
    return _json.dumps(rows, ensure_ascii=False)

def build_dashboard_html(df, fname):
    import json as _json, base64 as _b64

    js_data  = df_to_js(df)
    js_fname = fname.replace('"', "'")
    js_nrows = len(df)

    cols_avail = [c for c in ALL_COLS if c in df.columns]
    csv_str    = ",".join(f'"{c}"' for c in cols_avail) + "\n"
    for _, row in df.iterrows():
        csv_str += ",".join(f'"{str(row.get(c,"")).replace(chr(34),chr(39))}"' for c in cols_avail) + "\n"
    csv_b64 = _b64.b64encode(csv_str.encode("utf-8")).decode("utf-8")

    #<<<<<< INI COBA DIBEDAH

    html = HTML_TEMPLATE
    html = html.replace('FILE_NAME', js_fname)
    html = html.replace('FILE_ROWS', f'{js_nrows:,}')
    html = html.replace('__JS_DATA__', js_data)
    html = html.replace("'__CSV_B64__'", f"'{csv_b64}'")

    return html

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<!-- ======================================== HEAD ======================================== -->
<!-- ======================================== BAGIAN (STYLE) CSS ======================================== -->
<link href="https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#EEF2F9;--sur:#FFFFFF;--sur2:#F4F7FD;--bdr:#D6E0F0;--bdr2:#C3D1E8;
  --b900:#0A2463;--b800:#0D3080;--b700:#1045A8;--b600:#1558CC;--b500:#1A6BF0;
  --b400:#4D8FF5;--b300:#80B2FA;--b200:#B3D0FC;--b100:#D9EAFF;--b50:#EEF5FF;
  --acc:#1A6BF0;--acc2:#0D3080;
  --grn:#0EA66A;--red:#E53E3E;--amber:#D97706;
  --tx:#0D1B3E;--tx2:#2D3F6B;--mut:#5C6E99;--dim:#9AADC8;--dimr:#C8D5E8;
  --fn:'Sora',sans-serif;--mo:'IBM Plex Mono',monospace;
  --radius:14px;
  --sh:0 1px 3px rgba(13,27,62,.06),0 4px 12px rgba(13,27,62,.06);
  --sh-glow:0 0 0 1px rgba(26,107,240,.15),0 8px 32px rgba(26,107,240,.12);
}
html{background:var(--bg);overflow-x:hidden;overflow-y:auto}
body{font-family:var(--fn);color:var(--tx);background:var(--bg);min-height:100vh;position:relative;overflow:visible}
body::before{content:'';position:fixed;inset:0;pointer-events:none;z-index:0;
  background-image:linear-gradient(rgba(26,107,240,.04) 1px,transparent 1px),
  linear-gradient(90deg,rgba(26,107,240,.04) 1px,transparent 1px);
  background-size:48px 48px}
.orb{position:fixed;border-radius:50%;filter:blur(80px);pointer-events:none;z-index:0}
.orb1{width:500px;height:500px;background:rgba(26,107,240,.08);top:-120px;right:-100px}
.orb2{width:350px;height:350px;background:rgba(10,36,99,.05);bottom:0;left:-80px}
.orb3{width:250px;height:250px;background:rgba(14,166,106,.06);top:40%;right:15%}
.wrap{position:relative;z-index:1;max-width:1440px;margin:0 auto;padding:0 32px 80px}

/* ======================================== TOPBAR ======================================== */
.topbar{display:flex;align-items:center;justify-content:space-between;
  padding:0 32px;height:64px;background:rgba(255,255,255,.85);
  backdrop-filter:blur(16px);border-bottom:1px solid var(--bdr);
  position:sticky;top:0;z-index:200;
  box-shadow:0 1px 0 var(--bdr),var(--sh)}
.logo-name .bot{font-size:18px;font-weight:800;color:var(--b900);letter-spacing:-.4px;line-height:1.2}
.logo-name .sub{font-size:11px;font-weight:500;color:var(--mut);letter-spacing:.3px}
.topbar-right{display:flex;align-items:center;gap:10px}
.badge-live{display:flex;align-items:center;gap:5px;padding:5px 12px;
  background:linear-gradient(135deg,rgba(14,166,106,.1),rgba(14,166,106,.06));
  border:1px solid rgba(14,166,106,.25);border-radius:20px;font-size:11px;font-weight:600;color:#0A7A50}
.live-dot{width:6px;height:6px;border-radius:50%;background:var(--grn);animation:livePulse 2s infinite}
@keyframes livePulse{0%,100%{opacity:1;box-shadow:0 0 0 0 rgba(14,166,106,.4)}50%{opacity:.7;box-shadow:0 0 0 4px rgba(14,166,106,0)}}
.btn{padding:8px 18px;border-radius:9px;font-size:12px;font-weight:600;cursor:pointer;border:none;transition:all .15s;font-family:var(--fn)}
.btn-out{background:var(--sur);border:1px solid var(--bdr2);color:var(--tx2);box-shadow:var(--sh)}
.btn-out:hover{border-color:var(--acc);color:var(--acc)}
.btn-prim{background:linear-gradient(135deg,var(--b500),var(--b700));color:#fff;box-shadow:0 2px 8px rgba(26,107,240,.3)}
.btn-prim:hover{box-shadow:0 4px 16px rgba(26,107,240,.4);transform:translateY(-1px)}

/* ======================================== PAGE HEADER ======================================== */
.page-hdr{padding:28px 0 20px;display:flex;align-items:center;justify-content:space-between}
.page-hdr-l h1{font-size:26px;font-weight:800;color:var(--b900);letter-spacing:-.6px;line-height:1.2}
.page-hdr-l h1 span{color:var(--acc)}
.page-hdr-l p{font-size:13px;color:var(--mut);margin-top:4px}

/* ======================================== FILE BANNER ========================================*/
.fbanner{display:flex;align-items:center;justify-content:space-between;
  background:linear-gradient(135deg,rgba(26,107,240,.06),rgba(10,36,99,.04));
  border:1px solid rgba(26,107,240,.2);border-radius:var(--radius);
  padding:12px 18px;margin-bottom:28px;position:relative;overflow:hidden}

.fbanner::before{content:'';position:absolute;left:0;top:0;bottom:0;width:4px;
  background:linear-gradient(180deg,var(--b500),var(--b800));border-radius:4px 0 0 4px}

.fbanner-l{display:flex;align-items:center;gap:10px;padding-left:8px}

.fbanner-icon{width:32px;height:32px;background:var(--b50);border:1px solid var(--b200);
  border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:14px;flex-shrink:0}

.fbanner-name{font-size:13px;font-weight:700;color:var(--b800)}

.fbanner-meta{font-size:11px;color:var(--mut);margin-top:1px}

.ai-pill{display:inline-flex;align-items:center;gap:4px;padding:3px 9px;
  background:rgba(109,40,217,.08);border:1px solid rgba(109,40,217,.2);
  border-radius:20px;font-size:10px;font-weight:700;color:#5B21B6}

/* ======================================== EDIT FBANNER ======================================== */
.fbanner-r{
  font-size:11px;
  color:var(--mut);
  font-family:var(--mo);

  display:flex;              /* ⬅️ TAMBAH */
  align-items:center;        /* ⬅️ TAMBAH */
  gap:10px;                  /* ⬅️ TAMBAH */
  position:relative;         /* ⬅️ TAMBAH */
  z-index:2;                 /* ⬅️ TAMBAH */
}

/* ======================================== EDIT FBANNER ======================================== */
.fbanner-r button{
  background:#fff;
  border:1px solid #d1d5db;
  border-radius:8px;
  padding:6px 10px;
  font-size:11px;
  cursor:pointer;
  white-space:nowrap;
}

/* ======================================== SECTION LABEL ======================================== */
.sl{font-size:10px;font-weight:700;color:var(--dim);letter-spacing:2.5px;
  text-transform:uppercase;margin:28px 0 16px;display:flex;align-items:center;gap:10px}
.sl::after{content:'';flex:1;height:1px;background:var(--bdr)}

/* ======================================== KPI CARDS ========================================*/
.ig{display:grid;grid-template-columns:repeat(5,1fr);gap:14px;margin-bottom:28px}
.ic{background:var(--sur);border:1px solid var(--bdr);border-radius:var(--radius);
  padding:20px;position:relative;overflow:hidden;box-shadow:var(--sh);transition:all .25s;cursor:default}
.ic:hover{box-shadow:var(--sh-glow);transform:translateY(-3px);border-color:rgba(26,107,240,.3)}
.ic::after{content:'';position:absolute;top:0;left:-100%;width:60%;height:100%;
  background:linear-gradient(105deg,transparent,rgba(255,255,255,.5),transparent);
  transition:left .5s ease;pointer-events:none}
.ic:hover::after{left:140%}
.ic-top{position:absolute;top:0;left:0;right:0;height:3px;border-radius:14px 14px 0 0}
.ic:nth-child(1) .ic-top{background:linear-gradient(90deg,var(--b500),var(--b300))}
.ic:nth-child(2) .ic-top{background:linear-gradient(90deg,#7C3AED,#C4B5FD)}
.ic:nth-child(3) .ic-top{background:linear-gradient(90deg,var(--grn),#6EE7B7)}
.ic:nth-child(4) .ic-top{background:linear-gradient(90deg,var(--amber),#FDE68A)}
.ic:nth-child(5) .ic-top{background:linear-gradient(90deg,var(--red),#FCA5A5)}
.ic-icon{width:36px;height:36px;border-radius:9px;display:flex;align-items:center;justify-content:center;font-size:16px;margin-bottom:12px}
.ic:nth-child(1) .ic-icon{background:var(--b50);border:1px solid var(--b100)}
.ic:nth-child(2) .ic-icon{background:#F5F3FF;border:1px solid #DDD6FE}
.ic:nth-child(3) .ic-icon{background:#ECFDF5;border:1px solid #A7F3D0}
.ic:nth-child(4) .ic-icon{background:#FFFBEB;border:1px solid #FDE68A}
.ic:nth-child(5) .ic-icon{background:#FEF2F2;border:1px solid #FECACA}
.ic-label{font-size:10px;font-weight:600;color:var(--mut);letter-spacing:.5px;text-transform:uppercase;margin-bottom:5px}
.ic-value{font-size:20px;font-weight:800;color:var(--tx);letter-spacing:-.5px;line-height:1.2}
.ic-sub{font-size:11px;color:var(--dim);margin-top:4px;font-family:var(--mo)}

/* ======================================== PANEL ======================================== */
.panel{background:var(--sur);border:1px solid var(--bdr);border-radius:var(--radius);
  padding:22px;box-shadow:var(--sh);position:relative;overflow:hidden}
.panel::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent,rgba(26,107,240,.15),transparent)}
.ph3{display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:18px;gap:12px;flex-wrap:wrap}
.pti{font-size:14px;font-weight:700;color:var(--b900)}
.psu{font-size:11px;color:var(--mut);margin-top:3px}
.chip-grp{display:flex;flex-wrap:wrap;gap:4px}
.chip-sm{padding:4px 11px;border-radius:7px;font-size:11px;font-weight:600;cursor:pointer;
  border:1px solid var(--bdr2);background:var(--sur2);color:var(--mut);transition:all .15s;font-family:var(--fn)}
.chip-sm:hover{border-color:var(--acc);color:var(--acc);background:var(--b50)}
.chip-sm.on{background:linear-gradient(135deg,var(--b50),rgba(26,107,240,.08));
  border-color:rgba(26,107,240,.35);color:var(--acc);font-weight:700;
  box-shadow:inset 0 1px 3px rgba(26,107,240,.1)}

/* ======================================== HORIZONTAL BAR CHART ======================================== */
.hbar-row{display:flex;align-items:center;gap:8px;margin-bottom:8px}
.hbar-rank{font-family:var(--mo);font-size:10px;color:var(--dim);width:16px;text-align:right;font-weight:500}
.hbar-label{font-size:11px;color:var(--mut);width:130px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;flex-shrink:0}
.hbar-track{flex:1;background:var(--sur2);border:1px solid var(--bdr);border-radius:6px;height:18px;overflow:hidden;position:relative}
.hbar-fill{height:100%;border-radius:5px;display:flex;align-items:center;padding-left:10px;
  transition:width .7s cubic-bezier(.4,0,.2,1);position:relative;overflow:hidden}
.hbar-fill::after{content:'';position:absolute;top:0;left:-100%;width:50%;height:100%;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,.25),transparent);
  animation:barShimmer 2.5s infinite}
@keyframes barShimmer{0%{left:-100%}100%{left:200%}}
.hbar-fill span{font-size:10px;font-weight:700;color:rgba(255,255,255,.9);position:relative;z-index:1}
.hbar-val{font-family:var(--mo);font-size:11px;color:var(--tx2);font-weight:600;width:75px;text-align:right;flex-shrink:0}

/* ======================================== TOP 10 LIST ======================================== */
.t10-item{display:flex;align-items:center;gap:8px;padding:6px 0;border-bottom:1px solid var(--bdr)}
.t10-item:last-child{border-bottom:none}
.t10-rank{width:20px;height:20px;border-radius:5px;display:flex;align-items:center;justify-content:center;
  font-family:var(--mo);font-size:10px;font-weight:700;flex-shrink:0;background:var(--sur2);color:var(--mut)}
.t10-rank.r1{background:linear-gradient(135deg,#FDE68A,#F59E0B);color:#78350F}
.t10-rank.r2{background:linear-gradient(135deg,#E2E8F0,#CBD5E1);color:#475569}
.t10-rank.r3{background:linear-gradient(135deg,#FED7AA,#F97316);color:#7C2D12}
.t10-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
.t10-name{font-size:11px;font-weight:600;color:var(--tx);width:80px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;flex-shrink:0}
.t10-track{flex:1;height:6px;background:var(--sur2);border-radius:4px;overflow:hidden;border:1px solid var(--bdr)}
.t10-fill{height:100%;border-radius:4px;transition:width .6s ease}
.t10-val{font-family:var(--mo);font-size:10px;color:var(--mut);width:58px;text-align:right;flex-shrink:0}

/* ======================================== FUNNEL ========================================*/
.funnel-item{margin-bottom:4px}
.funnel-lbl{display:flex;justify-content:space-between;margin-bottom:5px}
.funnel-lname{font-size:12px;font-weight:600;color:var(--tx)}
.funnel-lval{font-family:var(--mo);font-size:12px;color:var(--mut)}
.funnel-track{height:28px;background:var(--sur2);border:1px solid var(--bdr);border-radius:8px;overflow:hidden}
.funnel-fill{height:100%;border-radius:7px;display:flex;align-items:center;padding-left:12px;
  font-size:11px;font-weight:700;color:#fff;transition:width .7s ease;position:relative;overflow:hidden}
.funnel-fill::after{content:'';position:absolute;top:0;left:-100%;width:50%;height:100%;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,.2),transparent);animation:barShimmer 3s infinite}
.funnel-drop{font-size:11px;color:var(--red);text-align:right;margin-top:2px;margin-bottom:8px;font-family:var(--mo)}

/* ======================================== GRID / FILTER ======================================== */
.grid-layout{display:grid;grid-template-columns:1fr 240px;gap:16px;align-items:start}
.filter-panel{background:var(--sur);border:1px solid var(--bdr);border-radius:var(--radius);
  padding:18px;position:sticky;top:80px;box-shadow:var(--sh)}
.fp-hdr{font-size:13px;font-weight:700;color:var(--b900);margin-bottom:14px;display:flex;align-items:center;justify-content:space-between}
.fp-clear{font-size:11px;font-weight:600;color:var(--acc);cursor:pointer;background:none;border:none;font-family:var(--fn)}
.fp-lbl{font-size:10px;font-weight:700;color:var(--dim);letter-spacing:1.5px;text-transform:uppercase;margin-bottom:7px;margin-top:12px}
.fp-chips{display:flex;flex-wrap:wrap;gap:4px}
.fpc{display:inline-block;padding:4px 10px;border-radius:6px;background:var(--sur2);
  border:1px solid var(--bdr);font-size:11px;color:var(--mut);cursor:pointer;margin:2px;transition:all .15s;user-select:none}
.fpc:hover{border-color:var(--acc);color:var(--acc)}
.fpc.on{background:var(--b50);border-color:var(--b300);color:var(--acc);font-weight:700}
.fp-sel{width:100%;padding:7px 10px;background:var(--sur);border:1px solid var(--bdr2);
  border-radius:8px;color:var(--tx);font-size:12px;font-family:var(--fn);outline:none;
  -webkit-appearance:none;appearance:none;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='%239AADC8' stroke-width='2'%3E%3Cpath d='M6 9l6 6 6-6'/%3E%3C/svg%3E");
  background-repeat:no-repeat;background-position:right 10px center;padding-right:28px}
.fp-div{height:1px;background:var(--bdr);margin:12px 0}
.fp-apply{width:100%;padding:10px;margin-top:4px;
  background:linear-gradient(135deg,var(--b500),var(--b800));border:none;border-radius:9px;
  color:#fff;font-size:12px;font-weight:700;cursor:pointer;font-family:var(--fn);
  box-shadow:0 2px 8px rgba(26,107,240,.25);transition:all .15s}
.fp-apply:hover{box-shadow:0 4px 16px rgba(26,107,240,.35);transform:translateY(-1px)}
.col-chip{padding:3px 8px;border-radius:5px;background:var(--sur2);border:1px solid var(--bdr);
  font-size:10px;color:var(--mut);cursor:pointer;transition:all .15s;user-select:none}
.col-chip.on{background:var(--b50);border-color:var(--b300);color:var(--acc)}
.af-badge{display:none;background:var(--acc);color:#fff;font-size:9px;font-weight:800;
  width:16px;height:16px;border-radius:50%;align-items:center;justify-content:center;margin-left:5px}
.af-badge.show{display:flex}

/* ======================================== TABLE ======================================== */
.sbar{display:flex;align-items:center;gap:8px;background:var(--sur2);
  border:1px solid var(--bdr2);border-radius:9px;padding:7px 14px}
.sbar input{background:transparent;border:none;outline:none;font-size:13px;color:var(--tx);font-family:var(--fn);width:100%}
.sbar input::placeholder{color:var(--dim)}
.dtbl{width:100%;border-collapse:collapse;font-size:12px}
.dtbl th{text-align:left;font-size:10px;font-weight:700;color:var(--mut);letter-spacing:.8px;
  text-transform:uppercase;padding:10px 13px;border-bottom:2px solid var(--bdr);
  background:linear-gradient(180deg,var(--sur2),var(--bg));white-space:nowrap;
  position:sticky;top:0;z-index:10;}
.dtbl td{padding:10px 13px;border-bottom:1px solid var(--bdr);vertical-align:middle;color:var(--tx)}
.dtbl tr:hover td{background:linear-gradient(90deg,var(--b50),transparent)}
.tag{display:inline-flex;padding:2px 9px;border-radius:20px;font-size:10px;font-weight:700}
.t0{background:var(--b50);color:var(--b700)}.t1{background:#F5F3FF;color:#5B21B6}
.t2{background:#ECFDF5;color:#065F46}.t3{background:#FFFBEB;color:#92400E}
.t4{background:#FEF2F2;color:#991B1B}.t5{background:#FFF7ED;color:#9A3412}.t6{background:#F0FDF4;color:#166534}
.nom{font-family:var(--mo);font-size:11px;color:var(--grn);font-weight:700}
.num{font-family:var(--mo);font-size:11px;color:var(--tx2)}

@keyframes fadeUp{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:translateY(0)}}
.fa{animation:fadeUp .4s ease both}
.d1{animation-delay:.04s}.d2{animation-delay:.08s}.d3{animation-delay:.12s}
.d4{animation-delay:.16s}.d5{animation-delay:.20s}.d6{animation-delay:.24s}
.d7{animation-delay:.28s}.d8{animation-delay:.32s}
::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--bdr2);border-radius:4px}
::-webkit-scrollbar-thumb:hover{background:var(--b300)}
</style>
</head>

<body>
<div class="orb orb1"></div>
<div class="orb orb2"></div>
<div class="orb orb3"></div>

<!-- TOPBAR -->
<div class="topbar">
  <div class="logo-name">
    <div class="bot">AutoIN</div>
    <div class="sub">Sheira Devina Kevin</div>
  </div>
  <div class="topbar-right">
    <div class="badge-live"><div class="live-dot"></div>Selesai</div>
    <span style="font-size:11px;color:var(--mut);font-family:var(--mo)" id="fTm"></span>
  </div>
</div>

<div class="wrap">
  <!-- PAGE HEADER -->
  <div class="page-hdr fa d1">
    <div class="page-hdr-l">
      <h1>SIPLAH DATA FLOW <span>AUTOMATION</span></h1>
      <p>Empowering data-driven E-Commerce Strategies</p>
    </div>
  </div>

  <!-- FILE BANNER -->
  <div class="fbanner fa d1">
    <div class="fbanner-l">
      <div class="fbanner-icon">📁</div>
      <div>
        <div class="fbanner-name">FILE_NAME</div>
        <div class="fbanner-meta">FILE_ROWS baris · Diupload <span id="fTm2"></span></div>
      </div>
      <span class="ai-pill">✨ AI Predicted</span>
    </div>
    <div class="fbanner-r">
      <div id="fTm3"></div> <!-- ⬅️ pindahin ke dalam -->
        <button 
          onclick="window.top.location.reload()"
          style="
            background:#f3f4f6;
            border:1px solid #d1d5db;
            border-radius:8px;
            padding:6px 10px;
            font-size:11px;
            cursor:pointer;">
          ✕ Ganti File
        </button>
      </div>
  </div>

<!-- ======================================== BODY ======================================== -->
<!-- ======================================== ELEMEN HTML ======================================== -->
  
  <!-- KPI CARDS -->
  <div class="sl fa d1">Indikator Utama</div>
  <div class="ig" id="insightCards"></div>

  <!-- CHART + TOP 10 -->
  <!-- CHART + FILTER PANEL -->
  <div class="sl fa d6">Dashboard Interaktif</div>
  <div style="display:grid;grid-template-columns:1fr 260px;gap:16px;margin-bottom:16px">

   <!-- CHART PANEL -->
    <div class="panel fa d6" style="height:480px;display:flex;flex-direction:column;">
      <div class="ph3">
        <div>
          <div class="pti">Visualisasi Top Kategori</div>
          <div class="psu" id="chSub">—</div>
        </div>
      </div>
      <div id="cBarsH" style="flex:1;display:flex;flex-direction:column;gap:7px;overflow-y:auto;padding-right:4px"></div>
    </div>
    <div class="filter-panel fa d7" style="position:sticky;top:80px;height:480px;overflow-y:auto;display:flex;flex-direction:column;justify-content:space-between;">
      <div class="fp-hdr">
        <span style="display:flex;align-items:center">Filter Visualisasi</span>
      </div>

      <div class="fp-lbl">Kelompokkan Berdasarkan</div>
      <div class="fp-chips">
        <div class="fpc on" data-cg="level_1" onclick="clickCg(this)">Level 1</div>
        <div class="fpc"    data-cg="level_2" onclick="clickCg(this)">Level 2</div>
        <div class="fpc"    data-cg="level_3" onclick="clickCg(this)">Level 3</div>
        <div class="fpc"    data-cg="level_4" onclick="clickCg(this)">Level 4</div>
        <div class="fpc"    data-cg="level_5" onclick="clickCg(this)">Level 5</div>
        <div class="fpc"    data-cg="level_6" onclick="clickCg(this)">Level 6</div>
        <div class="fpc"    data-cg="kawasan" onclick="clickCg(this)">kawasan</div>
        <div class="fpc"    data-cg="produk"  onclick="clickCg(this)">nama item</div>
      </div>

      <div class="fp-div"></div>

      <div class="fp-lbl">Metrik</div>
      <div class="fp-chips">
        <div class="fpc on" data-cm="pend" onclick="clickCm(this)">Pendapatan</div>
        <div class="fpc"    data-cm="view" onclick="clickCm(this)">Dilihat</div>
        <div class="fpc"    data-cm="beli" onclick="clickCm(this)">Dibeli</div>
        <div class="fpc"    data-cm="cart" onclick="clickCm(this)">Keranjang</div>
      </div>

      <div class="fp-div"></div>

      <button class="fp-apply" onclick="exportTopKategori()"> ↓ Unduh Ringkasan</button>
    </div>
  </div> 

  <!-- DATA GRID -->
  <div class="sl fa d7">Tabel Data</div>
  <div class="grid-layout">
   <div class="panel fa d8" style="padding:20px">
      <div style="display:flex;flex-direction:column;gap:10px;margin-bottom:14px">
                <div class="sbar" style="width:100%">
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="var(--dim)" stroke-width="2"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg>
                  <input type="text" id="srch" placeholder="Cari nama produk..." oninput="pg=1;renderTable()" style="width:100%">
                </div>
              </div>
      <div style="overflow-x:auto; overflow-y:auto; height:600px; border-radius:8px;">
        <table class="dtbl"><thead id="tHd"></thead><tbody id="tBd"></tbody></table>
      </div>
      <div style="display:flex;align-items:center;justify-content:space-between;margin-top:14px;padding-top:14px;border-top:1px solid var(--bdr)">
        <div style="font-size:12px;color:var(--mut)">
          Menampilkan <b style="color:var(--tx)" id="rSh">0</b> dari <b style="color:var(--tx)" id="rTo">0</b> baris
        </div>
        <div style="display:flex;gap:6px;align-items:center">
          <button class="btn btn-out" style="padding:5px 12px;font-size:12px" onclick="prevPg()">‹ Prev</button>
          <span id="pgI" style="font-size:12px;color:var(--mut);min-width:50px;text-align:center"></span>
          <button class="btn btn-prim" style="padding:5px 12px;font-size:12px" onclick="nextPg()">Next ›</button>
        </div>
      </div>
    </div>
    <div class="filter-panel fa d7">
      <div class="fp-hdr">
        <span style="display:flex;align-items:center">Filter<span class="af-badge" id="afB">0</span></span>
      </div>
      <div class="fp-lbl">Kawasan</div>
      <select class="fp-sel" id="kawSel" onchange="applyFilter()"><option value="">Semua Kawasan</option></select>
      <div class="fp-div"></div>
     
      <div class="fp-lbl">Level 1</div>
      <div class="fp-chips" id="lv1C"></div>
      <div class="fp-div"></div>
      <div class="fp-lbl">Level 2</div>
      <select class="fp-sel" id="lv2S" onchange="onLv2Ch()"><option value="">Semua Level 2</option></select>
      <div class="fp-div"></div>
      <div class="fp-lbl">Level 3</div>
      <select class="fp-sel" id="lv3S" onchange="onLv3Ch()"><option value="">Semua Level 3</option></select>
      <div class="fp-div"></div>
      <div class="fp-lbl">Level 4</div>
      <select class="fp-sel" id="lv4S" onchange="onLv4Ch()"><option value="">Semua Level 4</option></select>
      <div class="fp-div"></div>
      <div class="fp-lbl">Level 5</div>
      <select class="fp-sel" id="lv5S" onchange="onLv5Ch()"><option value="">Semua Level 5</option></select>
      <div class="fp-div"></div>
      <div class="fp-lbl">Level 6</div>
      <select class="fp-sel" id="lv6S" onchange="applyFilter()"><option value="">Semua Level 6</option></select>
      <div class="fp-div"></div>
      <div class="fp-lbl">Tampilkan Kolom</div>
      <div class="fp-chips" id="colC"></div>
      <div class="fp-div"></div>
      <button class="fp-apply" onclick="exportHasilPrediksi()">↓ Unduh Hasil Prediksi</button>
    </div>
  </div>
</div>

<!-- ======================================== BODY ======================================== -->
<!-- ======================================== SCRIPT(JS) ======================================== -->
<script>
const RAW_DATA = __JS_DATA__;
const C = {prod: 'nama item', lv: ['level_1','level_2','level_3','level_4','level_5','level_6'], 
kaw: 'kawasan', view: 'item dilihat', cart: 'item ditambahkan ke keranjang', buy: 'item yang dibeli', pend: 'pendapatan item'};
const ALL_COLS=[C.prod,C.kaw,...C.lv,C.view,C.cart,C.buy,C.pend];
const PAL=['#1A6BF0','#7C3AED','#059669','#D97706','#DC2626','#0891B2','#65A30D','#DB2777'];
const TAGS=['t0','t1','t2','t3','t4','t5','t6'];
const HCOLS=['#1A6BF0','#7C3AED','#059669','#D97706','#DC2626','#0891B2','#65A30D','#DB2777','#6B7280','#9333EA','#0D9488','#B45309','#7C2D12','#1D4ED8','#15803D','#9D174D','#92400E','#1E3A8A','#065F46','#6B21A8'];

let RAW=[...RAW_DATA],filtered=[...RAW],pg=1;
const PGS=15;
let hiddenCols=new Set(),fKaw='',fLv1='',fLv2='',fLv3='',fLv4='',fLv5='',fLv6='',metric='pend',chartGroup='level_1';

window.onload=()=>{
  const now=new Date();
  const ts=now.toLocaleTimeString('id-ID');
  const dt=now.toLocaleDateString('id-ID',{day:'2-digit',month:'short',year:'numeric'});
  ['fTm','fTm2'].forEach(id=>{const e=document.getElementById(id);if(e)e.textContent=ts});
  const e3=document.getElementById('fTm3');if(e3)e3.textContent='Diprediksi: '+ts;
  buildInsights();buildChart();buildSidebar();renderTable();
};

function cv(r,c){return r[c]??''}
function toN(v){
  if(v===null||v===undefined||v==='')return 0;
  if(typeof v==='number')return isNaN(v)?0:v;
  let s=String(v).trim();
  const dots=(s.split('.').length-1);
  const commas=(s.split(',').length-1);
  if(dots>1)s=s.split('.').join('');
  else if(commas>1)s=s.split(',').join('');
  else s=s.split(',').join('.');
  s=s.replace(/[^0-9.-]/g,'');
  const n=parseFloat(s);
  return isNaN(n)?0:n;
}
function fRp(n){
  if(n>=1e9) return 'Rp '+(n/1e9).toFixed(2)+' Miliar';
  if(n>=1e6) return 'Rp '+(n/1e6).toFixed(1)+' Juta';
  if(n>=1e3) return 'Rp '+(n/1e3).toFixed(1)+' Ribu';
  return 'Rp '+Math.round(n).toLocaleString('id-ID');
}
function fN(n){
  if(n>=1e6) return (n/1e6).toFixed(1)+' Juta';
  if(n>=1e3) return (n/1e3).toFixed(1)+' Ribu';
  return Math.round(n).toLocaleString('id-ID');
}
function txt(id,v){const e=document.getElementById(id);if(e)e.textContent=v??'—'}
function sumBy(vc,gc,src){
  src=src||RAW;const s={};
  src.forEach(r=>{const g=cv(r,gc);if(g)s[g]=(s[g]||0)+toN(cv(r,vc))});
  return Object.entries(s).sort((a,b)=>b[1]-a[1]);
}
function topFreq(col,src){
  src=src||RAW;const f={};
  src.forEach(r=>{const v=cv(r,col);if(v)f[v]=(f[v]||0)+1});
  return Object.entries(f).sort((a,b)=>b[1]-a[1]);
}
function metCol(){
  return metric==='view'?C.view:
         metric==='beli'?C.buy:
         metric==='cart'?C.cart:
         C.pend
}
function getSortColumn(visH){
  const hasView = visH.includes(C.view);
  const hasCart = visH.includes(C.cart);
  const hasBuy  = visH.includes(C.buy);
  const hasPend = visH.includes(C.pend);
  if(hasPend) return C.pend;
  if(hasBuy) return C.buy;

  if(hasView && hasCart) return null;

  if(hasCart) return C.cart;
  if(hasView) return C.view;

  return null;
}
function buildInsights(){
  function fFull(n){
    return (n || 0).toLocaleString('id-ID');
  }

  function fRpFull(n){
    return 'Rp ' + (n || 0).toLocaleString('id-ID');
  }

  var kf = topFreq(C.kaw);

  var tv = RAW.reduce(function(s,r){
    return s + toN(cv(r,C.view));
  },0);

  var tp = RAW.reduce(function(s,r){
    return s + toN(cv(r,C.pend));
  },0);

  var tb = RAW.reduce(function(s,r){
    return s + toN(cv(r,C.buy));
  },0);

  var sumCartLv2 = sumBy(C.cart, 'level_2', RAW);
  var topLv2 = sumCartLv2[0] ? sumCartLv2[0][0] : '—';
  var topLv2Jml = sumCartLv2[0] ? sumCartLv2[0][1] : 0;

  
  document.getElementById('insightCards').innerHTML =

    // 1. Dominasi Kawasan
    '<div class="ic fa d1"><div class="ic-top"></div><div class="ic-icon">🌍</div>'+
      '<div class="ic-label">Dominasi Kawasan</div>'+
      '<div class="ic-value">'+(kf[0]?kf[0][0]:'—')+'</div>'+
      '<div class="ic-sub">'+(kf[0]?kf[0][1]+' produk':'—')+'</div></div>'+

    // 2. Total Dilihat (FIXED)
    '<div class="ic fa d2"><div class="ic-top"></div><div class="ic-icon">👁️</div>'+
      '<div class="ic-label">Total Dilihat</div>'+
      '<div class="ic-value">'+fFull(tv)+'</div>'+
      '<div class="ic-sub">produk</div></div>'+

    // 3. Pendapatan Item (FULL FORMAT)
    '<div class="ic fa d3"><div class="ic-top"></div><div class="ic-icon">💰</div>'+
      '<div class="ic-label">Pendapatan Item</div>'+
      '<div class="ic-value">'+fRpFull(tp)+'</div>'+
      '<div class="ic-sub">Total keseluruhan</div></div>'+

    // 4. Item Dibeli (FULL FORMAT)
    '<div class="ic fa d4"><div class="ic-top"></div><div class="ic-icon">🛍️</div>'+
      '<div class="ic-label">Item Dibeli</div>'+
      '<div class="ic-value">'+fFull(tb)+'</div>'+
      '<div class="ic-sub">unit terjual</div></div>'+

    // 5. Top Keranjang
    '<div class="ic fa d5"><div class="ic-top"></div><div class="ic-icon">🛒</div>'+
      '<div class="ic-label">Dimasukkan Keranjang</div>'+
      '<div class="ic-value">'+topLv2+'</div>'+
      '<div class="ic-sub">'+topLv2Jml+' kali</div></div>';
}

function clickCg(el){
  document.querySelectorAll('[data-cg]').forEach(c=>c.classList.remove('on'));
  el.classList.add('on');
  chartGroup=el.dataset.cg;
  buildChart();
}
function clickCm(el){
  document.querySelectorAll('[data-cm]').forEach(c=>c.classList.remove('on'));
  el.classList.add('on');
  metric=el.dataset.cm;
  buildChart();
}
function resetChartFilter(){
  chartGroup='level_1';metric='pend';
  document.querySelectorAll('[data-cg]').forEach((c,i)=>c.classList.toggle('on',i===0));
  document.querySelectorAll('[data-cm]').forEach((c,i)=>c.classList.toggle('on',i===0));
  buildChart();
}
function buildChart(){
  const vc=metCol();
  const lbl={pend:'Pendapatan',view:'Dilihat',beli:'Dibeli',cart:'Keranjang'}[metric];
  const gl={level_1:'Level 1',level_2:'Level 2',level_3:'Level 3',level_4:'Level 4',
    level_5:'Level 5',level_6:'Level 6',kawasan:'Kawasan',produk:'Produk'}[chartGroup]||chartGroup;
  txt('chSub','Top '+gl+' · '+lbl);
  const gcol=chartGroup==='produk'?C.prod:chartGroup;
  const sorted=sumBy(vc,gcol,RAW).slice(0,20);
  const mx=sorted[0]?.[1]||1;
  document.getElementById('cBarsH').innerHTML=!sorted.length
    ?'<div style="color:var(--mut);font-size:13px;text-align:center;padding:30px 0">Tidak ada data</div>'
    :sorted.map(([lb,v],i)=>{
      const pct=Math.round(v/mx*100);
      const disp=metric==='pend'?fRp(v):fN(v);
      const name=lb.length>28?lb.slice(0,27)+'…':lb;
      const grad=`linear-gradient(90deg,${HCOLS[i]},${HCOLS[i]}cc)`;
      return`<div class="hbar-row">
        <span class="hbar-rank">${i+1}</span>
        <span class="hbar-label" title="${lb}">${name||'(kosong)'}</span>
        <div class="hbar-track"><div class="hbar-fill" style="width:${pct}%;background:${grad}">
          ${pct>20?`<span>${disp}</span>`:''}
        </div></div>
        <span class="hbar-val">${disp}</span>
      </div>`;
    }).join('');
}
function buildFunnel(){
  const tk=filtered.reduce((s,r)=>s+toN(cv(r,C.cart)),0);
  const tb=filtered.reduce((s,r)=>s+toN(cv(r,C.buy)),0);
  const cvr=tk>0?(tb/tk*100).toFixed(1):0;
  const drop=tk>0?((1-tb/tk)*100).toFixed(1):0;
  const mx=tk||1;
  const pctK=100;
  const pctB=Math.round(tb/mx*100);

  const insightCvr=parseFloat(cvr);
  let insightColor,insightIcon,insightText;
  if(insightCvr>=50){
    insightColor='#059669';insightIcon='🔥';
    insightText='Conversion sangat baik! Lebih dari setengah item yang dimasukkan ke keranjang berhasil dibeli.';
  } else if(insightCvr>=25){
    insightColor='#D97706';insightIcon='⚡';
    insightText='Conversion cukup baik, namun masih ada ruang untuk meningkatkan konversi dari keranjang ke pembelian.';
  } else {
    insightColor='#DC2626';insightIcon='⚠️';
    insightText='Conversion rendah. Banyak item ditambahkan ke keranjang tapi tidak jadi dibeli — perlu evaluasi harga atau proses checkout.';
  }

  document.getElementById('funnelWrap').innerHTML=`
    <div style="display:flex;flex-direction:column;gap:20px;grid-column:1/-1">

      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px">
        <div style="background:linear-gradient(135deg,rgba(217,119,6,.08),rgba(217,119,6,.04));
             border:1px solid rgba(217,119,6,.25);border-radius:12px;padding:16px 20px;text-align:center">
          <div style="font-size:10px;font-weight:700;color:#D97706;letter-spacing:1.5px;
               text-transform:uppercase;margin-bottom:6px">Ditambah ke Keranjang</div>
          <div style="font-size:24px;font-weight:800;color:#92400E">${fN(tk)}</div>
          <div style="font-size:11px;color:#D97706;margin-top:3px">item</div>
        </div>
        <div style="background:linear-gradient(135deg,rgba(5,150,105,.08),rgba(5,150,105,.04));
             border:1px solid rgba(5,150,105,.25);border-radius:12px;padding:16px 20px;text-align:center">
          <div style="font-size:10px;font-weight:700;color:#059669;letter-spacing:1.5px;
               text-transform:uppercase;margin-bottom:6px">Berhasil Dibeli</div>
          <div style="font-size:24px;font-weight:800;color:#065F46">${fN(tb)}</div>
          <div style="font-size:11px;color:#059669;margin-top:3px">item</div>
        </div>
        <div style="background:linear-gradient(135deg,rgba(26,107,240,.08),rgba(26,107,240,.04));
             border:1px solid rgba(26,107,240,.25);border-radius:12px;padding:16px 20px;text-align:center">
          <div style="font-size:10px;font-weight:700;color:#1A6BF0;letter-spacing:1.5px;
               text-transform:uppercase;margin-bottom:6px">Conversion Rate</div>
          <div style="font-size:24px;font-weight:800;color:#0D3080">${cvr}%</div>
          <div style="font-size:11px;color:#1A6BF0;margin-top:3px">keranjang → dibeli</div>
        </div>
      </div>

      <div>
        <div style="display:flex;justify-content:space-between;margin-bottom:6px">
          <span style="font-size:12px;font-weight:600;color:var(--tx)">🛒 Ditambah ke Keranjang</span>
          <span style="font-family:var(--mo);font-size:12px;color:var(--mut)">${fN(tk)}</span>
        </div>
        <div class="funnel-track">
          <div class="funnel-fill" style="width:100%;background:linear-gradient(90deg,#D97706,#FCD34D)">
            ${pctK>12?'100%':''}
          </div>
        </div>
        <div style="margin-top:8px;display:flex;justify-content:space-between;margin-bottom:6px">
          <span style="font-size:12px;font-weight:600;color:var(--tx)">✅ Berhasil Dibeli</span>
          <span style="font-family:var(--mo);font-size:12px;color:var(--mut)">${fN(tb)}</span>
        </div>
        <div class="funnel-track">
          <div class="funnel-fill" style="width:${pctB}%;background:linear-gradient(90deg,#059669,#34D399)">
            ${pctB>12?pctB+'%':''}
          </div>
        </div>
        <div style="font-size:11px;color:#DC2626;text-align:right;margin-top:4px;font-family:var(--mo)">
          ▼ drop ${drop}% tidak jadi dibeli
        </div>
      </div>

      <div style="background:linear-gradient(135deg,rgba(${insightCvr>=50?'5,150,105':insightCvr>=25?'217,119,6':'220,38,38'},.06),transparent);
           border:1px solid rgba(${insightCvr>=50?'5,150,105':insightCvr>=25?'217,119,6':'220,38,38'},.2);
           border-radius:10px;padding:14px 18px;display:flex;align-items:flex-start;gap:10px">
        <span style="font-size:20px;flex-shrink:0">${insightIcon}</span>
        <div>
          <div style="font-size:12px;font-weight:700;color:${insightColor};margin-bottom:3px">
            Insight Conversion
          </div>
          <div style="font-size:12px;color:var(--tx2);line-height:1.6">${insightText}</div>
        </div>
      </div>

    </div>`;
}
function buildSidebar(){
  const kaws=[...new Set(RAW.map(r=>cv(r,C.kaw)).filter(Boolean))].sort();
  document.getElementById('kawSel').innerHTML=
    '<option value="">Semua Kawasan</option>'+kaws.map(k=>`<option value="${k}">${k}</option>`).join('');
  refreshLv1();refreshLv2(RAW);refreshLv3(RAW);refreshLv4(RAW);refreshLv5(RAW);refreshLv6(RAW);
  document.getElementById('colC').innerHTML=ALL_COLS.map(h=>
    `<div class="col-chip on" data-col="${h}" onclick="this.classList.toggle('on');applyFilter()" title="${h}">${
      h.replace('Item ditambahkan ke keranjang','Keranjang')
       .replace('Item yang dibeli','Dibeli')
       .replace('Pendapatan item','Pendapatan')
       .replace('Item dilihat','Dilihat')
    }</div>`).join('');
}
function refreshLv1(){
  const vals=[...new Set(RAW.map(r=>cv(r,'level_1')).filter(Boolean))].sort();
  document.getElementById('lv1C').innerHTML=
    '<div class="fpc'+((!fLv1)?' on':'')+'" data-v="" onclick="clickLv1(this)">Semua</div>'+
    vals.map(function(v){return'<div class="fpc'+(v===fLv1?' on':'')+'" data-v="'+v+'" onclick="clickLv1(this)">'+v+'</div>'}).join('');
}
function refreshLv2(src){
  var base=fLv1?src.filter(function(r){return cv(r,'level_1')===fLv1}):src;
  var vals=[...new Set(base.map(function(r){return cv(r,'level_2')}).filter(Boolean))].sort();
  document.getElementById('lv2S').innerHTML=
    '<option value="">Semua Level 2</option>'+vals.map(function(v){return'<option value="'+v+'"'+(v===fLv2?' selected':'')+'>'+v+'</option>'}).join('');
}
function refreshLv3(src){
  var base=fLv1?src.filter(function(r){return cv(r,'level_1')===fLv1}):src;
  if(fLv2)base=base.filter(function(r){return cv(r,'level_2')===fLv2});
  var vals=[...new Set(base.map(function(r){return cv(r,'level_3')}).filter(Boolean))].sort();
  document.getElementById('lv3S').innerHTML=
    '<option value="">Semua Level 3</option>'+vals.map(function(v){return'<option value="'+v+'"'+(v===fLv3?' selected':'')+'>'+v+'</option>'}).join('');
}
function refreshLv4(src){
  var base=fLv1?src.filter(function(r){return cv(r,'level_1')===fLv1}):src;
  if(fLv2)base=base.filter(function(r){return cv(r,'level_2')===fLv2});
  if(fLv3)base=base.filter(function(r){return cv(r,'level_3')===fLv3});
  var vals=[...new Set(base.map(function(r){return cv(r,'level_4')}).filter(Boolean))].sort();
  document.getElementById('lv4S').innerHTML=
    '<option value="">Semua Level 4</option>'+vals.map(function(v){return'<option value="'+v+'"'+(v===fLv4?' selected':'')+'>'+v+'</option>'}).join('');
}
function refreshLv5(src){
  var base=fLv1?src.filter(function(r){return cv(r,'level_1')===fLv1}):src;
  if(fLv2)base=base.filter(function(r){return cv(r,'level_2')===fLv2});
  if(fLv3)base=base.filter(function(r){return cv(r,'level_3')===fLv3});
  if(fLv4)base=base.filter(function(r){return cv(r,'level_4')===fLv4});
  var vals=[...new Set(base.map(function(r){return cv(r,'level_5')}).filter(Boolean))].sort();
  document.getElementById('lv5S').innerHTML=
    '<option value="">Semua Level 5</option>'+vals.map(function(v){return'<option value="'+v+'"'+(v===fLv5?' selected':'')+'>'+v+'</option>'}).join('');
}
function refreshLv6(src){
  var base=fLv1?src.filter(function(r){return cv(r,'level_1')===fLv1}):src;
  if(fLv2)base=base.filter(function(r){return cv(r,'level_2')===fLv2});
  if(fLv3)base=base.filter(function(r){return cv(r,'level_3')===fLv3});
  if(fLv4)base=base.filter(function(r){return cv(r,'level_4')===fLv4});
  if(fLv5)base=base.filter(function(r){return cv(r,'level_5')===fLv5});
  var vals=[...new Set(base.map(function(r){return cv(r,'level_6')}).filter(Boolean))].sort();
  document.getElementById('lv6S').innerHTML=
    '<option value="">Semua Level 6</option>'+vals.map(function(v){return'<option value="'+v+'"'+(v===fLv6?' selected':'')+'>'+v+'</option>'}).join('');
}
function clickLv1(el){
  document.querySelectorAll('#lv1C .fpc').forEach(function(c){c.classList.remove('on')});
  el.classList.add('on');fLv1=el.dataset.v;fLv2='';fLv3='';fLv4='';fLv5='';fLv6='';
  refreshLv2(RAW);refreshLv3(RAW);refreshLv4(RAW);refreshLv5(RAW);refreshLv6(RAW);applyFilter();
}
function onLv2Ch(){fLv2=document.getElementById('lv2S').value;fLv3='';fLv4='';fLv5='';fLv6='';refreshLv3(RAW);refreshLv4(RAW);refreshLv5(RAW);refreshLv6(RAW);applyFilter()}
function onLv3Ch(){fLv3=document.getElementById('lv3S').value;fLv4='';fLv5='';fLv6='';refreshLv4(RAW);refreshLv5(RAW);refreshLv6(RAW);applyFilter()}
function onLv4Ch(){fLv4=document.getElementById('lv4S').value;fLv5='';fLv6='';refreshLv5(RAW);refreshLv6(RAW);applyFilter()}
function onLv5Ch(){fLv5=document.getElementById('lv5S').value;fLv6='';refreshLv6(RAW);applyFilter()}
function applyFilter(){
  fKaw=document.getElementById('kawSel').value;
  fLv2=document.getElementById('lv2S').value;
  fLv3=document.getElementById('lv3S').value;
  fLv4=document.getElementById('lv4S').value;
  fLv5=document.getElementById('lv5S').value;
  fLv6=document.getElementById('lv6S').value;
  hiddenCols=new Set([...document.querySelectorAll('#colC .col-chip:not(.on)')].map(function(c){return c.dataset.col}));
  filtered=RAW.filter(function(row){
    if(fKaw&&cv(row,C.kaw)!==fKaw)return false;
    if(fLv1&&cv(row,'level_1')!==fLv1)return false;
    if(fLv2&&cv(row,'level_2')!==fLv2)return false;
    if(fLv3&&cv(row,'level_3')!==fLv3)return false;
    if(fLv4&&cv(row,'level_4')!==fLv4)return false;
    if(fLv5&&cv(row,'level_5')!==fLv5)return false;
    if(fLv6&&cv(row,'level_6')!==fLv6)return false;
    return true;
  });
  pg=1;
  var n=0;if(fKaw)n++;if(fLv1)n++;if(fLv2)n++;if(fLv3)n++;if(fLv4)n++;if(fLv5)n++;if(fLv6)n++;if(hiddenCols.size)n++;
  var b=document.getElementById('afB');b.textContent=n;b.classList.toggle('show',n>0);
  setTimeout(function(){renderTable();},0);
}
function resetFilter(){
  fKaw='';fLv1='';fLv2='';fLv3='';fLv4='';fLv5='';fLv6='';hiddenCols=new Set();
  document.getElementById('kawSel').value='';
  refreshLv1();refreshLv2(RAW);refreshLv3(RAW);refreshLv4(RAW);refreshLv5(RAW);refreshLv6(RAW);
  document.querySelectorAll('#colC .col-chip').forEach(function(c){c.classList.add('on')});
  filtered=[...RAW];pg=1;
  document.getElementById('afB').classList.remove('show');
  buildChart();
  setTimeout(function(){renderTable();},0);
}
/* ── Fuzzy search + transliterasi Indonesia ── */
const TRANS_MAP={
  'krts':'kertas','kts':'kertas','kerts':'kertas',
  'bku':'buku','bk':'buku',
  'pnsl':'pensil','pnsil':'pensil','pencil':'pensil',
  'kprt':'komputer','komput':'komputer','komp':'komputer',
  'lptop':'laptop','lapt':'laptop',
  'krs':'kursi','krsi':'kursi',
  'mja':'meja',
  'sptu':'sepatu','spt':'sepatu',
  'bju':'baju','bj':'baju',
  'clna':'celana','cln':'celana',
  'tas':'tas','ts':'tas',
  'hp':'handphone','handphn':'handphone',
  'alts':'alat tulis','altu':'alat tulis',
  'atk':'alat tulis kantor',
  'olrg':'olahraga','or':'olahraga',
  'kshtn':'kesehatan','kesh':'kesehatan',
  'elktrnk':'elektronik','elek':'elektronik','elektro':'elektronik',
  'furnitr':'furnitur','furn':'furnitur',
  'mknn':'makanan','mkn':'makanan',
  'mnmn':'minuman','mnm':'minuman',
  'obat':'obat','obt':'obat',
  'msk':'masker','mask':'masker',
  'srng':'sarung','srg':'sarung',
  'kain':'kain','kn':'kain',
};
function normalizeQuery(q){
  q=q.toLowerCase().trim();
  if(TRANS_MAP[q])return[q,TRANS_MAP[q]];
  const alts=Object.entries(TRANS_MAP)
    .filter(([k,v])=>k.startsWith(q)||v.startsWith(q)||k.includes(q)||v.includes(q))
    .map(([k,v])=>[k,v]).flat();
  return[q,...new Set(alts)];
}
function ngrams(s,n){
  s=' '+s.toLowerCase()+' ';
  const g=new Set();
  for(let i=0;i<=s.length-n;i++)g.add(s.slice(i,i+n));
  return g;
}
function bigramSim(a,b){
  if(!a||!b)return 0;
  const ag=ngrams(a,2),bg=ngrams(b,2);
  let h=0;ag.forEach(g=>{if(bg.has(g))h++});
  return h/Math.max(ag.size,1);
}
function fuzzyMatch(query, target){
  if(!query) return true;
  target = target.toLowerCase();
  query  = query.toLowerCase().trim();
  const queries = normalizeQuery(query);
  for(const q of queries){
    if(q && target.includes(q)) return true;
  }
  if(query.length >= 3){
    return false;
  }
  for(const q of queries){
    if(!q) continue;
    if(q.length >= 2 && bigramSim(q, target) >= 0.45){
      return true;
    }
    const words = target.split(/\\s+/);
    for(const w of words){
      if(q.length >= 3 && bigramSim(q, w) >= 0.55){
        return true;
      }
    }
  }
  return false;
}
function fuzzyFilter(rows, query){
  if(!query) return rows;
  return rows.filter(r =>
    fuzzyMatch(query, String(r[C.prod] || ''))
  );
}

function renderTable(){
  const srch=(document.getElementById('srch')?.value||'').toLowerCase().trim();
  const visH=ALL_COLS.filter(h=>!hiddenCols.has(h));
  const lv1map={};
  [...new Set(RAW.map(r=>cv(r,'level_1')).filter(Boolean))].forEach((v,i)=>lv1map[v]=i%TAGS.length);
  let display=fuzzyFilter(filtered,srch);
  const sortCol = getSortColumn(visH);
  if(sortCol){
  display = [...display].sort((a,b)=>toN(b[sortCol]) - toN(a[sortCol]));
  }
  const total=display.length;
  const maxPg=Math.max(1,Math.ceil(total/PGS));
  if(pg>maxPg)pg=maxPg;
  const page=display.slice((pg-1)*PGS,pg*PGS);
  document.getElementById('tHd').innerHTML=`<tr>${visH.map(h=>`<th>${h}</th>`).join('')}</tr>`;
  document.getElementById('tBd').innerHTML=!page.length
    ?`<tr><td colspan="${visH.length||1}" style="text-align:center;color:var(--mut);padding:40px">Tidak ada data</td></tr>`
    :page.map(row=>`<tr>${visH.map(h=>{
        const v=cv(row,h);
        if(h==='level_1'&&v)return`<td><span class="tag ${TAGS[lv1map[v]??0]}">${v}</span></td>`;
        if(h.startsWith('level_')&&v)return`<td><span style="font-size:11px;color:var(--mut)">${v}</span></td>`;
        if(h===C.kaw&&v)return`<td><span style="font-size:12px;font-weight:600;color:var(--acc)">${v}</span></td>`;
        if(h===C.pend&&v){const n=toN(v);if(n>0)return`<td class="nom">${fRp(n)}</td>`}
        if([C.view,C.cart,C.buy].includes(h)&&v)return`<td class="num">${toN(v).toLocaleString('id-ID')}</td>`;
        if(h===C.prod)return`<td style="font-weight:600;color:var(--tx)" title="${v}">${v}</td>`;
        return`<td>${v}</td>`;
      }).join('')}</tr>`).join('');
  txt('rSh',page.length);txt('rTo',total);txt('pgI',`${pg} / ${maxPg}`);
  txt('gridSub',total.toLocaleString('id-ID')+' baris'+(srch?' (pencarian)':''));
  let title='Semua Data';
  if(fLv3)title=fLv3;else if(fLv2)title=fLv2;else if(fLv1)title=fLv1;else if(fKaw)title='Kawasan: '+fKaw;
  txt('gridTitle',title);
}
function prevPg(){if(pg>1){pg--;renderTable()}}
function nextPg(){
  const s=(document.getElementById('srch')?.value||'').toLowerCase().trim();
  const d=fuzzyFilter(filtered,s);
  if(pg<Math.ceil(d.length/PGS)){pg++;renderTable()}
}
function exportCSV(){
  const s=(document.getElementById('srch')?.value||'').toLowerCase().trim();
  const data=fuzzyFilter(filtered,s);
  const now = new Date();
  const tgl = now.toLocaleDateString('id-ID',{day:'2-digit',month:'2-digit',year:'numeric'}).replace(/\\//g,'-');
  const cols=ALL_COLS.filter(h=>!hiddenCols.has(h));
  const csv=[cols.join(','),...data.map(r=>cols.map(h=>`"${cv(r,h)}"`).join(','))].join('\\n');
  const a=Object.assign(document.createElement('a'),{
    href:URL.createObjectURL(new Blob([csv],{type:'text/csv'})),
    download:`Hasil_prediksi_(${tgl}).csv`
  });a.click();
}
function exportTopKategori(){
  var vc=metCol();
  var gcol=chartGroup==='produk'?C.prod:chartGroup;
  var sorted=sumBy(vc,gcol,filtered);
  var lbl=metric==='pend'?'Pendapatan':metric==='view'?'Dilihat':metric==='cart'?'Keranjang':'Dibeli';
  var gl=chartGroup==='level_1'?'Level 1':
         chartGroup==='level_2'?'Level 2':
         chartGroup==='level_3'?'Level 3':
         chartGroup==='level_4'?'Level 4':
         chartGroup==='level_5'?'Level 5':
         chartGroup==='level_6'?'Level 6':
         chartGroup==='Kawasan'?'Kawasan':'Nama Produk';
  var now=new Date();
  var tgl=now.toLocaleDateString('id-ID',{day:'2-digit',month:'2-digit',year:'numeric'}).replace(/\//g,'-');
  var rows=['No,' + gl + ',' + lbl];
  for(var i=0;i<sorted.length;i++){
    rows.push((i+1) + ',"' + sorted[i][0] + '",' + sorted[i][1]);
  }
  var csv=rows.join('\\n');
  var blob=new Blob([csv],{type:'text/csv'});
  var url=URL.createObjectURL(blob);
  var a=document.createElement('a');
  a.href=url;
  a.download='Ringkasan Data_' + gl + '_' + lbl + '_(' + tgl + ').csv';
  a.click();
}
function exportHasilPrediksi(){
  var now=new Date();
  var tgl=now.toLocaleDateString('id-ID',{day:'2-digit',month:'2-digit',year:'numeric'}).replace(/\//g,'-');
  var cols=ALL_COLS.filter(function(h){return!hiddenCols.has(h)});
  var srch=(document.getElementById('srch')?.value||'').toLowerCase().trim();
  var data=fuzzyFilter(filtered,srch);
  var sortCol = getSortColumn(cols);
  if(sortCol){
  data = [...data].sort((a,b)=>toN(b[sortCol]) - toN(a[sortCol]));
  }
  var rows=[cols.join(',')];
  for(var i=0;i<data.length;i++){
      var r=data[i];
      rows.push(cols.map(function(h){return '"'+cv(r,h)+'"'}).join(','));
    }
  var csv=rows.join('\\n');
  var blob=new Blob([csv],{type:'text/csv'});
  var url=URL.createObjectURL(blob);
  var a=document.createElement('a');
  a.href=url;
  a.download='Hasil_Prediksi_('+tgl+').csv';
  a.click();
}
</script>
</body>
</html>

"""

def main():
    # ===============================
    # INIT SESSION STATE (SAFE)
    # ===============================
    if "result_df" not in st.session_state:
        st.session_state.result_df = None
    if "result_fname" not in st.session_state:
        st.session_state.result_fname = ""

    # ===============================
    # HANDLE QUERY PARAMS (RESET)
    # ===============================
    query_params = st.query_params
    if "reset" in query_params:
        st.session_state.result_df = None
        st.session_state.result_fname = ""
        st.query_params.clear()
        st.rerun()

    # ===============================
    # CASE 1: SUDAH ADA HASIL → DASHBOARD
    # ===============================
    if st.session_state.result_df is not None:
        df = st.session_state.result_df
        fname = st.session_state.result_fname

        dashboard_html = build_dashboard_html(df, fname)

        # ngatur tingginya tampilan
        table_section = 500 + 80     # tabel + pagination
        fixed_section = 1300  # header + KPI + charts
        dynamic_height = fixed_section + table_section

        components.html(
            dashboard_html,
            height=dynamic_height,
            scrolling=False
        )

        return  # ⛔ penting: stop di sini

    # ===============================
    # CASE 2: BELUM ADA HASIL → UPLOAD SCREEN
    # ===============================
    result = render_upload_screen()

    if result is not None:
        df_result, fname = result

        # simpan ke state
        st.session_state.result_df = df_result
        st.session_state.result_fname = fname

        # rerun untuk pindah ke dashboard
        st.rerun()

if __name__ == "__main__":
    main()