"""
Identifikasi Gambar - Ini Kacamata??
Aplikasi Streamlit: deteksi kacamata / tidak kacamata dari foto (unggah / kamera)
Mendukung model Keras (.h5) dan PyTorch (.pt, termasuk YOLO hasil deteksi objek).
"""
import base64
import os

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

# ============================================================
# KONFIGURASI  -> sesuaikan di bagian ini saja
# ============================================================
MODEL_H5_PATH = "model.h5"          # path file model Keras
MODEL_PT_PATH = "best.pt"          # path file model PyTorch / YOLO
CLASS_NAMES = ["kacamata", "tidak kacamata"]   # urutan label = urutan output model
DEFAULT_IMG_SIZE = 224              # ukuran input (dipakai untuk .pt non-YOLO; .h5 dibaca otomatis)
HOME_URL = "https://halaman-beranda.vercel.app/"   # link halaman index.html Anda
CREDIT = "Projek ini dibuat oleh Dony Rizky Hidayat 2308108010002"
IMG_BACK = "karakter-belakang.png"  # opsional (sama seperti di index.html)
IMG_FRONT = "karakter-depan.png"    # opsional
IMG_MID = "karakter-tengah.png"    # opsional (gambar tengah, di antara belakang dan depan)

LABEL_INFO = {
    "kacamata": ("Kacamata", "👓", "Orang pada foto terdeteksi memakai kacamata."),
    "tidak kacamata": ("Tidak kacamata", "🙂", "Orang pada foto terdeteksi tidak memakai kacamata."),
}
NORM_OPTIONS = [
    "0 – 1 (dibagi 255)",
    "Tanpa normalisasi (0 – 255)",
    "-1 – 1 (MobileNet / Xception)",
    "ImageNet mean/std (umum di PyTorch)",
]

st.set_page_config(page_title="Ini Kacamata?", page_icon="👓", layout="wide",
                   initial_sidebar_state="collapsed")


# ============================================================
# TAMPILAN (meniru index.html: tema lukisan langit, coklat)
# ============================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Chau+Philomene+One&family=Kaushan+Script&display=swap');
:root{ --ink:#4a3a22; --dark:#2a2012; --card:rgba(255,255,255,.38); --script:'Kaushan Script',cursive; }

html, body, [class*="css"], .stApp, .stMarkdown, p, label, span, button, input{
  font-family:'Chau Philomene One', system-ui, sans-serif !important; color:var(--ink);
}
[data-testid="stExpander"] summary [data-testid="stIconMaterial"]{
  font-family:'Material Symbols Rounded','Material Icons' !important;
}
.stApp{
  background:
    radial-gradient(ellipse 40% 22% at 78% 14%, rgba(222,150,80,.65), transparent 70%),
    radial-gradient(ellipse 30% 18% at 14% 20%, rgba(190,140,95,.45), transparent 70%),
    radial-gradient(ellipse 35% 20% at 30% 62%, rgba(214,160,100,.45), transparent 70%),
    radial-gradient(ellipse 30% 18% at 70% 75%, rgba(225,175,110,.45), transparent 70%),
    linear-gradient(180deg,#e2c28f 0%,#ecd0a0 45%,#e3bf88 100%);
  background-attachment:fixed;
}
[data-testid="stHeader"], [data-testid="stToolbar"], #MainMenu, footer{display:none !important}
.block-container{max-width:1250px; padding:0 4vw 110px !important}

/* ---------- loading ---------- */
#splash{position:fixed; inset:0; z-index:99999; background:#ecd0a0; display:flex; flex-direction:column;
  align-items:center; justify-content:center; gap:22px;
  animation:splashOut .9s cubic-bezier(.77,0,.18,1) 2.5s forwards}
@keyframes splashOut{to{transform:translateY(-100%); visibility:hidden}}
#splash svg{width:150px; overflow:visible}
#splash path,#splash circle{fill:none; stroke:#4a3a22; stroke-width:5; stroke-linecap:round;
  stroke-dasharray:400; stroke-dashoffset:400; animation:draw 1.6s ease forwards}
#splash .r{animation-delay:.2s} #splash .b{animation-delay:.9s}
@keyframes draw{to{stroke-dashoffset:0}}
#splash .lt{font-size:18px}
#splash .bar{width:220px; height:6px; background:rgba(74,58,34,.18); border-radius:99px; overflow:hidden}
#splash .bar i{display:block; height:100%; width:0; background:#4a3a22; border-radius:99px;
  animation:fill 2.2s ease forwards}
@keyframes fill{to{width:100%}}

/* ---------- nav ---------- */
.nav{display:flex; justify-content:flex-end; align-items:center; gap:56px; padding:44px 0 0;
  font-size:24px; letter-spacing:.5px}
.nav a{color:#111 !important; text-decoration:none !important; text-transform:uppercase; position:relative}
.nav a::after{content:""; position:absolute; left:0; bottom:-5px; height:2px; width:0; background:var(--ink); transition:width .25s}
.nav a:hover::after{width:100%}

/* ---------- hero ---------- */
.hero-wrap{position:relative; min-height:62vh; padding-top:3vh}
.hero h1{font-weight:400; text-transform:uppercase; line-height:1; margin:0; padding:.08em 0;
  font-size:clamp(34px,5.2vw,86px); color:var(--ink)}
.hero .sub{font-family:var(--script) !important; font-size:clamp(44px,6.6vw,110px); line-height:1.1;
  transform:rotate(-3deg); transform-origin:left; margin:.1em 0 0 2vw; color:var(--ink)}
.hero .desc{font-size:clamp(17px,1.7vw,28px); line-height:1.4; margin-top:4vh; color:#3b2e1a; max-width:760px}
.hero h1,.hero .sub,.hero .desc{opacity:0; animation:rise .9s cubic-bezier(.2,.8,.2,1) 2.7s forwards}
.hero .sub{animation-delay:2.85s} .hero .desc{animation-delay:3s}
@keyframes rise{from{opacity:0; transform:translateY(40px)} to{opacity:1}}
.hero .sub{ } /* rotate dipertahankan lewat keyframe di bawah */
.hero .sub{animation-name:riseRot}
@keyframes riseRot{from{opacity:0; transform:translateY(40px) rotate(-3deg)} to{opacity:1; transform:rotate(-3deg)}}

/* ---------- karakter: posisi & ukuran disamakan dengan index.html ---------- */
[data-testid="stMain"], .stApp{overflow-x:hidden}
.hero-wrap{z-index:3}
.art{position:relative; z-index:1; display:block; pointer-events:none;
  width:min(48vw,880px); height:max(340px, calc(100vh - 140px)); max-height:100vh;
  margin-left:calc(100% - min(48vw,880px)); left:calc(max(0px, (100vw - 1250px)/2) + 6vw);
  opacity:0; animation:artIn 1s cubic-bezier(.2,.8,.2,1) 2.9s forwards}
@keyframes artIn{from{opacity:0; transform:translateX(60px)} to{opacity:1; transform:none}}
.art img{position:absolute; width:auto !important; max-width:none !important; max-height:none !important;
  object-fit:contain; object-position:bottom right; border:0 !important; border-radius:0 !important; box-shadow:none !important}
.art .back{right:0; top:-8%; height:80% !important; opacity:.55; z-index:1}
.art .mid{right:16%; top:-10%; height:110% !important; opacity:.5; z-index:2}
.art .front{right:25px; bottom:0; height:82% !important; z-index:3}
.art svg{position:absolute; right:0; bottom:10%; width:100%; max-width:480px}
@media (max-width:640px){
  .art{width:90%; height:auto; max-height:none; aspect-ratio:4/5; margin:20px auto 0; left:0}
  .art .back{height:70% !important; right:-15%; top:-20%}
  .art .mid{height:800px !important; width:220% !important; right:45%; top:-90%}
  .art .front{height:84% !important; right:15%; top:-5%}
}

/* ---------- section ---------- */
.sec-title{font-size:clamp(30px,4vw,48px); line-height:1.15; margin:70px 0 8px}
.lead{font-size:18px; max-width:640px; line-height:1.5}
.steps{display:grid; grid-template-columns:repeat(3,1fr); gap:22px; margin-top:28px}
.step{border-left:4px solid var(--ink); padding:4px 0 4px 18px}
.step b{display:block; font-size:24px; margin-bottom:2px}
.step h3{font-size:20px; margin:0 0 6px; font-weight:400}
.step p{line-height:1.5; font-size:16px; margin:0}
@media (max-width:820px){.steps{grid-template-columns:1fr} .nav{justify-content:center; gap:26px; font-size:18px; padding-top:24px}
  .hero .sub{margin-left:0; transform:none}}

/* ---------- widget Streamlit ---------- */
.stTabs [data-baseweb="tab-list"]{gap:10px; border-bottom:2px solid rgba(74,58,34,.35)}
[data-testid="stFileUploaderDropzone"]{background:var(--card); border:none; border-radius:none; padding:28px}
.stTabs [data-baseweb="tab"] p{font-size:20px !important}
.stTabs [data-baseweb="tab-highlight"]{background:var(--ink) !important; height:3px}
[data-testid="stFileUploaderDropzone"]{background:var(--card); border:2px dashed var(--ink); border-radius:18px; padding:28px}
[data-testid="stFileUploaderDropzone"] small, [data-testid="stFileUploaderDropzone"] span{color:var(--ink)}
.stButton>button, [data-testid="stFileUploaderDropzone"] button, [data-testid="stCameraInputButton"],
[data-testid="stBaseButton-secondary"]{
  background:var(--ink) !important; color:#fff !important; border:0 !important; border-radius:99px !important;
  padding:.55rem 1.6rem !important; box-shadow:0 6px 0 var(--dark); transition:transform .15s, box-shadow .15s}
.stButton>button:hover, [data-testid="stFileUploaderDropzone"] button:hover, [data-testid="stCameraInputButton"]:hover{
  transform:translateY(3px); box-shadow:0 3px 0 var(--dark)}
.stButton>button p, [data-testid="stFileUploaderDropzone"] button p, [data-testid="stCameraInputButton"] p{color:#fff !important}
[data-testid="stExpander"]{background:var(--card); border:none !important; border-radius:16px}
[data-testid="stImage"] img{border-radius:18px; border:3px solid var(--ink)}
div[data-baseweb="select"]>div, .stTextInput input, .stNumberInput input{background:rgba(255,255,255,.55) !important}
[data-testid="stExpander"]{background:var(--card); border:2px solid var(--ink); border-radius:16px}
[data-testid="stExpander"] summary p{font-size:18px}
.stRadio label p{font-size:17px}

/* ---------- hasil ---------- */
.result{background:var(--card); border:3px solid var(--ink); border-radius:22px; padding:26px 28px}
.result.ok{background:#f3dfb7}
.r-cap{font-size:16px; opacity:.8}
.r-main{font-size:clamp(34px,4.4vw,60px); line-height:1.1; margin:6px 0 4px}
.r-main.script{font-family:var(--script) !important}
.r-desc{font-size:17px; margin-bottom:16px}
.pbar{margin:10px 0}
.pbar .row{display:flex; justify-content:space-between; font-size:16px; margin-bottom:4px}
.placeholder{border:none; border-radius:22px; padding:44px 20px; text-align:center; font-size:18px; background:var(--card)}
.pbar .track i{display:block; height:100%; background:var(--ink); border-radius:99px}
.pbar.dim .track i{background:rgba(74,58,34,.45)}
.placeholder{border:2px dashed var(--ink); border-radius:22px; padding:44px 20px; text-align:center; font-size:18px; background:var(--card)}

/* ---------- ticker ---------- */
.ticker{position:fixed; left:0; right:0; bottom:0; z-index:999; overflow:hidden; white-space:nowrap;
  background:rgba(236,208,160,.94); padding:12px 0; border-top:2px solid var(--ink)}
.ticker .track{display:inline-block; animation:flow 22s linear infinite; font-size:18px}
.ticker .track span{padding-right:70px}
.ticker:hover .track{animation-play-state:paused}
/* hapus garis tepi tab, kotak unggah, placeholder */
.stTabs [data-baseweb="tab-list"]{border-bottom:none !important; box-shadow:none !important}
.stTabs [data-baseweb="tab-border"]{display:none !important}
[data-testid="stFileUploaderDropzone"]{border:none !important; outline:none !important}
[data-testid="stFileUploader"] section{border:none !important; outline:none !important}
.placeholder{border:none !important}
@keyframes flow{from{transform:translateX(0)} to{transform:translateX(-50%)}}
@media (prefers-reduced-motion:reduce){*{animation-duration:.01s !important; animation-delay:0s !important}
  .ticker .track{animation:none}}
</style>
"""

SPLASH = """
<div id="splash" aria-label="Memuat halaman">
  <svg viewBox="0 0 220 90" aria-hidden="true">
    <circle class="r" cx="55" cy="50" r="34"/><circle class="r" cx="165" cy="50" r="34"/>
    <path class="b" d="M89 46 Q110 28 131 46"/><path class="b" d="M21 46 L4 30"/><path class="b" d="M199 46 L216 30"/>
  </svg>
  <div class="lt">Memuat...</div><div class="bar"><i></i></div>
</div>
"""

FALLBACK_SVG = """
<svg viewBox="0 0 220 90" fill="none" stroke="#4a3a22" stroke-width="5" stroke-linecap="round" aria-hidden="true">
  <circle cx="55" cy="50" r="34"/><circle cx="165" cy="50" r="34"/>
  <path d="M89 46 Q110 28 131 46"/><path d="M21 46 L4 30"/><path d="M199 46 L216 30"/>
</svg>
"""


def img_tag(path, cls, alt=""):
    """Sisipkan gambar lokal (png) sebagai base64 supaya tampil di HTML Streamlit."""
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f'<img class="{cls}" src="data:image/png;base64,{b64}" alt="{alt}">'


def render_header():
    st.markdown(CSS, unsafe_allow_html=True)
    if "splash_done" not in st.session_state:       # animasi loading hanya sekali
        st.session_state.splash_done = True
        st.markdown(SPLASH, unsafe_allow_html=True)

    st.markdown(
        f'<div class="nav"><a href="{HOME_URL}" target="_blank">Beranda</a>'
        f'<a href="#coba">Coba</a><a href="#cara">Cara kerja</a></div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.35, 1])
    with left:
        st.markdown(
            """<div class="hero-wrap hero">
              <h1>identifikasi gambar</h1>
              <div class="sub">ini kacamata ??</div>
              <p class="desc">Unggah gambar atau ambil foto lewat kamera, lalu biarkan kecerdasan buatan
              mengidentifikasi apakah seseorang menggunakan kacamata atau tidak. Hasil muncul otomatis,
              cepat, dan praktis.</p></div>""",
            unsafe_allow_html=True,
        )
    with right:
        back, mid = img_tag(IMG_BACK, "back"), img_tag(IMG_MID, "mid")
        front = img_tag(IMG_FRONT, "front", "Ilustrasi karakter berkacamata")
        st.markdown(f'<div class="art">{back}{mid}{front}{"" if (back or mid or front) else FALLBACK_SVG}</div>',
                    unsafe_allow_html=True)


def render_steps():
    st.markdown(
        """<div id="cara"></div><h2 class="sec-title">Cara kerjanya</h2>
        <p class="lead">Tiga langkah singkat dari gambar sampai hasil.</p>
        <div class="steps">
          <div class="step"><b>1</b><h3>Unggah atau foto</h3><p>Pilih foto wajah dari galeri, atau ambil foto langsung dengan kamera.</p></div>
          <div class="step"><b>2</b><h3>Model menganalisis</h3><p>Model mencari ciri kacamata pada gambar.</p></div>
          <div class="step"><b>3</b><h3>Lihat hasilnya</h3><p>Aplikasi menampilkan apakah orang pada foto memakai kacamata, lengkap dengan tingkat keyakinan.</p></div>
        </div>""",
        unsafe_allow_html=True,
    )


def render_ticker():
    spans = "".join(f"<span>{CREDIT}</span>" for _ in range(4))
    st.markdown(f'<div class="ticker"><div class="track">{spans}</div></div>', unsafe_allow_html=True)


# ============================================================
# MODEL
# ============================================================
@st.cache_resource(show_spinner="Memuat model...")
def load_keras(path):
    import tensorflow as tf
    return tf.keras.models.load_model(path, compile=False)


@st.cache_resource(show_spinner="Memuat model...")
def load_torch(path):
    """Mengembalikan (model, jenis). Jenis: 'yolo' atau 'torch'."""
    import torch
    # 1) YOLO (deteksi objek / klasifikasi) - hanya jika paket ultralytics terpasang
    try:
        from ultralytics import YOLO
        return YOLO(path), "yolo"
    except Exception:
        pass
    # 2) TorchScript (torch.jit.save)
    try:
        m = torch.jit.load(path, map_location="cpu")
        m.eval()
        return m, "torch"
    except Exception:
        pass
    # 3) Model utuh (torch.save(model, ...))
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        raise RuntimeError(
            "File .pt ini hanya berisi state_dict (bobot), bukan model utuh. "
            "Tambahkan definisi arsitektur di fungsi load_torch(), contoh:\n"
            "model = NamaArsitektur(); model.load_state_dict(obj); model.eval()"
        )
    obj.eval()
    return obj, "torch"


def preprocess(img: Image.Image, size, mode_idx):
    h, w = size
    arr = np.asarray(img.convert("RGB").resize((w, h)), dtype=np.float32)
    if mode_idx == 0:
        arr /= 255.0
    elif mode_idx == 2:
        arr = arr / 127.5 - 1.0
    elif mode_idx == 3:
        arr = (arr / 255.0 - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / \
              np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return arr  # H, W, 3


def to_probs(out):
    """Ubah output model (sigmoid 1 neuron ATAU softmax 2 neuron) menjadi [p_kelas0, p_kelas1]."""
    v = np.asarray(out, dtype=np.float64).reshape(-1)
    if v.size == 1:                         # sigmoid: nilai = peluang kelas indeks 1
        p = v[0] if 0.0 <= v[0] <= 1.0 else 1.0 / (1.0 + np.exp(-v[0]))
        return np.array([1.0 - p, p])
    if v.min() < 0 or v.max() > 1 or not np.isclose(v.sum(), 1.0, atol=1e-3):
        e = np.exp(v - v.max())
        v = e / e.sum()                     # logits -> softmax
    return v[:2]


def yolo_detect(model, img):
    """
    Cara baca sama seperti dashboard.py: model(np.array(img)) -> boxes + plot().
    Mengembalikan (probs, gambar_berkotak). probs = None bila tidak ada objek terdeteksi.
    Jika model YOLO adalah model klasifikasi (r.probs ada), dipakai langsung.
    """
    r = model(np.array(img.convert("RGB")), verbose=False)[0]
    annotated = r.plot()[:, :, ::-1]           # BGR -> RGB agar warna benar di Streamlit

    # model klasifikasi YOLO (*-cls.pt)
    if getattr(r, "probs", None) is not None:
        return to_probs(r.probs.data.cpu().numpy()), None

    # model deteksi objek
    boxes = r.boxes
    if boxes is None or len(boxes) == 0:
        return None, annotated                 # tidak ada objek terdeteksi

    name_to_idx = {n.lower(): i for i, n in enumerate(CLASS_NAMES)}
    p = np.zeros(len(CLASS_NAMES))
    for c, s in zip(boxes.cls.cpu().numpy().astype(int),
                    boxes.conf.cpu().numpy()):
        i = name_to_idx.get(str(r.names[c]).lower(), int(c))   # cocokkan lewat nama kelas
        if i < len(p):
            p[i] = max(p[i], float(s))         # confidence tertinggi per kelas
    return (p if p.sum() > 0 else None), annotated


def predict(img, framework, path, norm_idx, torch_size):
    """Mengembalikan (probs, gambar_berkotak_atau_None)."""
    if framework == "Keras (.h5)":
        model = load_keras(path)
        shp = model.input_shape
        shp = shp[0] if isinstance(shp, list) else shp
        h = shp[1] or DEFAULT_IMG_SIZE
        w = shp[2] or DEFAULT_IMG_SIZE
        x = preprocess(img, (h, w), norm_idx)[None, ...]
        return to_probs(model.predict(x, verbose=0)), None

    model, kind = load_torch(path)
    if kind == "yolo":
        return yolo_detect(model, img)

    import torch
    x = preprocess(img, (torch_size, torch_size), norm_idx).transpose(2, 0, 1)[None, ...]
    with torch.no_grad():
        out = model(torch.from_numpy(np.ascontiguousarray(x)))
    if isinstance(out, (list, tuple)):
        out = out[0]
    return to_probs(out.detach().cpu().numpy()), None


# ============================================================
# HASIL
# ============================================================
def pbar(name, p, dim=False):
    return (f'<div class="pbar {"dim" if dim else ""}"><div class="row"><span>{name}</span>'
            f'<span>{p*100:.1f}%</span></div><div class="track"><i style="width:{p*100:.1f}%"></i></div></div>')


def show_result(probs, names):
    top = int(np.argmax(probs))
    label, emoji, desc = LABEL_INFO.get(names[top], (names[top].title(), "", ""))
    bars = "".join(pbar(LABEL_INFO.get(n, (n,))[0], float(p), dim=(i != top))
                   for i, (n, p) in enumerate(zip(names, probs)))
    st.markdown(
        f"""<div class="result ok"><div class="r-cap">Hasil identifikasi</div>
        <div class="r-main script">{label} {emoji}</div>
        <div class="r-desc">{desc} Keyakinan model {probs[top]*100:.1f}%.</div>{bars}</div>""",
        unsafe_allow_html=True,
    )


def analyze(file, settings):
    """Jalankan prediksi lalu tampilkan gambar (berkotak bila deteksi) + hasil."""
    img = ImageOps.exif_transpose(Image.open(file))
    c1, c2 = st.columns([1, 1], gap="large")

    if not os.path.exists(settings["path"]):
        with c1:
            st.image(img, use_container_width=True)
        with c2:
            st.error(f"File model tidak ditemukan: `{settings['path']}`. "
                     "Letakkan file model di folder yang sama dengan app.py atau ubah path di bagian Pengaturan model.")
        return

    try:
        with st.spinner("Model sedang menganalisis..."):
            probs, annotated = predict(img, settings["framework"], settings["path"],
                                       settings["norm"], settings["size"])
    except ImportError as e:
        with c1:
            st.image(img, use_container_width=True)
        with c2:
            st.error(f"Library belum terpasang: {e}. Jalankan `pip install -r requirements.txt`.")
        return
    except Exception as e:
        with c1:
            st.image(img, use_container_width=True)
        with c2:
            st.error(f"Gagal menjalankan model: {e}")
        return

    with c1:
        st.image(annotated if annotated is not None else img, use_container_width=True)
    with c2:
        if probs is None:
            st.warning("⚠️ Tidak ada objek yang terdeteksi pada gambar ini.")
        else:
            names = CLASS_NAMES[::-1] if settings["flip"] else CLASS_NAMES
            show_result(probs, names)


# ============================================================
# HALAMAN UTAMA
# ============================================================
def main():
    render_header()

    st.markdown('<div id="coba"></div><h2 class="sec-title" style="margin-top:20px">Coba sekarang</h2>',
                unsafe_allow_html=True)

    with st.expander("Pengaturan model"):
        framework = st.radio("Model yang dipakai", ["Keras (.h5)", "PyTorch (.pt)"], horizontal=True)
        path = st.text_input("Lokasi file model", MODEL_H5_PATH if framework.startswith("Keras") else MODEL_PT_PATH,
                             key=f"path_{framework}")
        c1, c2, c3 = st.columns(3)
        norm = c1.selectbox("Normalisasi input", range(len(NORM_OPTIONS)),
                            format_func=lambda i: NORM_OPTIONS[i],
                            index=0 if framework.startswith("Keras") else 0)
        size = c2.number_input("Ukuran input (khusus .pt non-YOLO)", 32, 1024, DEFAULT_IMG_SIZE, 32,
                               disabled=framework.startswith("Keras"))
        flip = c3.checkbox("Balik urutan label", help="Centang jika hasil terbalik "
                           "(kacamata terbaca tidak kacamata, dan sebaliknya). "
                           "Jangan dicentang untuk model YOLO: kelas sudah dicocokkan lewat nama.")
    settings = dict(framework=framework, path=path, norm=norm, size=int(size), flip=flip)

    tab_up, tab_cam = st.tabs(["📁  Unggah gambar", "📷  Kamera"])

    with tab_up:
        f = st.file_uploader("Pilih foto wajah (JPG / PNG)", type=["jpg", "jpeg", "png", "webp"],
                             label_visibility="collapsed")
        if f:
            analyze(f, settings)
        else:
            st.markdown('<div class="placeholder">Belum ada gambar. Seret foto ke kotak di atas atau klik Browse files.</div>',
                        unsafe_allow_html=True)

    with tab_cam:
        st.caption("Izinkan akses kamera di browser. Kamera hanya bisa dipakai lewat localhost atau HTTPS.")
        shot = st.camera_input("Ambil foto wajah", label_visibility="collapsed")
        if shot:
            analyze(shot, settings)

    render_steps()
    render_ticker()


main()