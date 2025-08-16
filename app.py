# app.py — Future Tech Digital Twin (2026) — Free Streamlit Edition
# -----------------------------------------------------------------
# This single-file app showcases 10 cutting‑edge tech skills/modules
# using ONLY lightweight, free, no‑GPU dependencies so it runs on
# Streamlit Community Cloud (free) + GitHub (free).
#
# Modules (mapped to the 10 skills):
# 1) Quantum Computing (simulated)
# 2) Agentic AI (rule‑based multi‑agent debate)
# 3) GenAI + RAG (tiny TF‑IDF‑like retrieval, no transformers)
# 4) Vision‑Language‑Action (image heuristics → action suggestions)
# 5) Multimodal AI (fuses text+image features)
# 6) Edge AI / 6G (simulated IoT stream + latency test)
# 7) Computer Vision / OCR / ICR (demo layout parsing on provided text)
# 8) Neurotech / BCI (simulated EEG + simple classifier)
# 9) Cloud + AI Infra (mock ops dashboard + real HTTP latency)
# 10) Post‑Quantum Security (SHA‑3 / SHAKE demo — illustrative only)
#
# How to run locally:
#   pip install -r requirements.txt
#   streamlit run app.py
#
# Included at bottom: a small requirements.txt you can copy to repo.

from __future__ import annotations
import io
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

# --- Light NLP (classical) ---
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer as SumyTokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.utils import get_stop_words

# Ensure NLTK data on first run (Streamlit Cloud safe)
for pkg, path in [("punkt", "tokenizers/punkt"), ("stopwords", "corpora/stopwords")]:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(pkg, quiet=True)

st.set_page_config(page_title="🌍 Future Tech Digital Twin (Free)", layout="wide", page_icon="🌐")
st.title("🌍 Future Tech Digital Twin — 10 Skills Demo (Free Tier)")
st.caption("Lightweight modules that run on Streamlit Community Cloud without GPUs or private keys.")

# ------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------

def kpi_row(items):
    cols = st.columns(len(items))
    for col, (label, value, helptext) in zip(cols, items):
        with col:
            st.metric(label, value, help=helptext)


def simple_bar(labels, values, title: str = ""):
    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel("Value")
    st.pyplot(fig)


# ------------------------------------------------------------
# 1) Quantum Computing (simulated)
# ------------------------------------------------------------

def module_quantum():
    st.subheader("1) Quantum Computing — Simulated Circuits")
    st.write("This demo simulates a 1‑qubit circuit with basic gates and shows measurement statistics.")

    gate = st.selectbox("Gate", ["I (identity)", "X (NOT)", "H (Hadamard)", "RX(θ)", "RZ(φ)"])
    theta = st.slider("θ (for RX)", 0.0, 2*math.pi, math.pi/2, 0.01)
    phi = st.slider("φ (for RZ)", 0.0, 2*math.pi, math.pi/3, 0.01)
    shots = st.slider("Shots", 100, 5000, 1000, 100)

    # State |0⟩ as vector
    state = np.array([1+0j, 0+0j])

    # Define gates
    X = np.array([[0,1],[1,0]], dtype=complex)
    H = (1/math.sqrt(2)) * np.array([[1,1],[1,-1]], dtype=complex)
    RX = np.array([[math.cos(theta/2), -1j*math.sin(theta/2)],
                   [-1j*math.sin(theta/2), math.cos(theta/2)]], dtype=complex)
    RZ = np.array([[np.exp(-1j*phi/2), 0],[0, np.exp(1j*phi/2)]], dtype=complex)

    if gate.startswith("X"):
        U = X
    elif gate.startswith("H"):
        U = H
    elif gate.startswith("RX"):
        U = RX
    elif gate.startswith("RZ"):
        U = RZ
    else:
        U = np.eye(2, dtype=complex)

    out = U @ state
    # Measurement probabilities
    p0 = float(np.abs(out[0])**2)
    p1 = float(np.abs(out[1])**2)

    # Sample shots
    results = np.random.choice([0,1], size=shots, p=[p0, p1])
    c0 = int(np.sum(results==0))
    c1 = shots - c0

    kpi_row([
        ("P(0)", f"{p0:.3f}", "Probability of measuring |0⟩"),
        ("P(1)", f"{p1:.3f}", "Probability of measuring |1⟩"),
        ("Shots", str(shots), "Number of measurements")
    ])

    simple_bar(["0","1"], [c0, c1], title="Measurement counts")


# ------------------------------------------------------------
# 2) Agentic AI (rule‑based multi‑agent debate)
# ------------------------------------------------------------

def module_agentic():
    st.subheader("2) Agentic AI — Rule‑based Multi‑Agent Debate")
    prompt = st.text_area("Give the agents a problem to discuss", "How should a city lower air pollution quickly?")

    if st.button("Run Agents"):
        # Three simple agents with different personas
        def analyst(p):
            return f"Analyst: Key factors include emissions, traffic, and industry. For '{p}', prioritize measuring PM2.5, traffic densification, and fuel mix."
        def planner(p):
            return f"Planner: Short‑term '{p}' steps: expand public transit, odd‑even traffic days, retrofit factories with filters."
        def skeptic(p):
            return f"Skeptic: Risks for '{p}': public pushback, economic slowdown. Need incentives and phased rollout."

        turns = [analyst(prompt), planner(prompt), skeptic(prompt)]
        for t in turns:
            st.write(t)

        st.success("Consensus: Pilot low‑emission zones + subsidize clean transit; monitor PM2.5 weekly.")


# ------------------------------------------------------------
# 3) GenAI + RAG (lightweight retrieval, no transformers)
# ------------------------------------------------------------

def rag_retrieve(query: str, docs: list[str], top_k=3):
    """Naive TF‑IDF‑ish retrieval with cosine on word counts (pure Python)."""
    stops = set(stopwords.words('english'))
    def bow(text):
        toks = [w.lower() for w in word_tokenize(text) if w.isalpha() and w.lower() not in stops]
        return Counter(toks)
    qv = bow(query)
    def cos(a: Counter, b: Counter):
        inter = set(a) & set(b)
        num = sum(a[t]*b[t] for t in inter)
        na = math.sqrt(sum(v*v for v in a.values()))
        nb = math.sqrt(sum(v*v for v in b.values()))
        return 0.0 if na==0 or nb==0 else num/(na*nb)
    scored = [(cos(qv, bow(d)), d) for d in docs]
    return [d for s,d in sorted(scored, reverse=True)[:top_k] if s>0]


def module_rag():
    st.subheader("3) GenAI + RAG — Lightweight Retrieval")
    st.caption("Uploads a small corpus, retrieves relevant chunks, then summarizes with classical NLP.")

    corpus = st.text_area("Paste knowledge base (one or more paragraphs)",
                          "Digital twins mirror real systems. They ingest real‑time telemetry, run simulations, and support decisions.")
    question = st.text_input("Your question", "How does a digital twin help operations?")

    if st.button("Retrieve + Summarize"):
        docs = [p.strip() for p in corpus.split("\n") if p.strip()]
") if p.strip()]
        hits = rag_retrieve(question, docs, top_k=3)
        if not hits:
            st.warning("No relevant chunks found.")
            return
        st.markdown("**Top retrieved chunks:**")
        for h in hits:
            st.info(h)

        # Summarize concatenated hits using LexRank/LSA/Luhn
        joined = "
".join(hits)
        parser = PlaintextParser.from_string(joined, SumyTokenizer("english"))
        for name, Summ in [("LexRank", LexRankSummarizer), ("LSA", LsaSummarizer), ("Luhn", LuhnSummarizer)]:
            summ = Summ()
            summ.stop_words = get_stop_words("english")
            out = " ".join(str(s) for s in summ(parser.document, 3))
            st.markdown(f"**{name} summary:**

{out}")


# ------------------------------------------------------------
# 4) Vision‑Language‑Action (heuristics)
# ------------------------------------------------------------

def module_vla():
    st.subheader("4) Vision‑Language‑Action — Heuristic Caption + Action")
    img_file = st.file_uploader("Upload an image (jpg/png)", type=["jpg","jpeg","png"])
    if not img_file:
        st.info("Upload an image to analyze dominant colors and propose an action.")
        return

    img = Image.open(img_file).convert("RGB")
    st.image(img, caption="Uploaded image", use_column_width=True)

    # Dominant color via simple resize + histogram
    small = img.resize((64,64))
    arr = np.array(small).reshape(-1,3)
    # Bin to 8x8x8
    bins = (arr//32).astype(int)
    keys, counts = np.unique(bins, axis=0, return_counts=True)
    top = keys[counts.argmax()]
    dom_rgb = tuple(int((v+0.5)*32) for v in top)

    st.write(f"Dominant color (approx): RGB {dom_rgb}")

    # Heuristic caption/action
    r,g,b = dom_rgb
    if g>r and g>b:
        caption = "Scene likely outdoors/greenery."
        action = "Suggest: Enhance contrast or plan outdoor route."
    elif b>r and b>g:
        caption = "Scene has lots of sky/water tones."
        action = "Suggest: Weather check before scheduling outdoor tasks."
    else:
        caption = "Warm/urban objects likely present."
        action = "Suggest: Detect edges/objects (advanced CV in full system)."

    st.success(f"Caption: {caption}

Action: {action}")


# ------------------------------------------------------------
# 5) Multimodal AI (fuse text+image)
# ------------------------------------------------------------

def module_multimodal():
    st.subheader("5) Multimodal AI — Fuse Text + Image Features")
    txt = st.text_area("Enter text context", "A coastal city preparing for storms.")
    img_file = st.file_uploader("Optional image (jpg/png)", type=["jpg","jpeg","png"])

    features = {}

    # Text keyword features
    toks = [w.lower() for w in word_tokenize(txt) if w.isalpha()]
    freq = Counter(toks)
    features.update({f"kw_{k}":v for k,v in freq.items() if v>=2})

    # Image dominant color feature
    if img_file:
        img = Image.open(img_file).convert("RGB")
        small = img.resize((32,32))
        arr = np.array(small).reshape(-1,3)
        dom = tuple(int(np.mean(arr[:,i])) for i in range(3))
        features["img_r"], features["img_g"], features["img_b"] = dom

    st.json(features if features else {"note":"No strong features extracted yet."})

    # Simple rule fusion
    if features.get("kw_storm",0) or features.get("kw_flood",0):
        if features.get("img_b",0) > 140:
            msg = "High storm risk with coastal/sky visuals → prepare sandbags and evacuation routes."
        else:
            msg = "Text indicates storms; image neutral → verify weather radar and readiness."
    else:
        msg = "Insufficient risk signals; continue monitoring."

    st.success(msg)


# ------------------------------------------------------------
# 6) Edge AI / 6G (simulated IoT + latency)
# ------------------------------------------------------------

def module_edge():
    st.subheader("6) Edge AI / 6G — Simulated Telemetry + Latency Probe")
    n = st.slider("Number of sensor points", 50, 500, 200, 50)
    np.random.seed(42)
    t = np.arange(n)
    temp = 25 + 2*np.sin(t/12) + np.random.normal(0, 0.3, size=n)
    latency_4g = np.random.normal(40, 6, size=n)
    latency_6g = np.random.normal(5, 1.2, size=n)

    fig, ax = plt.subplots()
    ax.plot(t, temp, label="Temperature (°C)")
    ax.set_title("Simulated IoT stream")
    ax.legend()
    st.pyplot(fig)

    fig2, ax2 = plt.subplots()
    ax2.plot(t, latency_4g, label="4G latency (ms)")
    ax2.plot(t, latency_6g, label="6G latency (ms)")
    ax2.legend()
    ax2.set_title("Latency comparison")
    st.pyplot(fig2)

    # Real HTTP latency (best‑effort; uses a public fast endpoint)
    target = st.text_input("Latency test URL", "https://httpbin.org/get")
    if st.button("Ping once"):
        t0 = time.time()
        try:
            r = requests.get(target, timeout=10)
            dt = (time.time()-t0)*1000
            st.success(f"HTTP GET {r.status_code} in {dt:.1f} ms")
        except Exception as e:
            st.warning(f"Latency test failed: {e}")


# ------------------------------------------------------------
# 7) Computer Vision / OCR / ICR (text layout demo)
# ------------------------------------------------------------

def module_ocr():
    st.subheader("7) OCR / ICR — Layout Parsing Demo (No heavy OCR)")
    st.caption("Free tier cannot run heavy OCR; paste raw text or simple receipt text to see parsing.")
    raw = st.text_area("Paste document text (e.g., receipt/invoice)",
                       """INVOICE
No: 12345
Date: 2026-08-15
Item A 2 x 10.00 = 20.00
Item B 1 x 5.00 = 5.00
TOTAL: 25.00
""")
    if st.button("Parse"):
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        fields = {}
        for ln in lines:
            if ln.lower().startswith("no:"):
                fields["invoice_no"] = ln.split(":",1)[1].strip()
            if ln.lower().startswith("date:"):
                fields["date"] = ln.split(":",1)[1].strip()
            if ln.upper().startswith("TOTAL"):
                parts = ln.replace(" ", "").split(":")
                if len(parts)>1:
                    fields["total"] = parts[1]
        st.json({"fields": fields, "lines": lines})
        st.success("Parsed fields extracted. In production, plug in Tesseract/EasyOCR server.")


# ------------------------------------------------------------
# 8) Neurotech / BCI (simulated EEG)
# ------------------------------------------------------------

def module_neuro():
    st.subheader("8) Neurotech / BCI — Simulated EEG Classifier")
    secs = st.slider("Duration (s)", 2, 20, 8)
    fs = 128
    t = np.linspace(0, secs, secs*fs)
    # alpha (10 Hz) and beta (20 Hz) components
    alpha = np.sin(2*np.pi*10*t)
    beta = 0.6*np.sin(2*np.pi*20*t)
    noise = 0.4*np.random.randn(len(t))
    eeg = alpha + beta + noise

    fig, ax = plt.subplots()
    ax.plot(t, eeg)
    ax.set_title("Simulated EEG signal")
    ax.set_xlabel("s")
    st.pyplot(fig)

    # Very naive classifier: power ratio
    def band_power(sig, f_low, f_high):
        sp = np.fft.rfft(sig)
        freqs = np.fft.rfftfreq(len(sig), 1/fs)
        mask = (freqs>=f_low) & (freqs<=f_high)
        return float(np.mean(np.abs(sp[mask])**2))

    p_alpha = band_power(eeg, 8, 13)
    p_beta = band_power(eeg, 13, 30)
    state = "Relaxed" if p_alpha>p_beta else "Focused"

    kpi_row([
        ("P_alpha", f"{p_alpha:.2f}", "8–13 Hz"),
        ("P_beta", f"{p_beta:.2f}", "13–30 Hz"),
        ("State", state, "Naive power ratio")
    ])


# ------------------------------------------------------------
# 9) Cloud + AI Infrastructure (mock ops)
# ------------------------------------------------------------

def module_cloud():
    st.subheader("9) Cloud + AI Infra — Mock Ops Dashboard")
    np.random.seed(0)
    ts = pd.date_range(end=pd.Timestamp.now(), periods=60, freq="min")
    cpu = np.clip(np.random.normal(55, 10, size=len(ts)), 0, 100)
    gpu = np.clip(np.random.normal(48, 12, size=len(ts)), 0, 100)
    mem = np.clip(np.random.normal(62, 8, size=len(ts)), 0, 100)
    df = pd.DataFrame({"time": ts, "cpu": cpu, "gpu": gpu, "mem": mem}).set_index("time")

    st.line_chart(df)
    kpi_row([
        ("CPU now", f"{df['cpu'][-1]:.1f}%", ""),
        ("GPU now", f"{df['gpu'][-1]:.1f}%", ""),
        ("Mem now", f"{df['mem'][-1]:.1f}%", ""),
    ])

    st.caption("Hook these to real metrics via Prometheus/Grafana or cloud APIs when available.")


# ------------------------------------------------------------
# 10) Post‑Quantum Security (illustrative only)
# ------------------------------------------------------------

def module_pq():
    st.subheader("10) Post‑Quantum Security — SHA‑3 / SHAKE Demo (Illustrative)")
    import hashlib
    msg = st.text_input("Message", "critical telemetry packet")
    if st.button("Hash with SHA3-256"):
        h = hashlib.sha3_256(msg.encode()).hexdigest()
        st.code(h)
    if st.button("XOF with SHAKE-128 (32 bytes)"):
        x = hashlib.shake_128(msg.encode()).hexdigest(32)
        st.code(x)
    st.info("Note: True PQC (e.g., CRYSTALS‑Kyber/Dilithium) requires extra libs/servers not available on free tier.")


# ------------------------------------------------------------
# Sidebar navigation
# ------------------------------------------------------------
modules = {
    "Quantum": module_quantum,
    "Agentic AI": module_agentic,
    "GenAI + RAG": module_rag,
    "VLA (Vision‑Language‑Action)": module_vla,
    "Multimodal": module_multimodal,
    "Edge / 6G": module_edge,
    "OCR / ICR": module_ocr,
    "Neurotech / BCI": module_neuro,
    "Cloud + AI Infra": module_cloud,
    "Post‑Quantum Sec": module_pq,
}

st.sidebar.header("Choose a module")
choice = st.sidebar.radio("Modules", list(modules.keys()))

# Render selected module
modules[choice]()

st.sidebar.divider()
st.sidebar.subheader("Requirements (for your repo)")
st.sidebar.code(
    """
    streamlit
    pandas
    numpy
    requests
    matplotlib
    pillow
    nltk
    sumy
    """.strip(), language="text"
)

# Footer
st.markdown("---")
st.caption("This is a free‑tier friendly educational demo. Swap the simulated pieces with real APIs/models when you upgrade infra.")


# -------------------------------
# requirements.txt (save as a separate file)
# -------------------------------
# Copy the exact lines below into a new file named requirements.txt
# in the same repo as app.py
"""
streamlit
pandas
numpy
requests
matplotlib
pillow
nltk
sumy
"""
