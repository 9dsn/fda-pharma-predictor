"""Streamlit deployment app for the FDA AdCom Vote Predictor."""

import re
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdfplumber
import shap
import streamlit as st


st.set_page_config(
    page_title="FDA AdCom Vote Predictor",
    page_icon="💊",
    layout="wide",
)


MODEL_DIR = Path("saved_model")
POSITIVE_VOCAB = [
    "efficacious", "beneficial", "significant improvement", "clinically meaningful",
    "durable response", "well tolerated", "favorable risk-benefit", "survival benefit",
    "superior", "statistically significant", "meets primary endpoint",
]
NEGATIVE_VOCAB = [
    "concern", "uncertainty", "insufficient evidence", "failed to demonstrate",
    "adverse events", "safety signal", "not statistically significant",
    "does not support", "unresolved", "limited data", "exploratory",
]


def inject_custom_css() -> None:
    """Inject the premium dark biopharma theme and motion system.

    The app keeps Streamlit's native controls, while CSS variables, selectors,
    and keyframe animations reshape the default UI into a trading-style surface.
    """
    st.markdown(
        """
        <style>
        :root {
            --blue: #378ADD;
            --teal: #1D9E75;
            --text: #E6F1FB;
            --muted: #8FB2CC;
            --line: rgba(55, 138, 221, 0.25);
            --glow-blue: 0 0 28px rgba(55, 138, 221, 0.45);
            --glow-teal: 0 0 28px rgba(29, 158, 117, 0.38);
            --bg: #0a0f1e;
            --card: #0d1a2e;
            --card-2: #10233b;
            --amber: #BA7517;
            --red: #A32D2D;
        }

        #MainMenu, footer, header, [data-testid="stToolbar"] {
            display: none !important;
            visibility: hidden !important;
            height: 0 !important;
        }

        iframe[title="st.iframe"] {
            position: fixed !important;
            inset: 0 !important;
            width: 100vw !important;
            height: 100vh !important;
            pointer-events: none !important;
            z-index: 0 !important;
            border: 0 !important;
        }

        html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
            background:
                radial-gradient(circle at 18% 12%, rgba(55, 138, 221, 0.18), transparent 32%),
                radial-gradient(circle at 82% 8%, rgba(29, 158, 117, 0.14), transparent 30%),
                linear-gradient(135deg, #0a0f1e 0%, #091426 48%, #07111f 100%) !important;
            color: var(--text) !important;
        }

        .main .block-container {
            max-width: 1220px;
            padding-top: 1.8rem;
            padding-bottom: 4rem;
        }

        h1, h2, h3, h4, p, label, span, div {
            color: var(--text);
        }

        a, a:visited {
            color: #7ec8ff;
        }

        .bio-hero {
            position: relative;
            overflow: hidden;
            min-height: 300px;
            padding: 2.2rem;
            border: 1px solid var(--line);
            border-radius: 24px;
            background:
                linear-gradient(125deg, rgba(13, 26, 46, 0.94), rgba(8, 18, 34, 0.84)),
                radial-gradient(circle at 74% 42%, rgba(29, 158, 117, 0.16), transparent 36%);
            box-shadow: var(--glow-blue), inset 0 1px 0 rgba(255,255,255,0.08);
            animation: fadeLift 0.85s ease both;
            transform-style: preserve-3d;
        }

        .hero-grid {
            display: grid;
            grid-template-columns: minmax(0, 1.1fr) minmax(260px, 0.9fr);
            gap: 2rem;
            align-items: center;
        }

        .eyebrow {
            display: inline-flex;
            gap: 0.45rem;
            align-items: center;
            padding: 0.35rem 0.7rem;
            border: 1px solid rgba(29, 158, 117, 0.42);
            border-radius: 999px;
            background: rgba(29, 158, 117, 0.11);
            color: #9ff4d2;
            font-size: 0.78rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .hero-title {
            margin: 1rem 0 0.7rem;
            font-size: clamp(2.4rem, 5vw, 4.9rem);
            line-height: 0.95;
            font-weight: 800;
            letter-spacing: 0;
            color: var(--text);
            text-shadow: 0 0 30px rgba(55, 138, 221, 0.38);
        }

        .hero-copy {
            max-width: 680px;
            color: var(--muted);
            font-size: 1.05rem;
            line-height: 1.7;
        }

        .hero-kpis {
            display: flex;
            flex-wrap: wrap;
            gap: 0.8rem;
            margin-top: 1.35rem;
        }

        .hero-chip {
            padding: 0.7rem 0.9rem;
            border-radius: 14px;
            border: 1px solid rgba(55, 138, 221, 0.24);
            background: rgba(6, 16, 32, 0.72);
            color: #cbe9ff;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.06);
            transition: all 0.3s ease;
        }

        .hero-chip:hover {
            transform: translateY(-2px);
            border-color: rgba(29, 158, 117, 0.55);
            box-shadow: var(--glow-teal);
        }

        .molecule-stage {
            position: relative;
            min-height: 250px;
            perspective: 900px;
        }

        .molecule {
            position: absolute;
            inset: 8% 5% 3% 8%;
            transform-style: preserve-3d;
        }

        .helix-line {
            position: absolute;
            left: 50%;
            top: 50%;
            width: 76%;
            height: 3px;
            border-radius: 999px;
            background: linear-gradient(90deg, transparent, var(--blue), var(--teal), transparent);
            box-shadow: 0 0 18px rgba(55, 138, 221, 0.7);
            transform-style: preserve-3d;
        }

        .helix-line:nth-child(1) { transform: translate(-50%, -50%) rotateZ(0deg) rotateY(58deg); }
        .helix-line:nth-child(2) { transform: translate(-50%, -50%) rotateZ(30deg) rotateY(58deg); }
        .helix-line:nth-child(3) { transform: translate(-50%, -50%) rotateZ(60deg) rotateY(58deg); }
        .helix-line:nth-child(4) { transform: translate(-50%, -50%) rotateZ(90deg) rotateY(58deg); }
        .helix-line:nth-child(5) { transform: translate(-50%, -50%) rotateZ(120deg) rotateY(58deg); }
        .helix-line:nth-child(6) { transform: translate(-50%, -50%) rotateZ(150deg) rotateY(58deg); }

        .atom {
            position: absolute;
            width: 18px;
            height: 18px;
            border-radius: 50%;
            background: radial-gradient(circle at 30% 30%, #e8fbff, var(--blue) 48%, #0b4f8d 100%);
            box-shadow: 0 0 22px rgba(55, 138, 221, 0.9);
        }

        .atom.teal {
            background: radial-gradient(circle at 30% 30%, #eafff6, var(--teal) 48%, #0c604b 100%);
            box-shadow: 0 0 22px rgba(29, 158, 117, 0.9);
        }

        .atom:nth-of-type(7) { left: 16%; top: 18%; transform: translateZ(80px); }
        .atom:nth-of-type(8) { left: 73%; top: 18%; transform: translateZ(-45px); }
        .atom:nth-of-type(9) { left: 24%; top: 52%; transform: translateZ(-70px); }
        .atom:nth-of-type(10) { left: 68%; top: 54%; transform: translateZ(65px); }
        .atom:nth-of-type(11) { left: 43%; top: 81%; transform: translateZ(35px); }

        .glass-panel, [data-testid="stExpander"], .stAlert {
            border: 1px solid var(--line) !important;
            border-radius: 18px !important;
            background: rgba(13, 26, 46, 0.78) !important;
            box-shadow: 0 18px 50px rgba(0, 0, 0, 0.28), inset 0 1px 0 rgba(255,255,255,0.06);
            backdrop-filter: blur(14px);
        }

        .section-shell {
            margin: 1.3rem 0;
            padding: 1.15rem;
            border: 1px solid var(--line);
            border-radius: 20px;
            background: rgba(13, 26, 46, 0.64);
            box-shadow: 0 18px 46px rgba(0,0,0,0.26);
            animation: fadeLift 0.75s ease both;
            transition: transform 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease;
        }

        .section-shell:hover {
            transform: perspective(900px) rotateX(0.8deg) translateY(-2px);
            border-color: rgba(55, 138, 221, 0.45);
            box-shadow: var(--glow-blue);
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.3rem;
            padding: 0.35rem;
            border: 1px solid var(--line);
            border-radius: 18px;
            background: rgba(7, 17, 31, 0.84);
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.06);
        }

        .stTabs [data-baseweb="tab"] {
            position: relative;
            min-height: 50px;
            padding: 0 1.2rem;
            border-radius: 14px;
            color: var(--muted);
            transition: all 0.3s ease;
        }

        .stTabs [aria-selected="true"] {
            color: var(--text) !important;
            background: linear-gradient(180deg, rgba(55,138,221,0.25), rgba(29,158,117,0.08));
            box-shadow: var(--glow-blue);
        }

        .stTabs [aria-selected="true"]::after {
            content: "";
            position: absolute;
            left: 16px;
            right: 16px;
            bottom: 2px;
            height: 3px;
            border-radius: 999px;
            background: linear-gradient(90deg, var(--blue), var(--teal));
            animation: activeTabGlow 1.8s ease-in-out infinite;
        }

        [data-testid="stFileUploader"] {
            padding: 1.2rem;
            border: 1px dashed rgba(55, 138, 221, 0.62);
            border-radius: 22px;
            background:
                linear-gradient(rgba(13, 26, 46, 0.74), rgba(13, 26, 46, 0.74)) padding-box,
                linear-gradient(120deg, rgba(55, 138, 221, 0.85), rgba(29, 158, 117, 0.85), rgba(55, 138, 221, 0.85)) border-box;
            box-shadow: var(--glow-blue);
            transition: all 0.3s ease;
        }

        [data-testid="stFileUploader"]:hover {
            transform: translateY(-2px);
            border-color: var(--teal);
            box-shadow: var(--glow-teal);
        }

        [data-testid="stFileUploaderDropzone"] {
            background: rgba(5, 12, 24, 0.58) !important;
            border: 0 !important;
            border-radius: 18px !important;
            min-height: 150px;
        }

        .upload-complete {
            margin-top: 0.75rem;
            padding: 0.75rem 1rem;
            border-radius: 14px;
            border: 1px solid rgba(29, 158, 117, 0.65);
            background: rgba(29, 158, 117, 0.14);
            color: #baffdf;
            box-shadow: var(--glow-teal);
            animation: fadeLift 0.45s ease both;
        }

        [data-testid="stMetric"] {
            position: relative;
            overflow: hidden;
            min-height: 132px;
            padding: 1.05rem;
            border: 1px solid var(--line);
            border-radius: 18px;
            background: linear-gradient(145deg, rgba(13, 26, 46, 0.94), rgba(8, 18, 35, 0.82));
            box-shadow: 0 18px 44px rgba(0,0,0,0.28);
            animation: metricRise 0.7s ease both;
            transition: all 0.3s ease;
        }

        [data-testid="stMetric"]::before {
            content: "";
            position: absolute;
            inset: -40%;
            background: conic-gradient(from 180deg, transparent, rgba(55,138,221,0.18), transparent, rgba(29,158,117,0.14), transparent);
        }

        [data-testid="stMetric"] > div {
            position: relative;
            z-index: 1;
        }

        [data-testid="stMetricLabel"] p {
            color: var(--muted) !important;
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        [data-testid="stMetricValue"] {
            color: var(--text) !important;
            text-shadow: 0 0 22px rgba(55, 138, 221, 0.45);
            animation: countUpFeel 0.85s ease both;
        }

        .metric-strip {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 1rem;
            margin: 1rem 0 1.25rem;
        }

        .custom-metric {
            position: relative;
            overflow: hidden;
            min-height: 116px;
            padding: 1rem;
            border: 1px solid var(--line);
            border-radius: 18px;
            background: linear-gradient(145deg, rgba(13, 26, 46, 0.94), rgba(8, 18, 35, 0.82));
            box-shadow: 0 18px 44px rgba(0,0,0,0.28);
            animation: metricRise 0.7s ease both;
            transition: all 0.3s ease;
        }

        .custom-metric:nth-child(2) { animation-delay: 0.08s; }
        .custom-metric:nth-child(3) { animation-delay: 0.16s; }
        .custom-metric:nth-child(4) { animation-delay: 0.24s; }

        .custom-metric::before {
            content: "";
            position: absolute;
            inset: -55%;
            background: conic-gradient(from 180deg, transparent, rgba(55,138,221,0.18), transparent, rgba(29,158,117,0.14), transparent);
        }

        .custom-metric span, .custom-metric strong {
            position: relative;
            z-index: 1;
            display: block;
        }

        .custom-metric span {
            color: var(--muted);
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .custom-metric strong {
            margin-top: 0.7rem;
            color: var(--text);
            font-size: clamp(1.7rem, 4vw, 2.4rem);
            line-height: 1;
            text-shadow: 0 0 22px rgba(55, 138, 221, 0.45);
            animation: countUpFeel 0.85s ease both;
        }

        .gauge-wrap {
            display: grid;
            grid-template-columns: minmax(240px, 360px) 1fr;
            gap: 1.4rem;
            align-items: center;
            padding: 1.25rem;
            border-radius: 22px;
            background: linear-gradient(145deg, rgba(13,26,46,0.96), rgba(8,18,34,0.84));
            border: 1px solid var(--line);
            box-shadow: var(--glow-blue);
            animation: resultCascade 0.55s ease both;
        }

        .svg-gauge-stage {
            position: relative;
            display: grid;
            place-items: center;
            perspective: 900px;
            min-height: 260px;
        }

        .svg-confidence-gauge {
            width: min(340px, 74vw);
            height: auto;
            overflow: visible;
            transform: perspective(900px) rotateX(22deg);
            filter: drop-shadow(0 24px 28px rgba(0,0,0,0.48)) drop-shadow(0 0 22px rgba(55,138,221,0.42));
        }

        .svg-confidence-gauge .gauge-value {
            stroke-dasharray: 100;
            stroke-dashoffset: 100;
            animation: svgGaugeSweep 1.2s cubic-bezier(.18,.9,.22,1) forwards;
        }

        .gauge-readout {
            position: absolute;
            text-align: center;
            transform: translateY(16px);
        }

        .gauge-number {
            font-size: 3rem;
            font-weight: 800;
            color: var(--text);
            text-shadow: 0 0 26px rgba(29,158,117,0.55);
            animation: countUpFeel 1s ease both;
        }

        .gauge-label {
            color: var(--muted);
            font-size: 0.82rem;
            letter-spacing: 0.11em;
            text-transform: uppercase;
        }

        .signal-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.55rem;
            margin-top: 1rem;
            padding: 0.85rem 1.1rem;
            border-radius: 999px;
            border: 1px solid rgba(29, 158, 117, 0.58);
            background: rgba(29, 158, 117, 0.13);
            color: #c8ffe5;
            box-shadow: var(--glow-teal);
            animation: resultCascade 0.65s ease 0.26s both;
        }

        .trade-signal-badge {
            --signal-color: var(--amber);
            position: relative;
            width: min(320px, 100%);
            margin: 1.25rem auto 0;
            padding: 1.2rem 1.6rem;
            border-radius: 999px;
            border: 1px solid color-mix(in srgb, var(--signal-color) 70%, transparent);
            background: linear-gradient(145deg, rgba(13,26,46,0.92), rgba(8,18,34,0.78));
            box-shadow: 0 0 30px color-mix(in srgb, var(--signal-color) 42%, transparent);
            text-align: center;
            animation: resultCascade 0.65s ease 0.32s both;
        }

        .trade-signal-badge::before {
            content: "";
            position: absolute;
            inset: -7px;
            border: 2px solid var(--signal-color);
            border-radius: inherit;
        }

        .trade-signal-badge span {
            display: block;
            color: var(--muted);
            font-size: 0.78rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
        }

        .trade-signal-badge strong {
            display: block;
            margin-top: 0.2rem;
            color: var(--text);
            font-size: clamp(2rem, 6vw, 3.2rem);
            line-height: 1;
            letter-spacing: 0.08em;
            text-shadow: 0 0 24px var(--signal-color);
        }

        .shap-frame {
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: 1rem;
            background: rgba(13, 26, 46, 0.72);
            box-shadow: 0 18px 46px rgba(0,0,0,0.25);
            animation: resultCascade 0.65s ease 0.16s both;
        }

        .shap-frame img, .shap-frame canvas {
            animation: shapBars 0.9s ease both;
            transform-origin: left center;
        }

        [data-testid="stDataFrame"], [data-testid="stTable"] {
            border: 1px solid var(--line);
            border-radius: 16px;
            overflow: hidden;
            box-shadow: 0 18px 46px rgba(0,0,0,0.24);
        }

        .stButton > button, button {
            border-radius: 12px !important;
            border: 1px solid rgba(55,138,221,0.42) !important;
            background: linear-gradient(135deg, rgba(55,138,221,0.28), rgba(29,158,117,0.16)) !important;
            color: var(--text) !important;
            transition: all 0.3s ease !important;
        }

        .stButton > button:hover, button:hover {
            transform: translateY(-1px);
            box-shadow: var(--glow-blue);
            border-color: var(--teal) !important;
        }

        @keyframes fadeLift {
            from { opacity: 0; transform: translateY(18px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes activeTabGlow {
            0%, 100% { opacity: 0.65; transform: scaleX(0.72); }
            50% { opacity: 1; transform: scaleX(1); }
        }

        @keyframes metricRise {
            from { opacity: 0; transform: translateY(14px) scale(0.98); }
            to { opacity: 1; transform: translateY(0) scale(1); }
        }

        @keyframes countUpFeel {
            from { opacity: 0; filter: blur(7px); transform: translateY(8px); }
            to { opacity: 1; filter: blur(0); transform: translateY(0); }
        }

        @keyframes svgGaugeSweep {
            from { stroke-dashoffset: 100; }
            to { stroke-dashoffset: var(--dash-final); }
        }

        @keyframes resultCascade {
            from { opacity: 0; transform: translateY(18px) scale(0.985); }
            to { opacity: 1; transform: translateY(0) scale(1); }
        }

        @keyframes gaugeFill {
            from { clip-path: inset(0 100% 0 0); filter: saturate(0.6); }
            to { clip-path: inset(0 0 0 0); filter: saturate(1); }
        }

        @keyframes shapBars {
            from { opacity: 0; transform: scaleX(0.82) translateX(-14px); filter: blur(4px); }
            to { opacity: 1; transform: scaleX(1) translateX(0); filter: blur(0); }
        }

        @media (max-width: 850px) {
            .hero-grid, .gauge-wrap, .metric-strip {
                grid-template-columns: 1fr;
            }
            .bio-hero {
                padding: 1.35rem;
            }
            .molecule-stage {
                min-height: 190px;
            }
            .hero-title {
                font-size: 2.6rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
def render_hero() -> None:
    """Render the animated 3D hero panel for the application header."""
    st.markdown(
        """
        <section class="bio-hero">
          <div class="hero-grid">
            <div>
              <div class="eyebrow">ODAC intelligence engine</div>
              <div class="hero-title">FDA AdCom Vote Predictor</div>
              <div class="hero-copy">
                NLP-derived clinical signal, Random Forest inference, SHAP explanation,
                and event-study context in one deployment-ready research terminal.
              </div>
              <div class="hero-kpis">
                <div class="hero-chip">25 validated meetings</div>
                <div class="hero-chip">LOOCV probabilities</div>
                <div class="hero-chip">XBI-adjusted backtest</div>
              </div>
            </div>
            <div class="molecule-stage">
              <div class="molecule">
                <i class="helix-line"></i><i class="helix-line"></i><i class="helix-line"></i>
                <i class="helix-line"></i><i class="helix-line"></i><i class="helix-line"></i>
                <b class="atom"></b><b class="atom teal"></b><b class="atom"></b>
                <b class="atom teal"></b><b class="atom"></b>
              </div>
            </div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def section_heading(title: str, subtitle: str = "") -> None:
    """Render a compact animated section heading with optional supporting text."""
    st.markdown(
        f"""
        <div class="section-shell">
          <h3 style="margin:0 0 0.25rem 0;">{title}</h3>
          <p style="margin:0;color:#8FB2CC;">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def count_vocab_hits(text: str, vocabulary: list[str]) -> int:
    """Count keyword and phrase hits with word boundaries for stable inference.

    The training corpus used TF-IDF, but a single uploaded PDF has no corpus IDF.
    These raw vocabulary counts support the documented single-document proxy.
    """
    lower_text = text.lower()
    return sum(
        len(re.findall(rf"\b{re.escape(term)}\b", lower_text))
        for term in vocabulary
    )


def build_tfidf_proxy_scores(text: str) -> tuple[float, float]:
    """Approximate the training TF-IDF features for one uploaded document.

    This is intentionally a keyword-count proxy, normalized per 1,000 words,
    because true IDF weights require fitting across a document corpus.
    """
    word_count = max(len(re.findall(r"\b\w+\b", text)), 1)
    positive_score = count_vocab_hits(text, POSITIVE_VOCAB) / word_count * 1000
    negative_score = count_vocab_hits(text, NEGATIVE_VOCAB) / word_count * 1000
    return positive_score, negative_score


def sentiment_ratios(text: str) -> tuple[float, float]:
    """Measure the share of substantive sentences with positive/negative terms."""
    sentences = re.split(r"[.!?]", text.lower())
    sentences = [sentence for sentence in sentences if len(sentence.strip()) >= 10]
    if not sentences:
        return 0.0, 0.0

    positive_count = 0
    negative_count = 0
    for sentence in sentences:
        if any(re.search(rf"\b{re.escape(word)}\b", sentence) for word in POSITIVE_VOCAB):
            positive_count += 1
        if any(re.search(rf"\b{re.escape(word)}\b", sentence) for word in NEGATIVE_VOCAB):
            negative_count += 1

    return positive_count / len(sentences), negative_count / len(sentences)


def concern_density(text: str) -> float:
    """Calculate concern-language density per 1,000 words."""
    matches = re.findall(r"\b(concern|risk|uncertainty|adverse|toxicity|safety)\b", text.lower())
    word_count = max(len(re.findall(r"\b\w+\b", text)), 1)
    return len(matches) / word_count * 1000


def is_float(value: str) -> bool:
    """Return whether a regex-captured p-value can be parsed as a float."""
    try:
        float(value)
        return True
    except ValueError:
        return False


def binary_flags(text: str) -> dict[str, int]:
    """Extract clinical and regulatory binary flags from briefing text."""
    lower_text = text.lower()
    p_values = re.findall(r"p\s*[=<]\s*([0-9.]+)", lower_text)
    p_strong = any(
        float(value) < 0.05
        for value in p_values
        if is_float(value) and float(value) > 0
    )
    os_mentioned = bool(re.search(r"\b(overall survival|os)\b", lower_text))
    pfs_mentioned = bool(re.search(r"\b(progression[-\s]?free survival|pfs)\b", lower_text))
    survival_positive = int(bool(re.search(
        r"(improved overall survival|overall survival benefit|"
        r"statistically significant overall survival|os benefit|os improvement)",
        lower_text,
    )))

    return {
        "survival_positive": survival_positive,
        "pfs_only": int(pfs_mentioned and not os_mentioned),
        "response_rate_mentioned": int(bool(re.search(r"\b(response rate|orr|objective response)\b", lower_text))),
        "safety_concern_flag": int(bool(re.search(
            r"(serious adverse|black box|sae|safety concern|toxicit|treatment-related death)",
            lower_text,
        ))),
        "accelerated_approval_flag": int(bool(re.search(r"\b(accelerated approval|breakthrough)\b", lower_text))),
        "p_value_strong": int(p_strong),
    }


def extract_features(text: str, feature_names: list[str]) -> pd.DataFrame:
    """Build one inference row and align it to the saved deployment feature order."""
    positive_score, negative_score = build_tfidf_proxy_scores(text)
    positive_ratio, negative_ratio = sentiment_ratios(text)
    feature_values = {
        "tfidf_positive_score": positive_score,
        "tfidf_negative_score": negative_score,
        "tfidf_balance": positive_score - negative_score,
        "sentiment_positive_ratio": positive_ratio,
        "sentiment_negative_ratio": negative_ratio,
        "concern_density": concern_density(text),
        **binary_flags(text),
    }
    return pd.DataFrame([{name: feature_values.get(name, 0) for name in feature_names}])


def extract_pdf_text(uploaded_file) -> str:
    """Read all text from an uploaded FDA briefing PDF with pdfplumber."""
    pages: list[str] = []
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return "\n".join(pages).strip()


# cache_resource is used because the model is a heavy Python object that should
# be loaded once per Streamlit process, not serialized like normal tabular data.
@st.cache_resource
def load_model_artifacts() -> tuple[object, list[str]]:
    """Load the persisted Random Forest and ordered feature names for inference."""
    model = joblib.load(MODEL_DIR / "model.pkl")
    feature_names = joblib.load(MODEL_DIR / "feature_names.pkl")
    return model, feature_names


def load_historical_results() -> pd.DataFrame | None:
    """Load historical LOOCV predictions, returning None if deployment data is absent."""
    path = MODEL_DIR / "historical.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def predict_yes_probability(model: object, features: pd.DataFrame) -> tuple[str, float]:
    """Predict the vote label and yes-class probability from aligned features."""
    yes_idx = list(model.classes_).index("yes")
    probability = float(model.predict_proba(features)[0, yes_idx])
    label = str(model.predict(features)[0])
    return label, probability


def trading_signal(prob_yes: float) -> str:
    """Translate calibrated model confidence into the project trading rule."""
    if prob_yes >= 0.65:
        return "Long signal"
    if prob_yes <= 0.35:
        return "Short signal"
    return "Skip / no trade"


def signal_tone(signal: str) -> str:
    """Return a color token for the trading signal badge."""
    if signal.startswith("Long"):
        return "#1D9E75"
    if signal.startswith("Short"):
        return "#A32D2D"
    return "#BA7517"


def display_signal(signal: str) -> str:
    """Map the existing trading rule text to the visual BUY/HOLD/SELL badge."""
    if signal.startswith("Long"):
        return "BUY"
    if signal.startswith("Short"):
        return "SELL"
    return "HOLD"


def render_probability_gauge(prob_yes: float, label: str, signal: str) -> None:
    """Render an animated 3D arc gauge for the model's yes probability."""
    percent = prob_yes * 100
    dash_final = 100 - percent
    tone = signal_tone(signal)
    badge_label = display_signal(signal)
    st.markdown(
        f"""
        <div class="gauge-wrap">
          <div class="svg-gauge-stage">
            <svg class="svg-confidence-gauge" viewBox="0 0 260 170" role="img" aria-label="Confidence gauge {percent:.1f} percent">
              <defs>
                <linearGradient id="confidenceGradient" x1="20" y1="140" x2="240" y2="140" gradientUnits="userSpaceOnUse">
                  <stop offset="0%" stop-color="#1D9E75" />
                  <stop offset="50%" stop-color="#378ADD" />
                  <stop offset="100%" stop-color="#E24B4A" />
                </linearGradient>
                <filter id="gaugeGlow" x="-30%" y="-60%" width="160%" height="220%">
                  <feGaussianBlur stdDeviation="4" result="blur" />
                  <feMerge>
                    <feMergeNode in="blur" />
                    <feMergeNode in="SourceGraphic" />
                  </feMerge>
                </filter>
              </defs>
              <path d="M 30 135 A 100 100 0 0 1 230 135"
                    fill="none"
                    stroke="rgba(55,138,221,0.15)"
                    stroke-width="18"
                    stroke-linecap="round"
                    pathLength="100" />
              <path class="gauge-value"
                    d="M 30 135 A 100 100 0 0 1 230 135"
                    fill="none"
                    stroke="url(#confidenceGradient)"
                    stroke-width="18"
                    stroke-linecap="round"
                    pathLength="100"
                    style="--dash-final:{dash_final:.2f};"
                    filter="url(#gaugeGlow)" />
            </svg>
            <div class="gauge-readout">
              <div class="gauge-number">{percent:.1f}%</div>
              <div class="gauge-label">P(yes)</div>
            </div>
          </div>
          <div>
            <div class="eyebrow">Prediction result</div>
            <h2 style="margin:0.75rem 0 0.35rem 0;font-size:2.2rem;">{label.upper()}</h2>
            <p style="margin:0;color:#8FB2CC;line-height:1.6;">
              The deployment Random Forest scores this briefing against the saved
              historical feature order and applies the project trading thresholds.
            </p>
            <div class="trade-signal-badge" style="--signal-color:{tone};">
              <span>Trading signal</span>
              <strong>{badge_label}</strong>
            </div>
            <div class="signal-badge" style="border-color:{tone};box-shadow:0 0 28px {tone}55;">
              <span>Rule detail</span><strong>{signal}</strong>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_strip(metrics: list[tuple[str, str]]) -> None:
    """Render animated custom metric cards while preserving displayed values."""
    cards = "".join(
        f"""
        <div class="custom-metric">
          <span>{label}</span>
          <strong>{value}</strong>
        </div>
        """
        for label, value in metrics
    )
    st.markdown(
        f"""
        <div class="metric-strip">
          {cards}
        </div>
        """,
        unsafe_allow_html=True,
    )


def plot_shap_waterfall(model: object, features: pd.DataFrame) -> None:
    """Explain one prediction with a SHAP waterfall plot for the yes class."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(features)
    values = shap_values.values
    base_values = shap_values.base_values

    if values.ndim == 3:
        # TreeExplainer may return (rows, features, classes) for classifiers;
        # class index 1 is the saved Random Forest's "yes" probability output.
        values = values[0, :, 1]
        base_value = base_values[0, 1] if np.asarray(base_values).ndim == 2 else base_values[1]
    else:
        values = values[0]
        base_value = base_values[0] if np.asarray(base_values).ndim else base_values

    explanation = shap.Explanation(
        values=values,
        base_values=base_value,
        data=features.iloc[0].values,
        feature_names=list(features.columns),
    )
    fig = plt.figure(figsize=(8, 4.5))
    shap.plots.waterfall(explanation, show=False, max_display=10)
    st.pyplot(fig)
    plt.close(fig)


def style_correct_column(value: bool) -> str:
    """Color historical correctness values green or red in the displayed table."""
    return "background-color: #d8f3dc" if bool(value) else "background-color: #ffd6d6"


def apply_correct_column_style(df: pd.DataFrame) -> object:
    """Style the correctness column across pandas versions.

    Pandas 3 uses `Styler.map`, while pandas 2 still commonly supports
    `Styler.applymap`, so this keeps the Hugging Face app tolerant of either.
    """
    styler = df.style
    if hasattr(styler, "map"):
        return styler.map(style_correct_column, subset=["correct"])
    return styler.applymap(style_correct_column, subset=["correct"])


def render_predict_tab() -> None:
    """Render the PDF upload, prediction, confidence, SHAP, and trading signal UI."""
    section_heading(
        "Briefing Inference",
        "Upload an FDA briefing PDF to score AdCom vote direction with the saved deployment model.",
    )
    st.info(
        "Inference uses a keyword-count proxy for the TF-IDF features because one uploaded "
        "document cannot provide real corpus-level IDF weights."
    )
    uploaded_file = st.file_uploader("Upload FDA briefing PDF", type=["pdf"])
    if uploaded_file is None:
        return
    st.markdown(
        f'<div class="upload-complete">File locked in: <strong>{uploaded_file.name}</strong></div>',
        unsafe_allow_html=True,
    )

    model, feature_names = load_model_artifacts()
    text = extract_pdf_text(uploaded_file)
    if not text:
        st.error("No extractable text was found in this PDF.")
        return

    features = extract_features(text, feature_names)
    label, prob_yes = predict_yes_probability(model, features)
    signal = trading_signal(prob_yes)

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Prediction", label.upper())
    col_b.metric("P(yes)", f"{prob_yes:.1%}")
    col_c.metric("Trading signal", signal)

    render_probability_gauge(prob_yes, label, signal)
    st.subheader("SHAP explanation")
    st.markdown('<div class="shap-frame">', unsafe_allow_html=True)
    plot_shap_waterfall(model, features)
    st.markdown("</div>", unsafe_allow_html=True)
    with st.expander("Feature row sent to model"):
        st.dataframe(features, use_container_width=True)


def render_historical_tab() -> None:
    """Render LOOCV results, historical scatter, table, and backtest summaries."""
    section_heading(
        "Historical Performance",
        "Out-of-fold predictions, correctness, and event-driven trading backtest context.",
    )
    historical = load_historical_results()
    if historical is None:
        st.error("Missing saved_model/historical.csv. Run PYTHONPATH=. python train_and_save.py first.")
        return

    accuracy = historical["correct"].mean()
    yes_count = int((historical["outcome"] == "yes").sum())
    no_count = int((historical["outcome"] == "no").sum())
    render_metric_strip(
        [
            ("LOOCV accuracy", f"{accuracy:.1%}"),
            ("Total meetings", str(len(historical))),
            ("Yes count", str(yes_count)),
            ("No count", str(no_count)),
        ]
    )

    y_numeric = historical["outcome"].map({"no": 0, "yes": 1}).astype(float)
    rng = np.random.default_rng(42)
    jitter = rng.normal(0, 0.025, size=len(historical))
    colors = historical["correct"].map({True: "#238b45", False: "#cb181d"})

    fig, ax = plt.subplots(figsize=(8, 4), facecolor="#0d1a2e")
    ax.set_facecolor("#0a0f1e")
    ax.scatter(historical["prob_yes"], y_numeric + jitter, c=colors, s=70, alpha=0.85)
    ax.axvline(0.5, color="#E6F1FB", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_yticks([0, 1], labels=["No", "Yes"])
    ax.set_xlabel("Out-of-fold P(yes)")
    ax.set_ylabel("Actual outcome")
    ax.set_title("Historical LOOCV predictions")
    ax.tick_params(colors="#E6F1FB")
    ax.xaxis.label.set_color("#E6F1FB")
    ax.yaxis.label.set_color("#E6F1FB")
    ax.title.set_color("#E6F1FB")
    for spine in ax.spines.values():
        spine.set_color("#378ADD")
        spine.set_alpha(0.45)
    ax.grid(color="#378ADD", alpha=0.12)
    st.pyplot(fig)
    plt.close(fig)

    styled = apply_correct_column_style(historical)
    st.dataframe(styled, use_container_width=True)

    st.subheader("Backtest summary")
    backtest_df = pd.DataFrame(
        [
            {"Universe": "Small-cap only", "Trades": 5, "Sharpe": 1.31, "Hit rate": "80.0%", "Total return": "190.8%", "Max drawdown": "-1.1%"},
            {"Universe": "All tradeable", "Trades": 13, "Sharpe": 0.92, "Hit rate": "46.2%", "Total return": "191.0%", "Max drawdown": "-2.1%"},
        ]
    )
    st.dataframe(backtest_df, use_container_width=True, hide_index=True)


def render_about_tab() -> None:
    """Render project context, methodology, stack, and limitations."""
    section_heading(
        "Research Context",
        "A compact methodology card for the model, explainability layer, and trading validation.",
    )
    st.write(
        "This app predicts FDA Oncologic Drugs Advisory Committee vote outcomes from "
        "briefing PDFs using clinical/NLP features and a Random Forest trained on "
        "historical public meetings."
    )
    st.table(pd.DataFrame(
        [
            {"Component": "Validation", "Method": "Leave-One-Out Cross-Validation on 25 labeled meetings"},
            {"Component": "Model", "Method": "Random Forest with 100 estimators"},
            {"Component": "Explanations", "Method": "SHAP TreeExplainer waterfall for each uploaded PDF"},
            {"Component": "Backtest", "Method": "Walk-forward event study with XBI benchmark adjustment"},
        ]
    ))
    st.write("Tech stack: Streamlit, pandas, scikit-learn, SHAP, pdfplumber, matplotlib, joblib.")
    st.warning(
        "Limitations: the dataset is tiny, the small-cap backtest has only n=5 trades, "
        "PDF text extraction can be noisy, and single-document inference approximates TF-IDF. "
        "Results are indicative research signals, not statistical proof or financial advice."
    )


def main() -> None:
    """Create the three-tab Streamlit interface for deployment."""
    inject_custom_css()
    render_hero()
    tab_predict, tab_history, tab_about = st.tabs(
        ["🧬 Predict New Briefing", "📈 Historical Results", "⚗️ About"]
    )
    with tab_predict:
        render_predict_tab()
    with tab_history:
        render_historical_tab()
    with tab_about:
        render_about_tab()


if __name__ == "__main__":
    main()
