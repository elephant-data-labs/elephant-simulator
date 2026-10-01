from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
LOGO_PATH = ROOT / "Elephant.png"
APP_VERSION = "Elephant Simulator · V1 educativa"

TEAL = "#157a8a"
INK = "#0b3d62"
MUTED = "#4a7186"
LINE = "#d9e7ee"
SURFACE = "#ffffff"
CANVAS = "#f4f8fa"

_STYLES = f"""
<style>
.stApp {{ background: {CANVAS}; }}
section.main .block-container {{ padding-top: 2.2rem; max-width: 1320px; }}
.edl-header {{
    background: linear-gradient(100deg, {SURFACE} 0%, #eef6f8 100%);
    border: 1px solid {LINE}; border-left: 5px solid {TEAL};
    border-radius: 14px; padding: 1.35rem 1.75rem 1.45rem;
    margin-bottom: 1.5rem;
}}
.edl-header h1 {{ color: {INK}; font-size: 2.3rem; font-weight: 800;
    margin: 0; letter-spacing: -0.025em; line-height: 1.1; }}
.edl-header .sub {{ color: {MUTED}; font-size: 1.02rem;
    margin-top: 0.35rem; line-height: 1.35; max-width: 68ch; }}
[data-testid="stSidebar"] {{ background: {SURFACE}; border-right: 1px solid {LINE}; }}
.edl-side {{ text-align: center; padding: 0.2rem 0 0.4rem; opacity: 0.92; }}
.edl-side img {{ max-width: 72px; height: auto; margin-bottom: 0.3rem; }}
.edl-side .lab {{ color: {TEAL}; font-size: 0.68rem; margin-top: 0.15rem;
    text-transform: uppercase; letter-spacing: 0.09em; font-weight: 700; }}
.edl-modulo {{ background: {SURFACE}; border: 1px solid {LINE};
    border-left: 4px solid {LINE}; border-radius: 12px; padding: 0.9rem 1.1rem;
    min-height: 8.5rem; height: 100%; }}
.edl-modulo.activo {{ border-left-color: {TEAL}; }}
.edl-modulo .titulo {{ color: {INK}; font-weight: 700; font-size: 1rem; }}
.edl-modulo .estado {{ color: {TEAL}; font-size: 0.68rem; font-weight: 700;
    letter-spacing: 0.07em; text-transform: uppercase; margin-top: 0.25rem; }}
.edl-modulo .texto {{ color: {MUTED}; font-size: 0.84rem; margin-top: 0.45rem; line-height: 1.45; }}
[data-testid="stHorizontalBlock"]:has(.edl-modulo) {{ align-items: stretch; }}
[data-testid="stHorizontalBlock"]:has(.edl-modulo) [data-testid="stColumn"] > div {{
    height: 100%; display: flex; flex-direction: column; }}
[data-testid="stMetric"] {{ background: {SURFACE}; border: 1px solid {LINE};
    border-radius: 10px; padding: 0.7rem 0.9rem; }}
[data-testid="stMetricValue"] {{ white-space: normal; overflow: visible;
    text-overflow: unset; line-height: 1.2; font-size: 1.3rem; color: {INK}; }}
[data-testid="stMetricLabel"] {{ color: {MUTED}; }}
.edl-firma {{ border-left: 3px solid {TEAL}; background: {SURFACE};
    border-radius: 0 10px 10px 0; padding: 0.85rem 1.1rem; margin: 0.4rem 0 0.8rem; }}
.edl-firma .texto {{ color: #23404f; line-height: 1.6; }}
.edl-firma .pie {{ color: {MUTED}; font-size: 0.78rem; margin-top: 0.6rem; }}
</style>
"""


def page_config() -> None:
    st.set_page_config(
        page_title="Elephant Simulator · Elephant Data Labs",
        page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def logo_html() -> str:
    if not LOGO_PATH.exists():
        return ""
    encoded = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
    return f'<img src="data:image/png;base64,{encoded}" alt="Elephant Data Labs" />'


def module_card(title: str, text: str) -> None:
    st.markdown(
        f'<div class="edl-modulo activo"><div class="titulo">{title}</div>'
        f'<div class="estado">Disponible</div><div class="texto">{text}</div></div>',
        unsafe_allow_html=True,
    )


def setup_page(title: str, subtitle: str = "") -> None:
    st.markdown(_STYLES, unsafe_allow_html=True)
    sub = f'<div class="sub">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f'<div class="edl-header"><h1>{title}</h1>{sub}</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.divider()
    st.sidebar.markdown(
        f'<div class="edl-side">{logo_html()}<div class="lab">Elephant Data Labs</div></div>',
        unsafe_allow_html=True,
    )
    st.sidebar.caption(APP_VERSION)


def footer() -> None:
    st.divider()
    st.caption("Elephant Data Labs · Herramienta educativa para explorar incertidumbre")
