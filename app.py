"""Enrutador principal: declara páginas y grupos de navegación."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from risk_ui.brand import page_config  # noqa: E402

page_config()

inicio = st.Page("pages/0_Inicio.py", title="Inicio", icon=":material/home:", default=True)

modelacion = [
    st.Page("pages/1_Simulacion.py", title="Modelo y simulación", icon=":material/monitoring:"),
    st.Page("pages/2_Ajuste_de_distribuciones.py", title="Ajuste de distribuciones", icon=":material/query_stats:"),
    st.Page("pages/3_Pronostico.py", title="Pronóstico", icon=":material/timeline:"),
]

decisiones = [
    st.Page("pages/4_Optimizacion.py", title="Optimización", icon=":material/target:"),
]

referencia = [
    st.Page("pages/5_Metodologia.py", title="Metodología", icon=":material/menu_book:"),
]

st.navigation(
    {
        "Inicio": [inicio],
        "Modelación": modelacion,
        "Decisiones": decisiones,
        "Referencia": referencia,
    }
).run()
