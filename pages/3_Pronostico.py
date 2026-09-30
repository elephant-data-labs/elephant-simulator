from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from risk_ui.brand import footer, setup_page

setup_page("Pronóstico exploratorio", "Genera trayectorias futuras remuestreando los cambios de una serie histórica.")
st.write(
    "Carga una serie ordenada en el tiempo. El método remuestrea sus diferencias o variaciones "
    "porcentuales para crear muchos caminos posibles. No detecta automáticamente estacionalidad "
    "ni cambios estructurales."
)

uploaded = st.file_uploader("Archivo CSV con historia", type=["csv"])
if uploaded is not None:
    try:
        frame = pd.read_csv(uploaded)
        numeric = frame.select_dtypes(include="number").columns.tolist()
        if not numeric:
            st.warning("El archivo no contiene columnas numéricas.")
        else:
            selected = st.selectbox("Serie a pronosticar", numeric)
            series = frame[selected].dropna().to_numpy(dtype=float)
            if len(series) < 12:
                st.warning("Se recomiendan al menos 12 observaciones.")
            elif not np.isfinite(series).all():
                st.warning("La serie tiene valores no finitos; límpiala antes de continuar.")
            else:
                a, b, c = st.columns(3)
                horizon = a.slider("Períodos futuros", 1, 36, 12)
                paths_n = b.select_slider("Trayectorias", options=[500, 1_000, 2_500, 5_000], value=1_000)
                method = c.selectbox("Cambio remuestreado", ["Diferencia absoluta", "Variación porcentual"])

                if method == "Diferencia absoluta":
                    increments = np.diff(series)
                    paths = series[-1] + np.cumsum(
                        np.random.default_rng(42).choice(increments, size=(paths_n, horizon), replace=True), axis=1
                    )
                elif np.any(series[:-1] <= 0):
                    st.error("La variación porcentual requiere valores históricos positivos; prueba diferencias absolutas.")
                    paths = None
                else:
                    changes = series[1:] / series[:-1] - 1
                    sampled = np.random.default_rng(42).choice(changes, size=(paths_n, horizon), replace=True)
                    paths = series[-1] * np.cumprod(1 + sampled, axis=1)

                if paths is not None:
                    p05, median, p95 = np.percentile(paths, [5, 50, 95], axis=0)
                    history_x = np.arange(len(series))
                    future_x = np.arange(len(series) - 1, len(series) + horizon)
                    chart = go.Figure()
                    chart.add_trace(go.Scatter(x=history_x, y=series, mode="lines", name="Historia", line=dict(color="#0b3d62", width=2)))
                    chart.add_trace(go.Scatter(x=future_x, y=np.r_[series[-1], p95], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
                    chart.add_trace(go.Scatter(x=future_x, y=np.r_[series[-1], p05], mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(21,122,138,.18)", name="P5–P95"))
                    chart.add_trace(go.Scatter(x=future_x, y=np.r_[series[-1], median], mode="lines", name="Mediana simulada", line=dict(color="#157a8a", dash="dash")))
                    chart.update_layout(title="Trayectorias por remuestreo", xaxis_title="Períodos en la serie", yaxis_title=selected, height=430)
                    st.plotly_chart(chart, width="stretch")
                    st.caption("El rango es exploratorio: refleja el historial cargado y el método elegido, no una predicción garantizada.")
    except Exception as error:
        st.error(f"No se pudo leer ese CSV: {error}")

footer()
