from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from risk_ui.brand import footer, setup_page
from risk_ui.format import decimal

setup_page("Ajuste de distribuciones", "Compara modelos probabilísticos candidatos con una muestra de datos.")
st.markdown(
    "El ajuste resume los datos disponibles y puede orientar la elección de una distribución. "
    "No demuestra que esa distribución describa observaciones futuras."
)

uploaded = st.file_uploader("Archivo CSV con observaciones", type=["csv"])
if uploaded is not None:
    try:
        frame = pd.read_csv(uploaded)
        numeric = frame.select_dtypes(include="number").columns.tolist()
        if not numeric:
            st.warning("El archivo no contiene columnas numéricas.")
        else:
            selected = st.selectbox("Columna a analizar", numeric)
            sample = frame[selected].dropna().to_numpy(dtype=float)
            sample = sample[np.isfinite(sample)]
            if len(sample) < 10:
                st.warning("Se necesitan al menos 10 observaciones válidas para comparar los ajustes.")
            else:
                candidates = []
                fitted = []
                mu, sigma = stats.norm.fit(sample)
                candidates.append(("Normal", stats.norm.logpdf(sample, mu, sigma)))
                fitted.append(("Normal", (mu, sigma), lambda x, m=mu, s=sigma: stats.norm.pdf(x, m, s), 2))
                low, high = float(sample.min()), float(sample.max())
                if low > 0:
                    shape, _, scale = stats.lognorm.fit(sample, floc=0)
                    candidates.append(("Lognormal", stats.lognorm.logpdf(sample, shape, loc=0, scale=scale)))
                    fitted.append(("Lognormal", (shape, scale), lambda x, sh=shape, sc=scale: stats.lognorm.pdf(x, sh, scale=sc), 2))
                if high > low:
                    candidates.append(("Uniforme", stats.uniform.logpdf(sample, loc=low, scale=high - low)))
                    fitted.append(("Uniforme", (low, high), lambda x, lo=low, sc=high-low: stats.uniform.pdf(x, loc=lo, scale=sc), 2))
                    mode = float(np.clip(np.median(sample), low + 1e-9, high - 1e-9))
                    shape = (mode - low) / (high - low)
                    candidates.append(("Triangular", stats.triang.logpdf(sample, shape, loc=low, scale=high - low)))
                    fitted.append(("Triangular", (low, mode, high), lambda x, c=shape, lo=low, sc=high-low: stats.triang.pdf(x, c, loc=lo, scale=sc), 3))

                rows = []
                for name, logpdf in candidates:
                    fit = next(row for row in fitted if row[0] == name)
                    likelihood = float(np.sum(logpdf))
                    rows.append({
                        "Distribución": name,
                        "Parámetros estimados": ", ".join(f"{value:.5g}" for value in fit[1]),
                        "Log-verosimilitud": likelihood,
                        "AIC (menor es mejor)": 2 * fit[3] - 2 * likelihood,
                    })
                comparison = pd.DataFrame(rows).sort_values("AIC (menor es mejor)")
                numeric_formats = {
                    "Log-verosimilitud": decimal,
                    "AIC (menor es mejor)": decimal,
                }
                st.dataframe(comparison.style.format(numeric_formats), hide_index=True, width="stretch")

                x = np.linspace(low, high, 400)
                chart = go.Figure(go.Histogram(x=sample, histnorm="probability density", nbinsx=35, opacity=0.45, name="Observaciones"))
                for name, _, density, _ in fitted:
                    chart.add_trace(go.Scatter(x=x, y=density(x), mode="lines", name=name))
                chart.update_layout(title="Muestra y ajustes candidatos", xaxis_title=selected, yaxis_title="Densidad", height=420)
                st.plotly_chart(chart, width="stretch")

                best = str(comparison.iloc[0]["Distribución"])
                if st.button("Usar el mejor ajuste como supuesto de un modelo"):
                    fit = next(row for row in fitted if row[0] == best)
                    if best == "Normal":
                        distribution, p1, p2, p3 = "Normal", fit[1][0], fit[1][1], 0.0
                    elif best == "Lognormal":
                        distribution, p1, p2, p3 = "Lognormal", float(np.log(fit[1][1])), fit[1][0], 0.0
                    elif best == "Uniforme":
                        distribution, p1, p2, p3 = "Uniforme", fit[1][0], fit[1][1], 0.0
                    else:
                        distribution, p1, p2, p3 = "Triangular", fit[1][0], fit[1][1], fit[1][2]
                    if "model_inputs" not in st.session_state:
                        st.session_state.model_inputs = []
                    base = "".join(char.lower() if char.isalnum() else "_" for char in str(selected)).strip("_") or "dato"
                    if base[0].isdigit():
                        base = "x_" + base
                    used = {row["name"] for row in st.session_state.model_inputs}
                    name = base
                    suffix = 2
                    while name in used:
                        name = f"{base}_{suffix}"
                        suffix += 1
                    st.session_state.model_inputs.append({
                        "id": f"ajuste_{name}", "name": name, "distribution": distribution,
                        "p1": float(p1), "p2": float(p2), "p3": float(p3),
                    })
                    st.success(f"Se agregó «{name}». Conecta la variable a una fórmula desde Modelo y simulación.")
                st.caption("El AIC compara estos candidatos con esta muestra; el menor AIC no prueba que esa distribución sea la verdadera.")
    except Exception as error:
        st.error(f"No se pudo leer ese CSV: {error}")

footer()
