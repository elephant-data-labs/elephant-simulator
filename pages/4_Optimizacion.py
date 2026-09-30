from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from risk_lab.engine import ModelError, evaluate_formula, formula_names
from risk_ui.brand import footer, setup_page
from risk_ui.format import decimal

setup_page("Optimización", "Compara decisiones usando los mismos sorteos de incertidumbre.")

if st.session_state.get("simulation_result") is None:
    st.info("Primero configura y ejecuta una simulación en Modelo y simulación.")
else:
    draws, _ = st.session_state.simulation_result
    formula = st.session_state.simulation_meta["formula"]
    st.write("Define una variable controlable que aparezca en la fórmula del modelo y una grilla de alternativas.")
    decision, minimum, maximum, steps, direction = st.columns(5)
    decision_name = decision.text_input("Variable de decisión", value="decision")
    low = minimum.number_input("Mínimo", value=0.0)
    high = maximum.number_input("Máximo", value=100.0)
    grid_size = steps.slider("Alternativas", min_value=5, max_value=100, value=31)
    objective = direction.selectbox("Objetivo", ["Maximizar", "Minimizar"])
    constraint = st.text_input("Restricción opcional", value="", help="Ejemplo: costos < 50")

    if st.button("Comparar decisiones", type="primary"):
        known = set(draws)
        if not decision_name.isidentifier() or decision_name in known:
            st.error("El nombre debe ser válido y distinto de los supuestos inciertos.")
        elif high <= low:
            st.error("El máximo debe ser mayor que el mínimo.")
        elif decision_name not in formula_names(formula):
            st.error(f"Incluye «{decision_name}» en la fórmula para optimizarla.")
        elif constraint and formula_names(constraint) - (known | {decision_name}):
            st.error("La restricción incluye variables que no existen en el modelo.")
        else:
            count = len(next(iter(draws.values())))
            candidates = np.linspace(low, high, grid_size)
            rows = []
            try:
                for value in candidates:
                    environment = {**draws, decision_name: np.full(count, value)}
                    outcomes = evaluate_formula(formula, environment)
                    valid = np.ones(count, dtype=bool)
                    if constraint:
                        valid = evaluate_formula(constraint, environment).astype(bool)
                    usable = outcomes[valid]
                    rows.append({
                        "Decisión": value,
                        "Resultado esperado": float(np.mean(usable)) if len(usable) else np.nan,
                        "Casos válidos": int(valid.sum()),
                    })
                table = pd.DataFrame(rows).dropna()
                if table.empty:
                    st.warning("Ninguna alternativa satisface la restricción.")
                else:
                    best_idx = table["Resultado esperado"].idxmax() if objective == "Maximizar" else table["Resultado esperado"].idxmin()
                    best = table.loc[best_idx]
                    st.success(f"Mejor alternativa de esta grilla: **{decision_name} = {decimal(best['Decisión'], 4)}**, resultado esperado **{decimal(best['Resultado esperado'], 4)}**.")
                    chart = go.Figure(go.Scatter(x=table["Decisión"], y=table["Resultado esperado"], mode="lines+markers", name="Resultado esperado"))
                    chart.add_vline(x=best["Decisión"], line_dash="dash", line_color="#bb4c3f", annotation_text="Mejor de la grilla")
                    chart.update_layout(title="Resultado esperado por decisión", xaxis_title=decision_name, yaxis_title="Resultado esperado", height=400)
                    st.plotly_chart(chart, width="stretch")
                    st.dataframe(table.style.format({
                        "Decisión": lambda value: decimal(value, 3),
                        "Resultado esperado": lambda value: decimal(value, 3),
                        "Casos válidos": lambda value: f"{int(value):,}".replace(",", "."),
                    }), hide_index=True, width="stretch")
                    st.caption("Compara estas alternativas dentro del modelo actual; no garantiza un óptimo global.")
            except (ModelError, ValueError, KeyError, ZeroDivisionError) as error:
                st.error(str(error))

footer()
