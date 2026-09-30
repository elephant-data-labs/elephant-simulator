from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from risk_lab.engine import Assumption, ModelError, run_simulation, scenario_table, sensitivity_table
from risk_ui.brand import footer, setup_page
from risk_ui.format import decimal, percent

setup_page("Modelo y simulación", "Define supuestos inciertos y observa la distribución de resultados posibles.")


def initialize() -> None:
    if "model_inputs" not in st.session_state:
        st.session_state.model_inputs = [
            {"id": "ventas", "name": "ventas", "distribution": "Uniforme", "p1": 90.0, "p2": 110.0, "p3": 0.0},
            {"id": "margen", "name": "margen", "distribution": "Triangular", "p1": 0.18, "p2": 0.22, "p3": 0.27},
            {"id": "costos", "name": "costos", "distribution": "Normal", "p1": 70.0, "p2": 5.0, "p3": 0.0},
        ]
    for idx, row in enumerate(st.session_state.model_inputs):
        row.setdefault("id", f"supuesto_{idx}_{row.get('name', 'variable')}")
    if "model_formula" not in st.session_state:
        st.session_state.model_formula = "ventas * margen - costos"
    if "simulation_result" not in st.session_state:
        st.session_state.simulation_result = None


def assumptions_from_state() -> list[Assumption]:
    return [
        Assumption(
            name=row["name"].strip(),
            distribution=row["distribution"],
            p1=float(row["p1"]),
            p2=float(row["p2"]),
            p3=float(row["p3"]),
        )
        for row in st.session_state.model_inputs
    ]


initialize()
st.markdown("#### 1. Supuestos inciertos")
st.caption("Agrega las variables que explican tu resultado y asigna una distribución a cada una.")

for idx, row in enumerate(list(st.session_state.model_inputs)):
    key_id = row["id"]
    with st.expander(f"{row['name']} · {row['distribution']}", expanded=idx == 0):
        c1, c2 = st.columns([1, 1])
        row["name"] = c1.text_input("Nombre de variable", value=row["name"], key=f"name_{key_id}")
        distributions = ["Normal", "Uniforme", "Triangular", "PERT", "Lognormal"]
        row["distribution"] = c2.selectbox(
            "Distribución", distributions, index=distributions.index(row["distribution"]), key=f"dist_{key_id}"
        )
        dist = row["distribution"]
        if dist == "Normal":
            a, b = st.columns(2)
            row["p1"] = a.number_input("Media", value=float(row["p1"]), key=f"p1_{key_id}")
            row["p2"] = b.number_input("Desviación estándar", min_value=0.000001, value=max(float(row["p2"]), 0.000001), key=f"p2_{key_id}")
            st.caption("La normal no tiene límites fijos. Si la variable no puede ser negativa, evalúa otra distribución.")
        elif dist == "Lognormal":
            a, b = st.columns(2)
            row["p1"] = a.number_input("Media del logaritmo", value=float(row["p1"]), key=f"p1_{key_id}")
            row["p2"] = b.number_input("Desviación del logaritmo", min_value=0.000001, value=max(float(row["p2"]), 0.000001), key=f"p2_{key_id}")
            st.caption("La lognormal produce valores positivos; sus parámetros están en escala logarítmica.")
        elif dist == "Uniforme":
            a, b = st.columns(2)
            row["p1"] = a.number_input("Mínimo", value=float(row["p1"]), key=f"p1_{key_id}")
            row["p2"] = b.number_input("Máximo", value=float(row["p2"]), key=f"p2_{key_id}")
        else:
            a, b, c = st.columns(3)
            row["p1"] = a.number_input("Mínimo", value=float(row["p1"]), key=f"p1_{key_id}")
            row["p2"] = b.number_input("Más probable", value=float(row["p2"]), key=f"p2_{key_id}")
            row["p3"] = c.number_input("Máximo", value=float(row["p3"]), key=f"p3_{key_id}")
        if st.button("Eliminar variable", key=f"remove_{key_id}", disabled=len(st.session_state.model_inputs) <= 1):
            st.session_state.model_inputs.pop(idx)
            st.session_state.simulation_result = None
            st.rerun()
    st.session_state.model_inputs[idx] = row

with st.form("add_assumption"):
    st.markdown("**Agregar supuesto**")
    a, b = st.columns(2)
    new_name = a.text_input("Nombre", value=f"variable_{len(st.session_state.model_inputs) + 1}")
    new_dist = b.selectbox("Distribución inicial", ["Normal", "Uniforme", "Triangular", "PERT", "Lognormal"])
    add_clicked = st.form_submit_button("Agregar variable")
if add_clicked:
    defaults = {
        "Normal": (10.0, 2.0, 0.0), "Uniforme": (0.0, 20.0, 0.0),
        "Triangular": (0.0, 10.0, 20.0), "PERT": (0.0, 10.0, 20.0),
        "Lognormal": (2.3, 0.2, 0.0),
    }
    p1, p2, p3 = defaults[new_dist]
    st.session_state.model_inputs.append({
        "id": f"nuevo_{len(st.session_state.model_inputs)}_{new_name}",
        "name": new_name, "distribution": new_dist, "p1": p1, "p2": p2, "p3": p3,
    })
    st.session_state.simulation_result = None
    st.rerun()

st.markdown("#### 2. Resultado que quieres analizar")
st.caption("Escribe una expresión con tus variables, operaciones aritméticas y funciones matemáticas básicas.")
formula = st.text_input("Fórmula", key="model_formula", help="Ejemplo: ventas * margen - costos")
m1, m2, m3 = st.columns(3)
iterations = m1.select_slider("Iteraciones", options=[1_000, 5_000, 10_000, 20_000, 50_000, 100_000], value=20_000)
seed = m2.number_input("Semilla reproducible", min_value=0, max_value=2_147_483_647, value=42, step=1)
output_unit = m3.text_input("Unidad del resultado", value="unidades")

assumptions = assumptions_from_state()
names = [item.name for item in assumptions]
corr = None
if len(set(names)) == len(names) and all(name.isidentifier() for name in names):
    st.markdown("#### 3. Dependencia entre supuestos")
    st.caption("Deja 0 si no tienes una relación que justificar. La matriz debe ser simétrica y compatible. La cópula gaussiana guía la dependencia; con distribuciones no normales no garantiza exactamente la correlación final indicada.")
    corr_key = "corr_" + "_".join(names)
    if st.session_state.get("corr_key") != corr_key:
        st.session_state.corr_key = corr_key
        st.session_state[corr_key] = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    corr = st.data_editor(
        st.session_state[corr_key], key=f"editor_{corr_key}", num_rows="fixed",
        hide_index=False, width="stretch",
        column_config={name: st.column_config.NumberColumn(name, min_value=-1.0, max_value=1.0, step=0.05, format="%.2f") for name in names},
    )
    st.session_state[corr_key] = corr
else:
    st.warning("Corrige los nombres: usa identificadores como ventas, costo_fijo o volumen2.")

a, b = st.columns([1, 2])
threshold = a.number_input("Umbral para la probabilidad", value=0.0)
run_clicked = b.button("Ejecutar simulación", type="primary", width="stretch")
if run_clicked:
    try:
        st.session_state.simulation_result = run_simulation(
            assumptions=assumptions, formula=formula, iterations=int(iterations), seed=int(seed), correlation=corr
        )
        st.session_state.simulation_meta = {
            "iterations": int(iterations), "seed": int(seed), "formula": formula,
            "threshold": float(threshold), "unit": output_unit,
        }
    except (ModelError, ValueError, KeyError) as exc:
        st.error(str(exc))

if st.session_state.simulation_result is not None:
    draws, outcome = st.session_state.simulation_result
    meta = st.session_state.simulation_meta
    st.divider()
    st.markdown("#### Resultados")
    columns = st.columns(5)
    columns[0].metric("Media", decimal(np.mean(outcome)), meta["unit"])
    columns[1].metric("Mediana", decimal(np.median(outcome)), meta["unit"])
    columns[2].metric("Percentil 5", decimal(np.percentile(outcome, 5)), meta["unit"])
    columns[3].metric("Percentil 95", decimal(np.percentile(outcome, 95)), meta["unit"])
    columns[4].metric(f"Probabilidad > {decimal(meta['threshold'])}", percent(np.mean(outcome > meta["threshold"])))

    chart_col, risk_col = st.columns([1.5, 1])
    fig = go.Figure(go.Histogram(x=outcome, nbinsx=70, marker_color="#157a8a", name="Resultados"))
    fig.add_vline(x=np.median(outcome), line_dash="dash", line_color="#17364a", annotation_text="Mediana")
    fig.add_vline(x=meta["threshold"], line_dash="dot", line_color="#bb4c3f", annotation_text="Umbral")
    fig.update_layout(title="Distribución del resultado", xaxis_title=meta["unit"], yaxis_title="Frecuencia", height=380, margin=dict(t=50, b=10))
    chart_col.plotly_chart(fig, width="stretch")
    lower_tail = outcome[outcome <= np.percentile(outcome, 5)]
    risk_col.markdown("**Lectura de los resultados**")
    risk_col.write(f"El 90% central cae entre **{decimal(np.percentile(outcome, 5), 3)}** y **{decimal(np.percentile(outcome, 95), 3)} {meta['unit']}**.")
    risk_col.write(f"El promedio del 5% de resultados más bajos es **{decimal(np.mean(lower_tail), 3)} {meta['unit']}**.")
    risk_col.caption("Son medidas condicionadas al modelo y a los supuestos ingresados.")

    left, right = st.columns(2)
    sensitivity = sensitivity_table(draws, outcome)
    sens_fig = go.Figure(go.Bar(
        x=sensitivity["Correlación de rangos"], y=sensitivity["Variable"], orientation="h",
        marker_color=["#bb4c3f" if x < 0 else "#157a8a" for x in sensitivity["Correlación de rangos"]],
    ))
    sens_fig.update_layout(title="Sensibilidad de los supuestos", xaxis_title="Correlación de rangos", height=360, margin=dict(t=50, b=10))
    left.plotly_chart(sens_fig, width="stretch")
    right.markdown("**Escenarios representativos**")
    scenarios = scenario_table(assumptions, formula, draws, outcome)
    scenario_formats = {column: (lambda value: decimal(value, 3)) for column in scenarios.columns if column != "Escenario"}
    right.dataframe(scenarios.style.format(scenario_formats), hide_index=True, width="stretch")
    right.caption("Cada fila muestra el sorteo real más cercano al percentil del resultado.")

    model_json = {
        "formula": formula, "iterations": int(iterations), "seed": int(seed),
        "assumptions": [item.__dict__ for item in assumptions],
        "correlation": corr.to_dict(orient="split") if corr is not None else None,
    }
    st.download_button("Descargar modelo (JSON)", json.dumps(model_json, indent=2), "modelo_incertidumbre.json", "application/json")
    result_csv = pd.DataFrame({**draws, "resultado": outcome}).to_csv(index=False).encode("utf-8")
    st.download_button("Descargar iteraciones (CSV)", result_csv, "resultados_simulacion.csv", "text/csv")

footer()
