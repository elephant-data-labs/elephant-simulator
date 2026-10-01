from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from risk_lab.engine import (
    Assumption, ModelError, convergence_summary, filter_draws, run_simulation,
    scenario_table, sensitivity_table, sobol_sensitivity,
)
from risk_ui.brand import footer, setup_page
from risk_ui.format import decimal, percent

setup_page("Modelo y simulación", "Define supuestos inciertos y observa la distribución de resultados posibles.")


def initialize() -> None:
    if "model_inputs" not in st.session_state:
        st.session_state.model_inputs = [
            {"id": "variable_1", "name": "variable_1", "distribution": "Uniforme", "p1": 0.0, "p2": 1.0, "p3": 0.0},
        ]
    for idx, row in enumerate(st.session_state.model_inputs):
        row.setdefault("id", f"supuesto_{idx}_{row.get('name', 'variable')}")
    if "model_formula" not in st.session_state:
        st.session_state.model_formula = ""
    if (
        [row.get("name") for row in st.session_state.model_inputs] == ["ventas", "margen", "costos"]
        and st.session_state.model_formula == "ventas * margen - costos"
    ):
        st.session_state.model_inputs = [
            {"id": "variable_1", "name": "variable_1", "distribution": "Uniforme", "p1": 0.0, "p2": 1.0, "p3": 0.0}
        ]
        st.session_state.model_formula = ""
        st.session_state.simulation_result = None
        st.session_state.simulation_comparison = None
        st.session_state.simulation_convergence = None
        st.session_state.simulation_sobol = None
        st.session_state.pop("simulation_meta", None)
        st.session_state.pop("simulation_signature", None)
    if "simulation_result" not in st.session_state:
        st.session_state.simulation_result = None
    if "simulation_comparison" not in st.session_state:
        st.session_state.simulation_comparison = None
    if "simulation_convergence" not in st.session_state:
        st.session_state.simulation_convergence = None
    if "simulation_sobol" not in st.session_state:
        st.session_state.simulation_sobol = None


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


def load_fictional_dcf() -> None:
    years = [
        ("fcf_1", 95.0, 110.0, 125.0),
        ("fcf_2", 103.0, 119.0, 136.0),
        ("fcf_3", 112.0, 129.0, 147.0),
        ("fcf_4", 121.0, 140.0, 160.0),
        ("fcf_5", 132.0, 152.0, 174.0),
    ]
    inputs = [
        {"id": name, "name": name, "distribution": "PERT", "p1": low, "p2": mode, "p3": high}
        for name, low, mode, high in years
    ]
    inputs.extend([
        {"id": "wacc", "name": "wacc", "distribution": "Triangular", "p1": 0.08, "p2": 0.10, "p3": 0.13},
        {"id": "g", "name": "g", "distribution": "Triangular", "p1": 0.015, "p2": 0.025, "p3": 0.04},
    ])
    terms = [f"fcf_{year}/(1+wacc)**{year}" for year in range(1, 6)]
    terms.append("(fcf_5*(1+g)/(wacc-g))/(1+wacc)**5")
    st.session_state.model_inputs = inputs
    st.session_state.model_formula = "+".join(terms)
    st.session_state.use_dcf_rule = True
    names = [row["name"] for row in inputs]
    corr_key = "corr_" + "_".join(names)
    matrix = np.eye(len(names))
    matrix[:5, :5] = 0.45
    np.fill_diagonal(matrix, 1.0)
    st.session_state.corr_key = corr_key
    st.session_state[corr_key] = pd.DataFrame(matrix, index=names, columns=names)
    st.session_state.simulation_result = None
    st.session_state.simulation_comparison = None
    st.session_state.simulation_convergence = None
    st.session_state.simulation_sobol = None


def append_formula_token(token: str, is_operator: bool = False) -> None:
    current = st.session_state.get("model_formula", "").rstrip()
    if is_operator:
        if token == ")":
            updated = current + ")"
        elif token == "(":
            updated = f"{current} (" if current else "("
        else:
            updated = f"{current} {token} " if current else token
    else:
        updated = f"{current} {token}" if current else token
    st.session_state.model_formula = updated


initialize()
example_col, note_col = st.columns([1, 2])
if example_col.button("Cargar caso DCF de empresa ficticia"):
    load_fictional_dcf()
    st.rerun()
note_col.caption("El caso es exclusivamente educativo: calcula valor empresa a partir de flujos y supuestos inventados, sin precio por acción ni comparación con mercado.")
if st.button("Nuevo modelo vacío"):
    st.session_state.model_inputs = [
        {"id": "variable_1", "name": "variable_1", "distribution": "Uniforme", "p1": 0.0, "p2": 1.0, "p3": 0.0}
    ]
    st.session_state.model_formula = ""
    st.session_state.use_dcf_rule = False
    st.session_state.corr_key = None
    st.session_state.simulation_result = None
    st.session_state.simulation_comparison = None
    st.session_state.simulation_convergence = None
    st.session_state.simulation_sobol = None
    st.rerun()
st.markdown("#### 1. Supuestos inciertos")
st.caption("Agrega las variables que explican tu resultado y asigna una distribución a cada una.")

original_names = {row["id"]: row["name"] for row in st.session_state.model_inputs}
original_corr_key = st.session_state.get("corr_key")
original_corr = st.session_state.get(original_corr_key) if original_corr_key else None
for idx, row in enumerate(list(st.session_state.model_inputs)):
    key_id = row["id"]
    with st.expander(f"{row['name']} · {row['distribution']}", expanded=idx == 0):
        c1, c2 = st.columns([1, 1])
        old_name = row["name"]
        row["name"] = c1.text_input("Nombre de variable", value=row["name"], key=f"name_{key_id}")
        if row["name"] != old_name:
            if old_name.isidentifier() and row["name"].isidentifier():
                st.session_state.model_formula = re.sub(
                    rf"\b{re.escape(old_name)}\b", row["name"], st.session_state.model_formula
                )
            rename_map = {
                original_names[item["id"]]: item["name"]
                for item in st.session_state.model_inputs
                if item["id"] in original_names
            }
            current_names = [item["name"] for item in st.session_state.model_inputs]
            next_corr_key = "corr_" + "_".join(current_names)
            if original_corr is not None:
                renamed_corr = original_corr.rename(index=rename_map, columns=rename_map)
                st.session_state[next_corr_key] = renamed_corr.reindex(
                    index=current_names, columns=current_names, fill_value=0.0
                )
            st.session_state.corr_key = next_corr_key
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

st.markdown("#### 2. Construir la fórmula")
st.caption("Parte de una fórmula vacía. Haz clic en las variables para insertarlas, y usa los menús para agregar funciones y operadores, como en el editor de Power Query. Si renombras una variable, se actualiza también en la fórmula.")
assumptions = assumptions_from_state()
valid_names = [item.name for item in assumptions if item.name.isidentifier() and item.name not in {"abs", "sqrt", "exp", "log", "log10", "min", "max"}]
if valid_names:
    st.markdown("**Variables disponibles — selecciona para insertar**")
    variable_columns = st.columns(min(len(valid_names), 5))
    for idx, variable_name in enumerate(valid_names):
        if variable_columns[idx % len(variable_columns)].button(
            f"＋ {variable_name}", key=f"insert_variable_{idx}_{variable_name}", width="stretch"
        ):
            append_formula_token(variable_name)
            st.rerun()
    function_col, operator_col = st.columns([1.5, 1.2])
    chosen_function = function_col.selectbox("Función", ["log", "log10", "sqrt", "exp", "abs"], key="builder_function")
    function_variable = function_col.selectbox("Aplicar a", valid_names, key="builder_function_variable")
    if function_col.button("Insertar función", width="stretch"):
        append_formula_token(f"{chosen_function}({function_variable})")
        st.rerun()
    chosen_operator = operator_col.selectbox("Operador o paréntesis", ["+", "-", "*", "/", "**", "(", ")"], key="builder_operator")
    if operator_col.button("Insertar operador", width="stretch"):
        append_formula_token(chosen_operator, is_operator=True)
        st.rerun()
else:
    st.warning("Agrega una variable con nombre válido para usar el constructor de fórmulas.")
formula = st.text_input("Fórmula resultante", key="model_formula", help="Ejemplos: log(ventas), ventas * margen - costos, exp(tasa)")
m1, m2, m3 = st.columns(3)
iterations = m1.select_slider("Iteraciones", options=[1_000, 5_000, 10_000, 20_000, 50_000, 100_000], value=20_000)
seed = m2.number_input("Semilla reproducible", min_value=0, max_value=2_147_483_647, value=42, step=1)
output_unit = m3.text_input("Unidad del resultado", value="unidades")
sampling_method = st.selectbox("Método de muestreo", ["Monte Carlo", "Latin Hypercube"])
compare_lhs = st.checkbox("Comparar Monte Carlo con Latin Hypercube en esta corrida", value=False)

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

st.markdown("#### 4. Restricciones por iteración (opcional)")
use_dcf_rule = st.checkbox("En un DCF, exigir crecimiento terminal g < WACC", key="use_dcf_rule")
growth_name = wacc_name = None
if use_dcf_rule:
    if len(names) < 2 or len(set(names)) != len(names):
        st.warning("Agrega al menos dos variables con nombres únicos para aplicar la regla.")
        use_dcf_rule = False
    else:
        c1, c2 = st.columns(2)
        growth_name = c1.selectbox("Variable de crecimiento terminal (g)", names, index=names.index("g") if "g" in names else 0)
        wacc_options = [name for name in names if name != growth_name]
        wacc_name = c2.selectbox("Variable WACC", wacc_options, index=wacc_options.index("wacc") if "wacc" in wacc_options else 0)
        st.caption("Se excluyen e informan las iteraciones donde g ≥ WACC. La salida queda condicionada a la restricción y no se reemplazan los sorteos descartados.")

independent_inputs = corr is None or np.allclose(corr.to_numpy(dtype=float), np.eye(len(names)), atol=1e-8)
run_sobol = st.checkbox(
    "Calcular índices de Sobol (SALib)", value=False,
    disabled=not independent_inputs or use_dcf_rule,
    help="Sobol requiere entradas independientes y una salida válida en todo el espacio de muestreo.",
)
if not independent_inputs:
    st.caption("Sobol está desactivado porque la matriz modela entradas correlacionadas. La correlación de rangos sí refleja esa dependencia conjunta.")
elif use_dcf_rule:
    st.caption("Sobol está desactivado con el filtro DCF: el análisis requiere evaluar el espacio completo sin descartar combinaciones.")

a, b = st.columns([1, 2])
threshold = a.number_input("Umbral para la probabilidad", value=0.0)
current_signature = json.dumps({
    "assumptions": [item.__dict__ for item in assumptions],
    "formula": formula,
    "iterations": int(iterations),
    "seed": int(seed),
    "unit": output_unit,
    "method": sampling_method,
    "compare_lhs": compare_lhs,
    "sobol": run_sobol,
    "threshold": float(threshold),
    "correlation": corr.to_dict(orient="split") if corr is not None else None,
    "dcf": [use_dcf_rule, growth_name, wacc_name],
}, sort_keys=True)
has_previous_result = st.session_state.simulation_result is not None
if has_previous_result and st.session_state.get("simulation_signature") != current_signature:
    st.session_state.simulation_result = None
    st.session_state.simulation_comparison = None
    st.session_state.simulation_convergence = None
    st.session_state.simulation_sobol = None
    st.session_state.pop("simulation_meta", None)
    st.session_state.simulation_signature = None
    st.info("El modelo cambió. Los resultados anteriores se quitaron; ejecuta la simulación para ver los resultados actualizados.")
run_clicked = b.button("Ejecutar simulación", type="primary", width="stretch", disabled=not formula.strip())
if run_clicked:
    try:
        draws, outcome = run_simulation(
            assumptions=assumptions, formula=formula, iterations=int(iterations), seed=int(seed),
            correlation=corr, sampling_method=sampling_method,
        )
        discarded = 0
        if use_dcf_rule and growth_name and wacc_name:
            draws, outcome, discarded = filter_draws(draws, outcome, draws[growth_name] < draws[wacc_name])
        st.session_state.simulation_result = (draws, outcome)
        st.session_state.simulation_comparison = None
        if compare_lhs:
            comparison_method = "Latin Hypercube" if sampling_method == "Monte Carlo" else "Monte Carlo"
            comparison_draws, comparison_outcome = run_simulation(
                assumptions=assumptions, formula=formula, iterations=int(iterations), seed=int(seed),
                correlation=corr, sampling_method=comparison_method,
            )
            if use_dcf_rule and growth_name and wacc_name:
                comparison_draws, comparison_outcome, comparison_discarded = filter_draws(
                    comparison_draws, comparison_outcome,
                    comparison_draws[growth_name] < comparison_draws[wacc_name],
                )
            else:
                comparison_discarded = 0
            st.session_state.simulation_comparison = {
                "method": comparison_method, "draws": comparison_draws, "outcome": comparison_outcome,
                "discarded": comparison_discarded,
            }
        standard_error, bootstrap_intervals = convergence_summary(outcome, seed=int(seed) + 1)
        st.session_state.simulation_convergence = {
            "standard_error": standard_error, "intervals": bootstrap_intervals,
        }
        st.session_state.simulation_sobol = (
            sobol_sensitivity(assumptions, formula, seed=int(seed))
            if run_sobol and independent_inputs and not use_dcf_rule else None
        )
        st.session_state.simulation_meta = {
            "iterations": int(iterations), "seed": int(seed), "formula": formula,
            "threshold": float(threshold), "unit": output_unit, "method": sampling_method,
            "discarded": discarded,
        }
        st.session_state.simulation_signature = current_signature
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
    st.caption(f"Muestreo: {meta['method']} · Iteraciones válidas: {len(outcome):,} de {meta['iterations']:,} · Descartadas: {meta['discarded']:,} · Semilla: {meta['seed']}")

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

    convergence = st.session_state.simulation_convergence
    st.markdown("#### Convergencia y precisión")
    st.metric("Error estándar de la media", f"{decimal(convergence['standard_error'], 4)} {meta['unit']}")
    st.dataframe(
        convergence["intervals"].style.format({
            column: (lambda value: decimal(value, 4))
            for column in convergence["intervals"].columns if column != "Percentil"
        }), hide_index=True, width="stretch",
    )
    st.caption("Los intervalos bootstrap del 95% cuantifican incertidumbre de muestreo sobre P5, P50 y P95; no incorporan incertidumbre adicional sobre el modelo.")

    comparison = st.session_state.simulation_comparison
    if comparison is not None:
        comparison_rows = []
        for label, values, discarded_count in [
            (meta["method"], outcome, meta["discarded"]),
            (comparison["method"], comparison["outcome"], comparison["discarded"]),
        ]:
            comparison_rows.append({
                "Método": label, "Iteraciones válidas": len(values), "Descartadas": discarded_count,
                "Media": np.mean(values), "Error estándar": np.std(values, ddof=1) / np.sqrt(len(values)),
                "P5": np.percentile(values, 5), "P50": np.percentile(values, 50), "P95": np.percentile(values, 95),
            })
        comparison_frame = pd.DataFrame(comparison_rows)
        st.markdown("#### Comparación de muestreo")
        st.dataframe(comparison_frame.style.format({
            column: (lambda value: decimal(value, 4))
            for column in comparison_frame.columns if column != "Método"
        }), hide_index=True, width="stretch")
        st.caption("Latin Hypercube estratifica cada entrada. La comparación usa la misma cantidad solicitada de iteraciones, semilla y supuestos.")

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

    sobol_frame = st.session_state.simulation_sobol
    if sobol_frame is not None:
        st.markdown("#### Sensibilidad global de Sobol")
        sobol_chart = go.Figure()
        sobol_chart.add_trace(go.Bar(name="Primer orden", x=sobol_frame["Variable"], y=sobol_frame["Sobol primer orden"]))
        sobol_chart.add_trace(go.Bar(name="Total", x=sobol_frame["Variable"], y=sobol_frame["Sobol total"]))
        sobol_chart.update_layout(title="Índices de Sobol · entradas independientes", barmode="group", yaxis_title="Índice", height=380)
        st.plotly_chart(sobol_chart, width="stretch")
        st.dataframe(sobol_frame, hide_index=True, width="stretch")
        st.caption("Sobol descompone la varianza por efectos principales e interacciones, bajo entradas independientes. La correlación de rangos mide asociación en la corrida conjunta y no equivale a causalidad.")

    model_json = {
        "formula": formula, "iterations": int(iterations), "seed": int(seed),
        "assumptions": [item.__dict__ for item in assumptions],
        "sampling_method": meta["method"],
        "dcf_constraint": {"growth": growth_name, "wacc": wacc_name} if use_dcf_rule else None,
        "correlation": corr.to_dict(orient="split") if corr is not None else None,
    }
    st.download_button("Descargar modelo (JSON)", json.dumps(model_json, indent=2), "modelo_incertidumbre.json", "application/json")
    result_csv = pd.DataFrame({**draws, "resultado": outcome}).to_csv(index=False).encode("utf-8")
    st.download_button("Descargar iteraciones (CSV)", result_csv, "resultados_simulacion.csv", "text/csv")

footer()
