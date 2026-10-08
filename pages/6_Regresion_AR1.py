from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm

from risk_lab.econometrics import AR1Experiment, simulate_independent_ar1_regressions
from risk_ui.brand import footer, setup_page
from risk_ui.format import decimal, percent

setup_page(
    "Regresión espuria con procesos AR(1)",
    "Comprueba cómo la persistencia puede distorsionar la prueba t de una regresión entre series independientes.",
)

st.markdown(
    r"""
Se generan dos series independientes:

\[Y_t = \phi_yY_{t-1}+e_t,\qquad X_t=\phi_xX_{t-1}+v_t\]

Luego se estima **con constante**:

\[
Y_t=\alpha+\beta X_t+u_t,\qquad H_0:\beta=0
\]
"""
)
st.info(
    "La hipótesis nula es verdadera por construcción: X e Y se generan de manera independiente. "
    "Al repetir el experimento podemos medir cuántas veces cada prueba la rechaza por error."
)

with st.form("ar1_experiment"):
    col1, col2, col3 = st.columns(3)
    observations = col1.select_slider(
        "Observaciones por serie", options=[100, 250, 500, 1_000, 1_500, 2_000], value=1_000
    )
    replications = col2.select_slider(
        "Experimentos independientes", options=[100, 250, 500, 1_000, 2_000], value=1_000
    )
    seed = col3.number_input("Semilla reproducible", min_value=0, max_value=2_147_483_647, value=42, step=1)

    col4, col5, col6 = st.columns(3)
    phi_y = col4.slider("Persistencia de Y (φᵧ)", 0.0, 0.99, 0.95, 0.01)
    phi_x = col5.slider("Persistencia de X (φₓ)", 0.0, 0.99, 0.95, 0.01)
    hac_lags = col6.slider("Rezagos HAC", 0, min(20, observations - 1), 6)
    submitted = st.form_submit_button("Ejecutar experimento", type="primary", width="stretch")

if submitted:
    try:
        settings = AR1Experiment(
            observations=int(observations),
            replications=int(replications),
            phi_y=float(phi_y),
            phi_x=float(phi_x),
            hac_lags=int(hac_lags),
            seed=int(seed),
        )
        results, first_run = simulate_independent_ar1_regressions(settings)
        st.session_state.ar1_results = results
        st.session_state.ar1_first_run = first_run
        st.session_state.ar1_settings = settings
    except ValueError as exc:
        st.error(str(exc))

if "ar1_results" in st.session_state:
    results = st.session_state.ar1_results
    first_run = st.session_state.ar1_first_run
    settings = st.session_state.ar1_settings
    alpha_level = 0.05
    rejection_ols = float(results["Rechaza OLS"].mean())
    rejection_hac = float(results["Rechaza HAC"].mean())
    persistence_product = settings.phi_y * settings.phi_x
    t_variance_factor = (1 + persistence_product) / (1 - persistence_product)
    t_sd_factor = float(np.sqrt(t_variance_factor))
    approximate_ols_size = float(2 * norm.sf(norm.ppf(0.975) / t_sd_factor))

    st.markdown("#### Frecuencia de rechazo de H₀: β = 0")
    st.caption(
        f"Resultados de {settings.replications:,} experimentos · n = {settings.observations:,} · "
        f"φᵧ = {settings.phi_y:.2f} · φₓ = {settings.phi_x:.2f} · rezagos HAC = {settings.hac_lags} · "
        f"semilla = {settings.seed}"
    )
    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Rechazo OLS usual", percent(rejection_ols), "nivel objetivo 5%")
    metric2.metric("Rechazo HAC", percent(rejection_hac), "nivel objetivo 5%")
    metric3.metric("Beta verdadero", "0", "series independientes")

    rates = go.Figure(go.Bar(
        x=["OLS usual", "HAC / Newey–West", "Nivel nominal"],
        y=[rejection_ols, rejection_hac, alpha_level],
        marker_color=["#bb4c3f", "#157a8a", "#8193a5"],
        text=[percent(rejection_ols), percent(rejection_hac), "5,0%"],
        textposition="outside",
    ))
    rates.update_layout(
        title="Tasa empírica de falsos rechazos",
        yaxis_title="Proporción de experimentos que rechazan H₀",
        yaxis_tickformat=".0%", yaxis_range=[0, min(1, max(rejection_ols, rejection_hac, 0.05) * 1.18)],
        height=380, margin=dict(t=55, b=20),
    )
    st.plotly_chart(rates, width="stretch")
    st.caption(
        "OLS usual usa el error estándar que supone errores sin autocorrelación. HAC corrige la "
        "varianza de la pendiente por autocorrelación y usa una aproximación normal; su desempeño "
        "puede depender del número de rezagos y del tamaño muestral."
    )

    st.markdown("#### Respuesta a la pregunta")
    st.success(
        f"La hipótesis nula **H₀: β = 0 es verdadera**: las series se generaron independientemente. "
        f"En estas {settings.replications:,} réplicas, OLS usual la rechazó en {percent(rejection_ols)} "
        f"de los casos, mientras que HAC la rechazó en {percent(rejection_hac)}. "
        "La tasa nominal es 5%; un rechazo aislado no demuestra una relación real."
    )
    st.write(
        f"Con φₓφᵧ = {persistence_product:.4f}, la aproximación asintótica para el t-estadístico "
        f"OLS usual tiene una desviación estándar de **{decimal(t_sd_factor, 2)}** veces la habitual "
        f"(factor de varianza **{decimal(t_variance_factor, 2)}**). Para φₓ = φᵧ = 0,95, eso implica "
        f"una tasa teórica aproximada de falsos rechazos de **{percent(approximate_ols_size)}**, "
        "en lugar de 5%. Es un referente asintótico; la tasa simulada arriba es la respuesta para "
        "el tamaño muestral, rezagos HAC y semilla seleccionados."
    )

    st.markdown("#### Una regresión representativa")
    representative = results.iloc[0]
    path = go.Figure()
    path.add_trace(go.Scatter(x=first_run["Periodo"], y=first_run["Y"], name="Y", line=dict(color="#17364a")))
    path.add_trace(go.Scatter(x=first_run["Periodo"], y=first_run["X"], name="X", line=dict(color="#157a8a")))
    path.update_layout(title="Primera réplica: series generadas independientemente", xaxis_title="Período", height=360)
    st.plotly_chart(path, width="stretch")
    st.markdown(
        f"**Errores estándar HAC y covarianza** · kernel Bartlett/Newey–West · "
        f"rezagos = {settings.hac_lags}"
    )
    coefficient_table = pd.DataFrame([
        {
            "Variable": "Constante",
            "Coeficiente": representative["Constante estimada"],
            "EE OLS": representative["EE OLS constante"],
            "t OLS": representative["t OLS constante"],
            "p OLS": representative["p OLS constante"],
            "EE HAC": representative["EE HAC constante"],
            "t HAC": representative["t HAC constante"],
            "p HAC": representative["p HAC constante"],
        },
        {
            "Variable": "X",
            "Coeficiente": representative["Beta estimado"],
            "EE OLS": representative["EE OLS beta"],
            "t OLS": representative["t OLS usual"],
            "p OLS": representative["p OLS usual"],
            "EE HAC": representative["EE HAC beta"],
            "t HAC": representative["t HAC"],
            "p HAC": representative["p HAC"],
        },
    ])
    numeric_columns = [column for column in coefficient_table.columns if column != "Variable"]
    st.dataframe(
        coefficient_table.style.format({column: (lambda value: decimal(value, 4)) for column in numeric_columns}),
        hide_index=True, width="stretch",
    )
    st.caption(
        f"Variable dependiente: Y · Mínimos cuadrados · Observaciones incluidas: {settings.observations:,} · "
        "La prueba HAC usa una aproximación normal; OLS usual usa distribución t con n − 2 grados de libertad."
    )
    diagnostics = pd.DataFrame([{
        "R cuadrado": representative["R cuadrado"],
        "R cuadrado ajustado": representative["R cuadrado ajustado"],
        "Error estándar regresión": representative["Error estándar regresión"],
        "Suma residuos cuadrados": representative["Suma residuos cuadrados"],
        "Durbin–Watson": representative["Durbin-Watson"],
        "Media dependiente": representative["Media Y"],
        "Desv. estándar dependiente": representative["Desv. estándar Y"],
        "AIC": representative["AIC"],
        "BIC": representative["BIC"],
    }])
    with st.expander("Diagnósticos de la regresión"):
        st.dataframe(
            diagnostics.style.format({column: (lambda value: decimal(value, 4)) for column in diagnostics.columns}),
            hide_index=True, width="stretch",
        )
    st.caption(
        "La constante α se estima en todas las regresiones. Una réplica puede dar un β aparentemente "
        "significativo aun cuando la relación verdadera es cero; la tasa de rechazos de muchas réplicas "
        "muestra el tamaño efectivo de cada prueba."
    )

    with st.expander("Ver resultados por experimento"):
        st.dataframe(results, hide_index=True, width="stretch")
        csv = results.to_csv(index=False).encode("utf-8")
        st.download_button("Descargar resultados CSV", csv, "experimento_ar1.csv", "text/csv")

st.warning(
    "φ < 1: el ejemplo usa procesos estacionarios pero muy persistentes, no caminatas aleatorias. "
    "La simulación ilustra distorsión de tamaño en inferencia OLS convencional; no es una conclusión "
    "sobre cualquier regresión de series de tiempo."
)
footer()
