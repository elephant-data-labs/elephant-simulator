from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm

from risk_lab.econometrics import (
    AR1Experiment, simulate_independent_ar1_regressions, simulate_persistence_sweep,
)
from risk_ui.brand import footer, setup_page
from risk_ui.format import decimal, percent

setup_page(
    "Regresión espuria con procesos AR(1)",
    "Comprueba cómo la persistencia puede distorsionar la prueba t de una regresión entre series independientes.",
)

st.markdown("**Proceso que se simula**")
st.latex(r"Y_t = \phi_y Y_{t-1}+e_t,\qquad X_t=\phi_x X_{t-1}+v_t")
st.markdown("**Regresión estimada, incluyendo constante**")
st.latex(r"Y_t=\alpha+\beta X_t+u_t,\qquad H_0:\beta=0")
st.caption(
    "La forma de estas ecuaciones se mantiene fija para este ejercicio. Puedes cambiar los valores de "
    "persistencia y el diseño de la simulación en los controles; todavía no es un editor de ecuaciones libres."
)
with st.expander("¿Qué significa cada símbolo?"):
    st.markdown(
        "- **t − 1**: el periodo anterior.\n"
        "- **φ (phi)**: cuánto influye el valor anterior; más cerca de 1 significa mayor persistencia.\n"
        "- **eₜ y vₜ**: shocks aleatorios nuevos e independientes que reciben las series.\n"
        "- **α**: constante de la regresión; **β**: pendiente cuya hipótesis se prueba.\n"
        "- **Rezagos HAC**: periodos pasados usados para corregir el error estándar, el t y el p-valor; no cambian α ni β."
    )
st.info(
    "La hipótesis nula es verdadera por construcción: X e Y se generan de manera independiente. "
    "Al repetir el experimento podemos medir cuántas veces cada prueba la rechaza por error."
)

with st.form("ar1_experiment"):
    col1, col2, col3 = st.columns(3)
    observations = col1.select_slider(
        "Observaciones por serie", options=[100, 250, 500, 1_000, 1_500, 2_000], value=1_000,
        help="Cantidad de periodos incluidos en cada serie Y y X. El ejercicio original usa 1.000.",
    )
    replications = col2.select_slider(
        "Repeticiones del experimento", options=[100, 250, 500, 1_000, 2_000], value=1_000,
        help="Cuántas veces se generan dos series nuevas y se vuelve a estimar la regresión. Sirve para medir la tasa de falsos rechazos.",
    )
    seed = col3.number_input(
        "Semilla aleatoria", min_value=0, max_value=2_147_483_647, value=42, step=1,
        help="Fija la secuencia aleatoria. Si mantienes la misma semilla y los mismos controles, obtendrás la misma simulación.",
    )

    col4, col5, col6 = st.columns(3)
    phi_y = col4.slider(
        "Persistencia de Y (φᵧ)", 0.0, 0.99, 0.95, 0.01,
        help="Coeficiente que multiplica Y del periodo anterior. Cerca de 1, Y cambia lentamente y es más persistente.",
    )
    phi_x = col5.slider(
        "Persistencia de X (φₓ)", 0.0, 0.99, 0.95, 0.01,
        help="Coeficiente que multiplica X del periodo anterior. Cerca de 1, X cambia lentamente y es más persistente.",
    )
    hac_lags = col6.slider(
        "Periodos para corrección HAC", 0, min(20, observations - 1), 6,
        help="Periodos pasados que se consideran al corregir el error estándar HAC. Cambia la inferencia y el p-valor, no la fórmula ni el beta estimado.",
    )
    st.caption(
        "HAC significa errores estándar robustos a heterocedasticidad y autocorrelación. "
        "Los rezagos determinan hasta cuántos periodos hacia atrás se ajusta el cálculo."
    )
    col7, col8 = st.columns(2)
    significance_percent = col7.slider(
        "Nivel de significancia (%)",
        min_value=1, max_value=20, value=10, step=1,
        help="Puedes elegir entre 1% y 20%. Si H₀ es verdadera y la prueba está bien calibrada, este porcentaje es la tasa esperada de falsos rechazos. La lámina usa 10%.",
    )
    show_sweep = col8.checkbox(
        "Mostrar cómo cambia el rechazo al aumentar φ",
        value=True,
        help="Repite la simulación usando el mismo nivel de persistencia en X e Y para cada punto de la curva.",
    )
    submitted = st.form_submit_button("Ejecutar experimento", type="primary", width="stretch")

if submitted:
    try:
        settings = AR1Experiment(
            observations=int(observations),
            replications=int(replications),
            phi_y=float(phi_y),
            phi_x=float(phi_x),
            hac_lags=int(hac_lags),
            significance_level=float(significance_percent) / 100,
            seed=int(seed),
        )
        results, first_run = simulate_independent_ar1_regressions(settings)
        sweep = simulate_persistence_sweep(settings) if show_sweep else None
        st.session_state.ar1_results = results
        st.session_state.ar1_first_run = first_run
        st.session_state.ar1_settings = settings
        st.session_state.ar1_sweep = sweep
    except ValueError as exc:
        st.error(str(exc))

if "ar1_results" in st.session_state:
    results = st.session_state.ar1_results
    first_run = st.session_state.ar1_first_run
    settings = st.session_state.ar1_settings
    alpha_level = settings.significance_level
    rejection_ols = float(results["Rechaza OLS"].mean())
    rejection_hac = float(results["Rechaza HAC"].mean())
    persistence_product = settings.phi_y * settings.phi_x
    t_variance_factor = (1 + persistence_product) / (1 - persistence_product)
    t_sd_factor = float(np.sqrt(t_variance_factor))
    approximate_ols_size = float(
        2 * norm.sf(norm.ppf(1 - alpha_level / 2) / t_sd_factor)
    )

    st.markdown("#### Frecuencia de rechazo de H₀: β = 0")
    st.caption(
        f"Resultados de {settings.replications:,} experimentos · n = {settings.observations:,} · "
        f"φᵧ = {settings.phi_y:.2f} · φₓ = {settings.phi_x:.2f} · rezagos HAC = {settings.hac_lags} · "
        f"semilla = {settings.seed}"
    )
    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Rechazo OLS usual", percent(rejection_ols), f"nivel objetivo {percent(alpha_level)}")
    metric2.metric("Rechazo HAC", percent(rejection_hac), f"nivel objetivo {percent(alpha_level)}")
    metric3.metric("Beta verdadero", "0", "series independientes")

    rates = go.Figure(go.Bar(
        x=["OLS usual", "HAC / Newey–West", "Nivel nominal"],
        y=[rejection_ols, rejection_hac, alpha_level],
        marker_color=["#bb4c3f", "#157a8a", "#8193a5"],
        text=[percent(rejection_ols), percent(rejection_hac), percent(alpha_level)],
        textposition="outside",
    ))
    rates.update_layout(
        title="Tasa empírica de falsos rechazos",
        yaxis_title="Proporción de experimentos que rechazan H₀",
        yaxis_tickformat=".0%", yaxis_range=[0, min(1, max(rejection_ols, rejection_hac, alpha_level) * 1.18)],
        height=380, margin=dict(t=55, b=20),
    )
    st.plotly_chart(rates, width="stretch")
    st.caption(
        "OLS usual usa el error estándar que supone errores sin autocorrelación. HAC corrige la "
        "varianza de la pendiente por autocorrelación y usa una aproximación normal; su desempeño "
        "puede depender del número de rezagos y del tamaño muestral."
    )

    sweep = st.session_state.get("ar1_sweep")
    if sweep is not None:
        st.markdown("#### Efecto de aumentar la persistencia en ambos procesos")
        sweep_chart = go.Figure()
        sweep_chart.add_trace(go.Scatter(
            x=sweep["Persistencia φ"], y=sweep["Rechazo OLS usual"],
            mode="lines+markers", name="OLS usual", line=dict(color="#bb4c3f", width=3),
        ))
        sweep_chart.add_trace(go.Scatter(
            x=sweep["Persistencia φ"], y=sweep["Rechazo HAC"],
            mode="lines+markers", name="HAC / Newey–West", line=dict(color="#157a8a", width=3),
        ))
        sweep_chart.add_trace(go.Scatter(
            x=sweep["Persistencia φ"], y=sweep["Nivel nominal"],
            mode="lines", name=f"Nivel esperado ({percent(alpha_level)})",
            line=dict(color="#596a78", dash="dash"),
        ))
        sweep_chart.update_layout(
            title="Porcentaje de veces que se rechaza H₀, aunque β = 0",
            xaxis_title="Persistencia común φ de X e Y",
            yaxis_title="Tasa de rechazo", yaxis_tickformat=".0%",
            yaxis_range=[0, min(1, max(float(sweep["Rechazo OLS usual"].max()),
                                      float(sweep["Rechazo HAC"].max()), alpha_level) * 1.15)],
            height=410,
        )
        st.plotly_chart(sweep_chart, width="stretch")
        st.caption(
            "La línea discontinua es el nivel de significancia elegido. Si OLS usual la supera al "
            "aumentar φ, está rechazando demasiado una hipótesis nula que sabemos que es cierta. "
            "HAC intenta corregir la inferencia por autocorrelación."
        )

    st.markdown("#### Respuesta a la pregunta")
    st.success(
        f"La hipótesis nula **H₀: β = 0 es verdadera**: las series se generaron independientemente. "
        f"En estas {settings.replications:,} réplicas, OLS usual la rechazó en {percent(rejection_ols)} "
        f"de los casos, mientras que HAC la rechazó en {percent(rejection_hac)}. "
        f"La tasa nominal elegida es {percent(alpha_level)}; un rechazo aislado no demuestra una relación real."
    )
    st.write(
        f"Con φₓφᵧ = {persistence_product:.4f}, la aproximación asintótica para el t-estadístico "
        f"OLS usual tiene una desviación estándar de **{decimal(t_sd_factor, 2)}** veces la habitual "
        f"(factor de varianza **{decimal(t_variance_factor, 2)}**). Para φₓ = φᵧ = 0,95, eso implica "
        f"una tasa teórica aproximada de falsos rechazos de **{percent(approximate_ols_size)}**, "
        f"en lugar del nivel nominal de {percent(alpha_level)}. Es un referente asintótico; la tasa simulada arriba es la respuesta para "
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
