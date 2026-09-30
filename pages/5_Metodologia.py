from __future__ import annotations

import streamlit as st

from risk_ui.brand import footer, setup_page

setup_page(
    "Metodología y límites",
    "Qué calcula esta versión, cómo interpretar sus gráficos y qué queda pendiente.",
)

st.markdown("#### Simulación Monte Carlo")
st.write(
    "Cada iteración toma un valor de cada distribución definida y evalúa la fórmula. "
    "El conjunto de resultados forma una distribución que permite resumir rangos y "
    "probabilidades condicionadas a los supuestos. La semilla permite repetir la misma corrida."
)

st.markdown("#### Dependencia entre variables")
st.write(
    "La matriz se usa con una cópula gaussiana para generar supuestos relacionados. "
    "La matriz debe ser simétrica y semidefinida positiva. Con distribuciones marginales "
    "no normales, las correlaciones observadas al final pueden diferir de los valores ingresados."
)

st.markdown("#### Sensibilidad y escenarios")
st.write(
    "La sensibilidad usa correlaciones de rangos entre cada supuesto y el resultado; "
    "muestra asociación, no causalidad. Los escenarios P10, P50 y P90 presentan sorteos "
    "representativos de la distribución de salida y conservan la dependencia modelada."
)

st.markdown("#### Pronósticos y ajuste de distribuciones")
st.write(
    "El pronóstico exploratorio remuestrea cambios históricos, en forma absoluta o porcentual. "
    "No detecta automáticamente tendencia, estacionalidad ni cambios estructurales. "
    "El ajuste compara un conjunto acotado de distribuciones mediante AIC; el menor AIC entre "
    esos candidatos no demuestra que una distribución sea verdadera."
)

st.markdown("#### Optimización")
st.write(
    "La herramienta actual compara una grilla finita de decisiones con los mismos sorteos de "
    "incertidumbre. El mejor punto de esa grilla no es una garantía de óptimo global. "
    "Las restricciones se evalúan sobre cada iteración y se informa cuántos casos cumplen."
)

st.markdown("#### Trabajo pendiente")
st.write(
    "Importación de modelos de Excel, modelos reutilizables, guardado de proyectos, "
    "colaboración y optimización avanzada requieren diseño adicional. Los resultados dependen "
    "del modelo y de los supuestos ingresados; no son predicciones garantizadas ni recomendaciones."
)

footer()
