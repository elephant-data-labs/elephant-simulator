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
    "muestra asociación conjunta, no causalidad. No es una medida de efecto individual: "
    "al haber entradas correlacionadas, la asociación puede recoger efectos compartidos. "
    "Los índices de Sobol estiman efectos de primer orden y totales, incluidas interacciones, "
    "pero solo se habilitan con entradas independientes y sin filtro de factibilidad. "
    "Los escenarios P10, P50 y P90 presentan sorteos representativos de la distribución de salida."
)

st.markdown("#### Muestreo y convergencia")
st.write(
    "Monte Carlo genera sorteos pseudoaleatorios; Latin Hypercube estratifica cada entrada "
    "para cubrir mejor sus marginales con un número finito de corridas. La comparación entre "
    "métodos es diagnóstica: resultados cercanos no validan los supuestos. El error estándar "
    "resume la precisión de la media, mientras que los intervalos bootstrap para P5, P50 y P95 "
    "cuantifican variación de esos estimadores por muestreo. Ninguna de estas medidas incorpora "
    "incertidumbre sobre la selección de distribuciones o la estructura del modelo."
)

st.markdown("#### Regla de crecimiento terminal en DCF")
st.write(
    "El caso educativo usa una empresa ficticia y calcula valor empresa desde flujos de caja, "
    "sin estimar un precio implícito por acción ni compararlo con una cotización. Para evitar "
    "un valor terminal no válido, se exige g < WACC por iteración; las combinaciones que no "
    "cumplen se descartan y se informan. Por ello, el resultado es una distribución condicionada "
    "a la regla y al resto de supuestos."
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
