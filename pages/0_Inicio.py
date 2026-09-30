from __future__ import annotations

import streamlit as st

from risk_ui.brand import footer, module_card, setup_page

setup_page(
    "Laboratorio de incertidumbre",
    "Simulación, pronósticos y análisis de decisiones desde una aplicación web.",
)

st.info(
    "**Prototipo educativo.** Define supuestos, conecta variables con una fórmula y observa "
    "cómo la incertidumbre se propaga al resultado. El nombre y el alcance del proyecto siguen abiertos."
)

st.markdown("#### ¿Qué quieres hacer?")
st.caption(
    "Cada módulo resuelve una parte del análisis. Puedes empezar con el ejemplo incluido "
    "o cargar tus propios datos en las herramientas de análisis."
)

columns = st.columns(3)
with columns[0]:
    module_card(
        "Construir y simular un modelo",
        "Define variables inciertas, distribuciones, correlaciones y una fórmula de resultado; "
        "luego revisa percentiles, probabilidades y sensibilidad.",
    )
    if st.button("Abrir simulación", type="primary", width="stretch"):
        st.switch_page("pages/1_Simulacion.py")
with columns[1]:
    module_card(
        "Analizar datos históricos",
        "Compara ajustes de distribuciones candidatas con observaciones o explora trayectorias "
        "futuras mediante remuestreo.",
    )
    if st.button("Abrir herramientas de datos", width="stretch"):
        st.switch_page("pages/2_Ajuste_de_distribuciones.py")
with columns[2]:
    module_card(
        "Explorar decisiones",
        "Compara alternativas de una variable controlable y encuentra cuál ofrece el mejor "
        "resultado esperado bajo los supuestos del modelo.",
    )
    if st.button("Abrir optimización", width="stretch"):
        st.switch_page("pages/4_Optimizacion.py")

st.markdown("#### Aplicaciones posibles")
st.write(
    "La misma estructura puede apoyar modelos de presupuesto, ventas, inventario, costos de "
    "proyectos, plazos, ingeniería o finanzas. La valoración de empresas es un ejemplo, no "
    "el límite del proyecto."
)
st.warning(
    "Esta primera versión define modelos mediante variables y fórmulas en la aplicación. "
    "Todavía no importa ni ejecuta libros Excel."
)
footer()
