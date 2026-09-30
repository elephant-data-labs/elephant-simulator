# Elephant Simulator

Simulador educativo y auditable de incertidumbre en Python y Streamlit. Permite construir modelos con distribuciones, correlaciones, Monte Carlo, Latin Hypercube, sensibilidad global y ajuste a observaciones propias.

La organización sigue el patrón de Analizador Financiero: `app.py` declara la navegación, `pages/` contiene una página por módulo, `src/risk_lab/` concentra el cálculo sin importar Streamlit y `src/risk_ui/` centraliza identidad visual y formato. La marca usa la paleta y el logo de Elephant Data Labs.

## Estructura

```text
app.py                          enrutador y navegación
pages/                          una página por módulo
  0_Inicio.py                   orientación y accesos
  1_Simulacion.py               modelo, Monte Carlo y resultados
  2_Ajuste_de_distribuciones.py comparación de ajustes a datos
  3_Pronostico.py               remuestreo exploratorio de series
  4_Optimizacion.py             comparación de decisiones
  5_Metodologia.py              metodología y límites
src/risk_lab/                   motor de cálculo, independiente de Streamlit
src/risk_ui/                    marca, estilos y formatos compartidos
data/                           reservado para ejemplos públicos y modelos
```

## Funciones de V1

- Simulación Monte Carlo reproducible con hasta 100.000 iteraciones y semilla configurable.
- Muestreo Latin Hypercube y comparación con Monte Carlo.
- Distribuciones Normal, Uniforme, Triangular, PERT y Lognormal.
- Fórmulas aritméticas seguras con variables y funciones matemáticas básicas.
- Dependencia entre supuestos mediante matriz declarada y cópula gaussiana.
- Distribución de resultados, percentiles, probabilidad frente a un umbral y resumen de cola inferior.
- Sensibilidad por correlación de rangos; índices de Sobol con SALib para entradas independientes.
- Error estándar de la media e intervalos bootstrap del 95% para P5, P50 y P95.
- Regla DCF opcional por iteración para exigir crecimiento terminal g < WACC; informa sorteos descartados.
- Exportación de la especificación a JSON y de iteraciones a CSV.
- Comparación de distribuciones candidatas ajustadas a observaciones de archivos CSV.
- Pronóstico exploratorio por remuestreo de cambios históricos.
- Optimización en grilla de una variable de decisión con restricción opcional.

## Ejecutar localmente

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

## Publicación

El punto de entrada es `app.py`; las dependencias están en `requirements.txt` y no se requieren claves secretas. La aplicación está organizada para desplegarse desde este repositorio en Streamlit Community Cloud.

## Límites y metodología

- La matriz de correlación guía una cópula gaussiana; con marginales no normales, la correlación resultante puede diferir de la declarada.
- La correlación de rangos mide asociación en la corrida conjunta, no un efecto causal individual.
- Sobol requiere entradas independientes y una salida válida en todo el espacio; se desactiva si hay correlaciones o filtro DCF.
- Los intervalos bootstrap describen error de muestreo y no incertidumbre sobre la especificación o las distribuciones.
- El caso DCF educativo usa una empresa ficticia y supuestos de ejemplo. Calcula valor empresa sin precio implícito por acción ni comparación con cotizaciones; se descartan e informan las iteraciones con g ≥ WACC.
- El pronóstico remuestrea cambios históricos y no modela automáticamente tendencia, estacionalidad ni quiebres estructurales.
- El AIC compara candidatos sencillos, pero no demuestra que una distribución sea verdadera.
- La optimización compara una grilla finita y no garantiza un óptimo global.
- No incluye persistencia de proyectos ni colaboración entre usuarios.
- Los resultados dependen del modelo y los supuestos; son una herramienta educativa, no predicciones garantizadas ni recomendaciones.

La aplicación no carga ni ejecuta libros Excel. La valoración de empresas es uno de varios usos posibles, no el propósito exclusivo del simulador.
