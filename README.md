# Laboratorio de incertidumbre

Aplicación educativa y experimental en Python y Streamlit para construir modelos numéricos, representar incertidumbre y explorar decisiones. El nombre y el alcance del proyecto siguen abiertos.

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

## Funciones del prototipo

- Simulación Monte Carlo reproducible con semilla configurable.
- Distribuciones Normal, Uniforme, Triangular, PERT y Lognormal.
- Fórmulas aritméticas seguras con variables y funciones matemáticas básicas.
- Dependencia entre supuestos mediante matriz y cópula gaussiana.
- Distribución de resultados, percentiles, probabilidad frente a un umbral y resumen de cola inferior.
- Sensibilidad por correlación de rangos y escenarios de salida representativos.
- Exportación de la especificación a JSON y de iteraciones a CSV.
- Comparación de ajustes de distribuciones candidatas a datos CSV.
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

El punto de entrada es `app.py`; las dependencias están en `requirements.txt` y no se requieren claves secretas. La aplicación está organizada para desplegarse desde un repositorio en Streamlit Community Cloud.

## Límites actuales

- No carga ni ejecuta libros Excel; el modelo se define con variables y una fórmula en la interfaz.
- El pronóstico remuestrea cambios históricos y no modela automáticamente tendencia, estacionalidad ni quiebres estructurales.
- El ajuste estadístico compara candidatos sencillos; el AIC no demuestra que una distribución sea verdadera.
- La matriz guía dependencia mediante cópula gaussiana; con marginales no normales, la correlación final no necesariamente coincide con el valor ingresado.
- La optimización compara una grilla finita y no garantiza un óptimo global.
- No incluye persistencia de proyectos ni colaboración entre usuarios.
- Los resultados dependen del modelo y de sus supuestos; no son recomendaciones ni predicciones garantizadas.

## Evolución posible

La prioridad de producto sería facilitar modelos que ya existen en hojas de cálculo. La compatibilidad con Excel debe definirse por etapas y con una lista explícita de fórmulas y funciones soportadas. La valoración financiera del Analizador Financiero puede integrarse como un ejemplo adicional, no como el único uso de la aplicación.
