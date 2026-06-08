# Mundial Predictor — Contexto para Gemini

## Qué es esto
App web de predicción de partidos del Mundial 2026 usando modelo 
Poisson bivariado + simulación Monte Carlo. Deployada en Streamlit Cloud.

## Stack
- Python: pandas, numpy, scipy, requests, plotly, streamlit
- Deploy: Streamlit Cloud (gratuito)
- Repo: https://github.com/fabenitez87/mundial-predictor

## Estructura de archivos
src/carga_datos.py   — descarga CSVs de football-data.co.uk + internacionales
src/parametros.py    — MLE L-BFGS-B, decay temporal, calibración confederaciones
src/prediccion.py    — Poisson bivariado, matriz de marcadores, 1X2
src/mundial.py       — grupos FIFA 2026, bracket, Monte Carlo, shrinkage
app.py               — interfaz Streamlit (3 secciones)
main.py              — pipeline CLI
validar.py           — validación retrospectiva WC2022

## Modelo estadístico
- Poisson bivariado: lambda_h = gamma * alpha_local * beta_visitante
- Parámetros estimados por MLE (scipy.optimize.minimize, L-BFGS-B)
- Decay temporal xi=0.0018 (clubes), xi=0.004 (selecciones)
- 6 parámetros de confederación estimados por MLE: 
  CONMEBOL=1.30, CAF=1.00, UEFA=1.00, OFC=0.95, AFC=0.94, CONCACAF=0.77
- Shrinkage adaptativo por n_partidos hacia media global WC

## Datos de entrenamiento
- Clubes: Premier League + La Liga, temporadas 21/22, 22/23, 23/24 (football-data.co.uk)
- Selecciones: 1017 partidos internacionales 2018-2026 (football-data.co.uk)
- Fallback: media global si equipo sin datos

## Validación WC2022
- Accuracy 1X2: 54.7% (mercado apuestas: 53.1%)
- Accuracy eliminatorias: 62.5% (mercado: 56.2%)
- Brier Score: 0.61

## Problema actual a resolver
Shrinkage insuficiente para equipos con pocos partidos internacionales.
Haiti aparece casi igual de fuerte que Argentina.
Implementando shrinkage adaptativo por n_partidos (ver conversación).

## Lo que falta
1. Fix shrinkage adaptativo (en curso)
2. Migración a FastAPI + React (post-validación Streamlit)
3. Datos FIFA Rankings como prior para anclar alphas

## Pregunta para Gemini
[ACÁ ESCRIBÍS LO QUE NECESITÁS CONSULTAR]
