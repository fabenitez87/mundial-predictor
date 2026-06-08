# Mundial Predictor — Pedido de documentacion tecnica completa

## Contexto

El proyecto es un predictor del Mundial 2026 basado en modelo Poisson bivariado
+ Monte Carlo, con prior ELO (MAP estimation) y simulacion del bracket oficial FIFA.
Deployado en Streamlit Cloud. Repo: https://github.com/fabenitez87/mundial-predictor

El CLAUDE.md completo del proyecto esta al final de este mensaje.

## Pedido

Necesito un documento tecnico extenso y didactico que cubra todo
lo implementado en este proyecto. El objetivo es que yo pueda
estudiarlo, entender cada decision, y tener una referencia completa.

El documento debe incluir:

### 1. Fundamentos teoricos
- Que es el modelo Poisson bivariado y por que es apropiado para futbol
- Diferencia entre MLE clasico y MAP (Maximum A Posteriori)
- Que es el decay temporal y por que mejora el modelo
- Que son los ratings ELO y como capturan la calidad de un equipo

### 2. Cada decision de diseno y por que
- Por que L-BFGS-B y no otro optimizador
- Por que parametros en log-escala
- Por que xi=0.0018 para clubes y xi=0.004 para selecciones
- Por que 6 parametros de confederacion estimados por MLE
- Por que shrinkage adaptativo por n_partidos
- Por que lambda_prior=25 para el prior ELO

### 3. Limitaciones conocidas del modelo
- Que no puede capturar Poisson puro
- Por que el Brier Score es mas importante que el accuracy para calibracion
- Que datos mejorarian el modelo significativamente

### 4. Proximos pasos recomendados
- Migracion a FastAPI + React
- Que agregarias si tuvieras mas datos
- Como monitorear el modelo durante el Mundial 2026

Formato: markdown, con formulas matematicas donde corresponda,
ejemplos de codigo Python comentado para cada concepto clave.

---

## CLAUDE.md completo del proyecto

# mundial-predictor — CLAUDE.md

Estado del proyecto al 2026-06-08. Leer esto antes de tocar cualquier archivo.

---

## Stack y versiones

```
Python 3.13 (Windows 11)
pandas      2.3.2
numpy       2.3.2
scipy       1.17.1
requests    (ultima)
streamlit   1.50.0
openpyxl    (ultima)   <- leer .xlsx de football-data.co.uk
matplotlib  (ultima)   <- requerido por requirements
plotly      (ultima)   <- heatmap px.imshow + bar chart en app.py
lxml        (ultima)   <- dependencia de pandas.read_html
```

## Estructura de archivos

```
mundial-predictor/
|-  app.py          <- Streamlit app (3 secciones)
|-  main.py         <- Pipeline CLI completo para testing
|-  validar.py      <- Validacion retrospectiva WC2022
|-  data/
|   |-  .gitkeep
|   |-  *.csv       <- ignorado: descargados en runtime
|   |-  *.xlsx      <- ignorado: descargados en runtime
|-  src/
    |-  carga_datos.py
    |-  parametros.py
    |-  prediccion.py
    |-  mundial.py
```

## src/carga_datos.py

- `cargar_partidos()` — 6 CSVs football-data.co.uk (Premier E0, La Liga SP1, 21/22-23/24). 2280 filas. Cachea en /data/.
- `cargar_internacionales()` — WorldCup2026.xlsx: WC2018 (64) + WC2022 (64) + Qualifiers2026 (889). 1017 filas. Normaliza nombres.
- `cargar_elo()` — eloratings.net/World.tsv: ELO actual de los 48 equipos del WC. Cachea en data/elo_selecciones.csv. Retorna None si falla.
- `ELO_CODIGO_EQUIPO` — mapeo codigo eloratings.net -> nombre en GRUPOS

## src/parametros.py

Funciones:
- `escalar_elo_a_alpha(elo_val, elo_mean, elo_std, alpha_mean=1.88, alpha_std=0.5)`
  Escala ELO (~1000-2100) a alpha (~1.0-3.0) via z-score en log-espacio:
  alpha = exp(log(alpha_mean) + k * (elo - elo_mean) / elo_std)
  donde k = log(alpha_mean + alpha_std) - log(alpha_mean)

- `estimar_parametros(df, xi=0.0018)` — MLE L-BFGS-B estandar para clubes.

- `estimar_selecciones(df, elo_series=None, lambda_prior=25.0)`
  MLE extendido con:
  (a) 5 parametros de confederacion (UEFA=1.0 fijo)
  (b) prior ELO MAP si elo_series no es None:
      F(theta) = -logL + lambda_conf*sum(log_s^2) + lambda_prior*sum((log_alpha - log_alpha_prior)^2)
  (c) Inicializa en el prior ELO para mejor convergencia
  Retorna (fuerzas_normalizadas, gamma, conf_strengths)

## Prior ELO (MAP estimation) — diseno

### Problema que resuelve
MLE puro: France alpha=1.02 (golea poco en UEFA competitivo) -> 0% campeon.
Con ELO: France ELO=2062 -> alpha_prior=2.69 -> MAP anchor -> France 5.38% campeon.

### Objetivo MAP completo
```
F(theta) = -log L(theta|data)
         + 1.0 * sum(log_s_free^2)           [reg. confederaciones]
         + 25.0 * sum((log_alpha_i - log_alpha_prior_i)^2)  [prior ELO]
```

### Parametros calibrados (con ELO, lambda_prior=25)
```
gamma = 1.3391

Conf strengths (UEFA=1.0):
  AFC       1.1665   CONMEBOL  1.0205   CAF  0.9970
  OFC       0.9068   CONCACAF  0.8972

ELO ratings usados:
  Spain     2155   Argentina  2114   France  2062
  England   2021   Brazil     1991   Portugal 1986
  Germany   1932   Japan      1906   Norway  1914
  Haiti     1548   (...)
```

### Resultados partidos test (lambda=25)
| Partido | Sin ELO | Con ELO |
|---|---|---|
| France vs Argentina | 12.9% Fra | 28.3% Fra |
| Argentina vs Haiti  | 66.3% Arg | 68.4% Arg |
| Brazil vs Germany   | 50.7% Bra | 38.9% Bra ✓ |
| Spain vs France     | 56.7% Spa | 40.1% Spa ✓ |

## src/mundial.py

GRUPOS: 12 grupos oficiales FIFA 2026 (sorteo 5-dic-2025), grupos A-L.
TERCEROS_SLOTS: slots del Annex C FIFA para 8 mejores terceros (backtracking matching).

Pipeline calcular_lambdas (3 capas):
1. Shrinkage adaptativo: sf=0.30/0.60/0.85 segun n>=50/20/<20 partidos
   target = elo_alpha_prior[e] si hay ELO, else alpha_media_wc
   alpha_adj = (1-sf)*alpha_raw + sf*target
2. Factor conf sobre alpha: SOLO si skip_conf_alpha=False (sin ELO)
3. Beta cross-conf: beta_defensor sube cuando atacante viene de conf mas fuerte

conf_ctx = {
    "strengths": dict,         "equipo_conf": dict,
    "beta_media": float,       "alpha_media": float,
    "beta_media_wc": float,    "n_partidos": dict,
    "elo_alpha_prior": dict,   "skip_conf_alpha": bool,
}

Bracket FIFA 2026 implementado (R32->R16->QF->SF->Final):
R32: w73-w88 con cruces oficiales Annex C
R16: w89=W74vsW77, w90=W73vsW75, w91=W76vsW78, w92=W79vsW80,
     w93=W83vsW84, w94=W81vsW82, w95=W86vsW88, w96=W85vsW87
QF:  w97=W89vsW90, w98=W93vsW94, w99=W91vsW92, w100=W95vsW96
SF/F: estandar

## app.py — Streamlit

cargar_modelo() con @st.cache_resource (corre UNA vez):
  cargar_internacionales() -> cargar_elo() -> estimar_selecciones(elo, lambda=25)
  construye conf_ctx con elo_alpha_prior, skip_conf_alpha=True

lambdas_neutral correcto (promedia goles propios de cada equipo):
  lambda_eq1 = (lh1 + la2) / 2    lambda_eq2 = (la1 + lh2) / 2

## Validacion WC2022

Training: ~1134 partidos pre-WC2022 (sin clasificatorias 2019-2022).
Test: 64 partidos WC2022 con odds de mercado.

Modelo Poisson basico:  accuracy=54.7%  Brier=0.6108
Mercado (Pinnacle):     accuracy=53.1%  Brier=0.5836
Eliminatorias:          Poisson 62.5% > Mercado 56.2%

## Problemas conocidos

- France vs Argentina: 28.3% (target 38-42%) — Argentina ELO>France genuinamente
- ~60% equipos WC tienen sf=0.85 (pocos datos individuales)
- England domina MC con 10.4% — grupo favorable (L: Eng/Cro/Gha/Pan) + ELO alto
- cargar_elo() depende de eloratings.net (site externo); cache mitiga fallos

## Proximos pasos

1. Nations League + amistosos FIFA en training (France/Germany tendrian mas datos)
2. Validacion formal con ELO historico sobre WC2022
3. Migracion FastAPI + React (POST /predecir, POST /simular)
4. Monitoreo en tiempo real durante WC2026

## Fuentes de datos

- football-data.co.uk: CSVs ligas clubes + WorldCup2026.xlsx + internationals.xlsx
- eloratings.net: ELO ratings selecciones nacionales (World.tsv + en.teams.tsv)
- Grupos FIFA 2026: sorteo oficial 5-dic-2025 (hardcoded)
- Bracket: Annex C FIFA 2026 (495 combinaciones terceros)
