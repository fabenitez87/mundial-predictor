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
matplotlib  (ultima)   <- requerido por requirements (no se usa directamente)
plotly      (ultima)   <- heatmap px.imshow + bar chart en app.py
lxml        (ultima)   <- dependencia de pandas.read_html
```

Instalar con: `pip install -r requirements.txt`

---

## Estructura de archivos

```
mundial-predictor/
|-  app.py                  <- Streamlit app (3 secciones)
|-  main.py                 <- Pipeline CLI completo para testing
|-  validar.py              <- Validacion retrospectiva WC2022
|-  requirements.txt
|-  .gitignore
|-  CLAUDE.md               <- este archivo
|-  GEMINI_CONTEXT.md       <- contexto para consultas a Gemini
|-  data/
|   |-  .gitkeep            <- directorio rastreado en git, archivos ignorados
|   |-  *.csv               <- ignorado: descargados en runtime
|   |-  *.xlsx              <- ignorado: descargados en runtime
|-  src/
    |-  __init__.py
    |-  carga_datos.py
    |-  parametros.py
    |-  prediccion.py
    |-  mundial.py
```

---

## src/carga_datos.py

**Constantes:**
- `DATA_DIR` — ruta absoluta a `/data/`, creada automaticamente con `os.makedirs`
- `NOMBRE_MAP_INT` — normaliza nombres del xlsx: `"Bosnia & Herzegovina" -> "Bosnia and Herzegovina"`, etc.
- `ELO_CODIGO_EQUIPO` — dict con los 48 equipos del WC: codigo eloratings.net -> nombre en GRUPOS

**Funciones publicas:**

| Funcion | Que hace |
|---|---|
| `cargar_partidos()` | Descarga 6 CSVs de football-data.co.uk (Premier E0, La Liga SP1, temporadas 21/22-23/24). Cachea en `/data/`. Devuelve 2280 filas con HomeTeam, AwayTeam, FTHG, FTAG. |
| `cargar_internacionales()` | Descarga `WorldCup2026.xlsx`. Extrae 3 sheets (WC2018: 64, WC2022: 64, Qualifiers2026: 889 partidos). Filtra desde 2018-01-01. Normaliza nombres. Cachea en `data/internacionales.csv`. Devuelve 1017 filas con columna Date. |
| `cargar_elo()` | Descarga `eloratings.net/World.tsv` (codigo + ELO actual) y `en.teams.tsv` (codigo -> nombre). Mapea los 48 equipos del WC via `ELO_CODIGO_EQUIPO`. Cachea en `data/elo_selecciones.csv`. Retorna `pd.Series(index=equipo, values=elo)` o `None` si falla silenciosamente. |

**Nota sobre cache:** cada funcion solo descarga si el archivo local no existe. Para forzar re-descarga, borrar el archivo en `/data/`.

---

## src/parametros.py

**Constantes:**
- `XI_CLUBES = 0.0018` — decay temporal para datos de clubes
- `XI_SELECCIONES = 0.004` — decay mas agresivo para selecciones
- `CONFS = ["UEFA","CONMEBOL","CONCACAF","AFC","CAF","OFC"]` — UEFA=indice 0, referencia fija
- `SELECCION_CONFEDERACION` — dict con los 207 equipos del dataset -> confederacion

**Funciones:**

| Funcion | Que hace |
|---|---|
| `_pesos_temporales(df, fecha_ref, xi)` | Pesos exponenciales `exp(-xi*dias)` por fila. Si no hay columna `Date`, devuelve pesos uniformes. |
| `escalar_elo_a_alpha(elo_val, elo_mean, elo_std, alpha_mean=1.88, alpha_std=0.5)` | Escala ELO (rango ~1000-2100) a alpha (rango ~1.0-3.0) via z-score en log-espacio: `alpha = exp(log(alpha_mean) + k*z)` donde `k = log(alpha_mean+alpha_std) - log(alpha_mean)`. |
| `estimar_parametros(df, xi=XI_CLUBES)` | MLE L-BFGS-B estandar. Parametros en log-escala, 2n+1 variables. Devuelve `(fuerzas, gamma)`. Usado para clubes. |
| `estimar_selecciones(df, elo_series=None, lambda_prior=25.0)` | MLE extendido con 5 parametros de confederacion + prior ELO MAP opcional. Ver seccion abajo. |

---

## Prior ELO (MAP estimation)

### Que es y por que

El MLE puro depende solo de goles anotados en clasificatorias. France (ELO=2062, ganadora del WC2018, finalista WC2022) tiene `alpha_raw=1.00` porque juega en UEFA qualifying competitivo y no golea. Sin prior, France tenia 0% de probabilidad de campeon.

La solucion: **MAP estimation** con prior ELO. El objetivo se convierte en:

```
F(theta) = -log L(theta|data)          <- MLE clasico
         + lambda_conf * sum(log_s^2)  <- regularizacion confederaciones
         + lambda_prior * sum((log_alpha_i - log_alpha_prior_i)^2)  <- prior ELO
```

El prior `log_alpha_prior_i` se deriva del ELO de cada equipo via `escalar_elo_a_alpha()`.

### Parametros de `estimar_selecciones()`

```python
fuerzas, gamma, conf_strengths = estimar_selecciones(
    df_int,
    elo_series=elo_series,   # None -> comportamiento MLE puro (sin prior)
    lambda_prior=25.0        # fuerza del prior: 25 equilibra datos e intuicion
)
```

**Rango de `lambda_prior`:**
- `lambda_prior=0`: MLE puro (France alpha=1.02, Argentina=2.51)
- `lambda_prior=15`: correccion moderada
- `lambda_prior=25`: balanceado — elegido por resultados empiricos
- `lambda_prior=35`: resultados muy similares a 25

A mayor `lambda_prior`, mayor peso al ELO relativo al MLE de goles.

### Parametros calibrados actuales (con ELO, lambda=25)

```
gamma = 1.3391   (ventaja de localia en internacionales)

Fuerzas de confederacion (UEFA=1.0 referencia) — con ELO:
  AFC       1.1665   <- Asia rinde bien vs otros en WC, ELO corrige
  CONMEBOL  1.0205   <- era 1.30 sin ELO (inflado por Argentina WC2022)
  CAF       0.9970
  OFC       0.9068
  CONCACAF  0.8972

Nota: con ELO las conf_strengths se rebalancean porque el ELO ya
captura las diferencias entre confederaciones. El CONMEBOL baja de
1.30 a 1.02 — ya no necesita boostearse sobre el MLE.
```

### Resultados en partidos test (lambda=25, con ELO)

| Partido | Sin ELO | Con ELO | Target |
|---|---|---|---|
| France vs Argentina | 12.9% Fra | **28.3% Fra** | 38-42% |
| Argentina vs Haiti | 66.3% Arg | **68.4% Arg** | 70-75% |
| Brazil vs Germany | 50.7% Bra | **38.9% Bra** ✓ | 35-40% |
| Spain vs France | 56.7% Spa | **40.1% Spa** ✓ | 40-45% |

France vs Argentina no llega al 38-42% porque Argentina ELO (2114) > France ELO (2062) genuinamente — el modelo refleja que Argentina es el actual campeon del mundo con mayor ELO.

### ELO como target del shrinkage en `calcular_lambdas`

Ademas del MAP en la estimacion, el `conf_ctx` incluye:
- `elo_alpha_prior`: dict equipo -> alpha ELO-implied (target individual por equipo)
- `skip_conf_alpha=True`: omite el conf_factor sobre alpha (ELO ya lo captura)

En `calcular_lambdas`, la capa de shrinkage usa el prior ELO de cada equipo como target en vez del mean global, lo que da resultados mas precisos para equipos con pocos datos.

---

## src/prediccion.py

Solo usado por el pipeline CLI (`main.py`). La app Streamlit usa directamente `calcular_lambdas` de `mundial.py`.

| Funcion | Que hace |
|---|---|
| `predecir_partido(local, visitante, fuerzas, gamma, max_goles=8)` | Calcula lambdas Poisson, construye matriz de marcadores, devuelve dict con `lambda_h`, `lambda_a`, `prob_local`, `prob_empate`, `prob_visitante`, `matriz`. Sin ajuste de confederacion. |
| `marcador_mas_probable(matriz)` | Devuelve `(i, j)` del argmax de la matriz. |

---

## src/mundial.py

Modulo central de simulacion.

**Pipeline de ajuste en `calcular_lambdas`** (tres capas):

```
1. Shrinkage adaptativo por n_partidos:
   sf = 0.30 si n>=50, 0.60 si n>=20, 0.85 si n<20
   target = elo_alpha_prior[e]  si hay ELO
          = alpha_media_wc       si no hay ELO
   alpha_adj = (1-sf)*alpha_raw + sf*target

2. Factor de confederacion sobre alpha:
   SOLO si skip_conf_alpha=False (es decir, sin ELO)
   alpha_adj *= conf_strength[confederacion_equipo]
   Con ELO se omite (doble-conteo con el prior).

3. Beta cross-confederacion:
   si atacante viene de conf mas fuerte -> beta del defensor sube
   shrinkage_beta = max(0, s_atacante - s_defensor) * 0.4
```

**`conf_ctx` — diccionario de contexto de confederacion:**
```python
conf_ctx = {
    "strengths":       dict[str, float],  # conf -> s_k (de estimar_selecciones)
    "equipo_conf":     dict[str, str],    # equipo -> confederacion
    "beta_media":      float,             # media beta de 207 equipos
    "alpha_media":     float,             # target fallback si no hay elo_alpha_prior
    "beta_media_wc":   float,             # media beta de los 48 WC (target shrinkage)
    "n_partidos":      dict[str, int],    # equipo -> n partidos en dataset
    "elo_alpha_prior": dict[str, float],  # equipo -> alpha ELO-implied (vacio sin ELO)
    "skip_conf_alpha": bool,              # True con ELO, False sin ELO
}
```

**Bracket FIFA 2026 implementado:**
```
R32: 16 partidos (w73-w88), cruces oficiales segun Annex C FIFA
R16: w89=W74vsW77, w90=W73vsW75, w91=W76vsW78, w92=W79vsW80,
     w93=W83vsW84, w94=W81vsW82, w95=W86vsW88, w96=W85vsW87
QF:  w97=W89vsW90, w98=W93vsW94, w99=W91vsW92, w100=W95vsW96
SF:  w101=W97vsW98, w102=W99vsW100
F:   W101vsW102
```

---

## app.py — Streamlit

Arranca con: `streamlit run app.py`

**`@st.cache_resource cargar_modelo()`** — se ejecuta UNA sola vez al arrancar:
1. `cargar_internacionales()` -> descarga datos internacionales
2. `cargar_elo()` -> descarga ELO ratings de eloratings.net
3. `estimar_selecciones(df_int, elo_series=elo, lambda_prior=25)` -> MAP + conf (~20s)
4. Computa `elo_alpha_prior`, `beta_media_wc`, `n_partidos`
5. Construye `conf_ctx` con `skip_conf_alpha=True`
6. Devuelve `(fuerzas, gamma, conf_strengths, conf_ctx, media)`

**`lambdas_neutral(eq1, eq2, ...)`** — calculo correcto de sede neutral:
```python
lh1, la1 = calcular_lambdas(eq1, eq2, ...)  # eq1 como local
lh2, la2 = calcular_lambdas(eq2, eq1, ...)  # eq2 como local
return (lh1 + la2) / 2, (la1 + lh2) / 2   # goles propios de cada equipo
```

**Tres secciones:**
1. **Predictor de partido** — selectbox 48 equipos, metricas 1X2, lambdas, top-5 tabla, heatmap `px.imshow` + bar chart top-10 `px.bar`
2. **Simulacion del torneo** — slider 1k-50k, `monte_carlo()` con spinner, bar chart top-10, tabla, `st.download_button` CSV
3. **Sobre el modelo** — explicacion, metricas de validacion, fuentes

---

## validar.py — Validacion retrospectiva WC2022

Corrida manualmente con `python validar.py`. No se carga en la app.

**Training (sin data leakage):** WC2018 qualifiers 2015-2017 + WC2018 = ~1134 partidos

**Resultados sobre WC2022 (64 partidos):**
```
Modelo Poisson basico:  accuracy=54.7%  Brier=0.6108  LogLoss=1.042
Mercado (Pinnacle avg): accuracy=53.1%  Brier=0.5836  LogLoss=0.998
Uniforme (1/3,1/3,1/3): accuracy=45.3%  Brier=0.6667

Por fase:
  Grupos (48):       Poisson 52.1%  Mercado 52.1%  (empate)
  Eliminatoria (16): Poisson 62.5%  Mercado 56.2%  (modelo supera al mercado)

IC 95%: +/-12.2% (n=64)
```

**Caveat critico:** training pre-WC2022 sin clasificatorias 2019-2022 (gap de 4 anos). El modelo actual con datos 2023-2026 deberia ser significativamente mejor.

---

## Problemas conocidos y limitaciones

### France vs Argentina: 28.3% para France (target 38-42%)
France ELO (2062) < Argentina ELO (2114). El modelo correctamente refleja que Argentina es el campeon actual con mayor ELO. Para que France llegue al 40%, necesitariamos un lambda_prior mucho mas alto (>50) que distorsionaria el resto del modelo.

Nota: antes del prior ELO, France tenia 12.9%. El ELO lo sube a 28.3% — mejora sustancial aunque no llega al target aspiracional.

### Teams con pocos partidos (sf=0.85)
~60% de los equipos del WC tienen <20 partidos en el dataset (sf=0.85 — 85% hacia el prior ELO o la media). Incluye Germany (12), Spain (14), Portugal (15), Norway (8), New Zealand (5), Mexico (7), USA (4), Canada (3).

### England domina el MC (10.4%)
England tiene ELO alto (2021), grupo favorable (L: England, Croatia, Ghana, Panama) y una bracket path manejable. Estadisticamente correcto pero puede sorprender visualmente.

### ELO disponibilidad
eloratings.net es un sitio externo. Si falla el scraping, el modelo cae back a MLE puro (sin prior). El cache en `data/elo_selecciones.csv` cubre el caso de scraping intermitente.

---

## Convenciones del proyecto

### Correr localmente
```bash
python main.py        # pipeline completo
python validar.py     # validacion retrospectiva
streamlit run app.py  # app web
```

### Que NO subir al repo
```
data/*.csv    <- descargados automaticamente al iniciar
data/*.xlsx   <- descargados automaticamente al iniciar
__pycache__/  *.pyc  .env
```

### Convenciones de commits
- `feat:` nueva funcionalidad | `fix:` bug | `refactor:` interno | `docs:` documentacion
- Nunca hacer commit de .csv o .xlsx
- Incluir output numerico en el mensaje cuando aplica

### Python 3.10+ requerido
Usa `X | Y` para type unions y `tuple[float, float]` sin `from __future__ import annotations`.

---

## Proximos pasos acordados

### Pendientes inmediatos
1. **France vs Argentina calibracion** — el gap de 28% vs target 38-42% es la limitacion restante. Explorar: (a) lambda_prior mas alto, (b) Nations League + amistosos FIFA en el training set.
2. **Validacion con ELO** — no hay una validacion formal del modelo CON prior ELO sobre WC2022. Requeriria ELO historico de 2022.
3. **Monitorear durante WC2026** — comparar predicciones del modelo vs resultados reales en tiempo real.

### Post-validacion Streamlit — Migracion FastAPI + React
- Backend: `FastAPI` con endpoints `POST /predecir` y `POST /simular`
- Frontend: React con recharts para los graficos
- El core src/*.py no cambia — FastAPI lo importa directamente
- Hosting: Railway/Render (FastAPI), Vercel/Netlify (React)

### Ideas futuras (no comprometidas)
- Datos Nations League y amistosos FIFA (mejora el training set significativamente)
- Modelo bayesiano jerarquico con prior ELO como hiperparametro
- Calibracion separada fase de grupos vs eliminatoria
- Dashboard de monitoring con resultados reales del WC2026

---

## Fuentes de datos

| Dataset | URL | Que contiene |
|---|---|---|
| Club leagues | `football-data.co.uk/mmz4281/{season}/{liga}.csv` | Premier + La Liga, 21/22-23/24, 2280 partidos |
| International | `football-data.co.uk/WorldCup2026.xlsx` | WC2018 (64), WC2022 (64), Qualifiers 2026 (889) |
| Old qualifiers | `football-data.co.uk/internationals.xlsx` | WC2018 qualifiers (860) + torneos 2014-2017 — solo en `validar.py` |
| ELO ratings | `eloratings.net/World.tsv` | ELO actual de 244 selecciones nacionales |

**Grupos FIFA 2026:** sorteo oficial del 5 de diciembre de 2025 (hardcoded en `GRUPOS`).
**Bracket:** Annex C del reglamento FIFA 2026 (backtracking bipartite matching para terceros).
