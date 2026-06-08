# mundial-predictor — CLAUDE.md

Estado del proyecto al 2026-06-08. Leer esto antes de tocar cualquier archivo.

---

## Stack y versiones

```
Python 3.13 (Windows 11)
pandas      2.3.2
numpy       2.3.2
scipy       1.17.1
requests    (última)
streamlit   1.50.0
openpyxl    (última)   ← leer .xlsx de football-data.co.uk
matplotlib  (última)   ← requerido por st.dataframe background_gradient (aunque se usa plotly)
plotly      (última)   ← heatmap px.imshow + bar chart en app.py
```

Instalar con: `pip install -r requirements.txt`

---

## Estructura de archivos

```
mundial-predictor/
├── app.py                  ← Streamlit app (3 secciones)
├── main.py                 ← Pipeline CLI completo para testing
├── validar.py              ← Validación retrospectiva WC2022
├── requirements.txt
├── .gitignore
├── data/
│   ├── .gitkeep            ← directorio rastreado en git, archivos ignorados
│   ├── *.csv               ← ignorado: descargados en runtime
│   └── *.xlsx              ← ignorado: descargados en runtime
└── src/
    ├── __init__.py
    ├── carga_datos.py
    ├── parametros.py
    ├── prediccion.py
    └── mundial.py
```

---

## src/carga_datos.py

**Constantes:**
- `DATA_DIR` — ruta absoluta a `/data/`, creada automáticamente con `os.makedirs`
- `NOMBRE_MAP_INT` — normaliza nombres del xlsx al formato de GRUPOS: `"Bosnia & Herzegovina"→"Bosnia and Herzegovina"`, etc.

**Funciones públicas:**

| Función | Qué hace |
|---|---|
| `cargar_partidos()` | Descarga 6 CSVs de football-data.co.uk (Premier E0, La Liga SP1, temporadas 21/22–23/24). Cachea en `/data/`. Devuelve 2280 filas con HomeTeam, AwayTeam, FTHG, FTAG. |
| `cargar_internacionales()` | Descarga `WorldCup2026.xlsx` de football-data.co.uk. Extrae 3 sheets (WC2018: 64, WC2022: 64, Qualifiers2026: 889 partidos). Filtra desde 2018-01-01. Normaliza nombres. Cachea en `data/internacionales.csv`. Devuelve 1017 filas con columna Date. |

**Nota sobre cache:** ambas funciones solo descargan si el archivo local no existe. Para forzar re-descarga, borrar el archivo en `/data/`.

---

## src/parametros.py

**Constantes:**
- `XI_CLUBES = 0.0018` — decay temporal para datos de clubes
- `XI_SELECCIONES = 0.004` — decay más agresivo para selecciones (el fútbol internacional cambia más rápido)
- `CONFS = ["UEFA","CONMEBOL","CONCACAF","AFC","CAF","OFC"]` — UEFA=índice 0, referencia fija
- `SELECCION_CONFEDERACION` — dict con los 207 equipos del dataset → confederación (construido manualmente, cubre todos los equipos que aparecen en los datos)

**Funciones:**

| Función | Qué hace |
|---|---|
| `_pesos_temporales(df, fecha_ref, xi)` | Pesos exponenciales `exp(-xi*dias)` por fila. Si no hay columna `Date`, devuelve pesos uniformes. |
| `estimar_parametros(df, xi=XI_CLUBES)` | MLE L-BFGS-B estándar sobre `df`. Parámetros en log-escala, 2n+1 variables. Devuelve `(fuerzas, gamma)` donde `fuerzas[equipo] = (alpha, beta)`. Usado para clubes. |
| `estimar_selecciones(df)` | MLE extendido con 5 parámetros libres de fuerza de confederación (UEFA=1.0 fijo). Modelo: `λ_h = γ·α_i·s_k·β_j/s_l`. Normaliza post-estimación: `α_norm = α·s_conf`, `β_norm = β/s_conf`. Regularización L2 (coef=1.0) sobre los parámetros de confederación para evitar divergencia con pocos datos cross-conf. Devuelve `(fuerzas_normalizadas, gamma, conf_strengths)`. |

**Parámetros calibrados actuales** (sobre 1017 partidos internacionales 2018-2026):
```
gamma = 1.3791   (ventaja de localía en internacionales)

Fuerzas de confederación (UEFA=1.0 referencia):
  CONMEBOL  1.3031   ← boosted por Argentina WC2022 + CONMEBOL qualifier data
  CAF       0.9921   ← ~igual a UEFA (inflado por Morocco semifinal WC2022)
  OFC       0.9565
  AFC       0.9196
  CONCACAF  0.7938
```

---

## src/prediccion.py

Solo usado por el pipeline CLI (`main.py`). La app Streamlit usa directamente `calcular_lambdas` de `mundial.py`.

| Función | Qué hace |
|---|---|
| `predecir_partido(local, visitante, fuerzas, gamma, max_goles=8)` | Calcula lambdas Poisson, construye matriz de marcadores (max_goles+1)×(max_goles+1), devuelve dict con `lambda_h`, `lambda_a`, `prob_local`, `prob_empate`, `prob_visitante`, `matriz`. Sin ajuste de confederación. |
| `marcador_mas_probable(matriz)` | Devuelve `(i, j)` del argmax de la matriz. |

---

## src/mundial.py

Módulo central de simulación. Todas las funciones relevantes de predicción pasan por aquí.

**Datos hardcoded:**
- `GRUPOS` — 12 grupos oficiales FIFA 2026 (sorteo 5-dic-2025), grupos A-L con 4 equipos cada uno
- `TERCEROS_SLOTS` — dict `match_id → frozenset(grupos_elegibles)` para los 8 terceros según Annex C del reglamento FIFA (495 combinaciones posibles)
- `_ALIASES` — fallback de nombres alternativos (no se usa para los 48 del WC, que todos tienen nombre exacto)

**Pipeline de ajuste en `calcular_lambdas`** (tres capas, aplicadas en orden):

```
1. Shrinkage adaptativo por n_partidos:
   sf = 0.30 si n≥50, 0.60 si n≥20, 0.85 si n<20
   alpha_adj = (1-sf)*alpha_raw + sf*alpha_media_wc
   beta_adj  = (1-sf)*beta_raw  + sf*beta_media_wc

2. Factor de confederación sobre alpha:
   alpha_adj *= conf_strength[confederacion_equipo]
   (Argentina*1.30, France*1.0, Haiti*0.79, Japan*0.92)

3. Beta cross-confederación:
   si atacante viene de conf más fuerte → beta del defensor se encoge hacia beta_media_global
   shrinkage_beta = max(0, s_atacante - s_defensor) * 0.4
```

**Funciones principales:**

| Función | Qué hace |
|---|---|
| `_shrinkage_adaptivo(n)` | Devuelve el factor de shrinkage: 0.30/0.60/0.85 según n_partidos. |
| `_ajustar_beta(beta, s_atacante, s_defensor, beta_media)` | Capa 3 del pipeline: encoge beta defensivo cuando el atacante es de conf más fuerte. |
| `calcular_lambdas(local, visitante, fuerzas, gamma, media, conf_ctx)` | Núcleo del modelo. Devuelve `(lambda_h, lambda_a)` aplicando las 3 capas si `conf_ctx` está presente. |
| `simular_partido(local, visitante, ..., fase_grupos, conf_ctx)` | Muestrea Poisson sobre `calcular_lambdas`. En fase de grupos devuelve `(gh, ga)`; en KO devuelve el ganador (con penales si empate). Penales: 50/50 con leve ventaja al de mayor ataque raw (sin ajuste de conf). |
| `simular_grupo(equipos, ..., conf_ctx)` | 6 partidos de ida simple. Coin flip 50/50 para localía (sede neutral). Desempate: pts → dif_goles → goles_favor. |
| `_asignar_terceros(mejor_8, slots)` | Backtracking bipartite matching para asignar los 8 mejores terceros a los 8 slots del bracket. Más restringido primero. |
| `simular_torneo(grupos, fuerzas, gamma, media, conf_ctx)` | Una iteración MC completa: grupos → R32 (16 partidos) → R16 → QF → SF → Final. Devuelve `{cuartos, semis, final, campeon}`. |
| `monte_carlo(grupos, fuerzas, gamma, n, conf_ctx)` | Corre `n` iteraciones de `simular_torneo`. Devuelve DataFrame 48×5 con `equipo, campeon_%, final_%, semifinal_%, cuartos_%`, ordenado por campeon_%. |

**`conf_ctx` — diccionario de contexto de confederación:**
```python
conf_ctx = {
    "strengths":     dict[str, float],  # conf → s_k (de estimar_selecciones)
    "equipo_conf":   dict[str, str],    # equipo → confederación (SELECCION_CONFEDERACION)
    "beta_media":    float,             # media beta de 207 equipos (para ajuste cross-conf)
    "alpha_media":   float,             # media alpha de los 48 WC (target shrinkage)
    "beta_media_wc": float,             # media beta de los 48 WC (target shrinkage)
    "n_partidos":    dict[str, int],    # equipo → n partidos en dataset (determina sf)
}
```

**Bracket FIFA 2026 implementado:**
```
R32: 16 partidos (w73–w88) con cruces oficiales según Annex C
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
1. `cargar_internacionales()` → descarga datos
2. `estimar_selecciones(df_int)` → calibra modelo (~20s)
3. Computa `alpha_media_wc`, `beta_media_wc`, `n_partidos`
4. Construye `conf_ctx`
5. Devuelve `(fuerzas, gamma, conf_strengths, conf_ctx, media)`

**`lambdas_neutral(eq1, eq2, ...)`** — cálculo correcto de sede neutral:
```python
lh1, la1 = calcular_lambdas(eq1, eq2, ...)  # eq1 como local
lh2, la2 = calcular_lambdas(eq2, eq1, ...)  # eq2 como local
return (lh1 + la2) / 2, (la1 + lh2) / 2   # goles propios de cada equipo
```

**Tres secciones en sidebar:**
1. **Predictor de partido** — selectbox 48 equipos, métricas 1X2, lambdas, top-5 tabla, heatmap `px.imshow` + bar chart top-10 `px.bar`
2. **Simulación del torneo** — slider 1k-50k, `monte_carlo()` con spinner, bar chart top-10, tabla completa, `st.download_button` CSV
3. **Sobre el modelo** — explicación, métricas de validación, fuentes

---

## validar.py — Validación retrospectiva WC2022

Corrida manualmente con `python validar.py`. No se carga en la app.

**Metodología** (sin data leakage):
- Training: partidos internacionales anteriores al WC2022 (WC2018 qualifiers 2015-2017 + WC2018 = ~1134 matches)
- Test: 64 partidos del WC2022 con odds de mercado (H-Avg, D-Avg, A-Avg)

**Resultados sobre WC2022:**
```
Modelo Poisson básico:  accuracy=54.7%  Brier=0.6108  LogLoss=1.042
Mercado (Pinnacle avg): accuracy=53.1%  Brier=0.5836  LogLoss=0.998
Uniforme (1/3,1/3,1/3): accuracy=45.3%  Brier=0.6667

Por fase:
  Grupos (48):       Poisson 52.1%  Mercado 52.1%  (empate)
  Eliminatoria (16): Poisson 62.5%  Mercado 56.2%  (modelo supera al mercado)

IC 95% accuracy: ±12.2% (n=64 — alta varianza, interpretar con cautela)
```

**Caveat crítico:** el training pre-WC2022 no tiene clasificatorias 2019-2022. El modelo actual (con datos 2023-2026) debería ser significativamente mejor para WC2026.

---

## Problemas conocidos y limitaciones

### Calibración de France/UEFA teams
France tiene `alpha_raw=1.00` porque juega en UEFA qualifying contra Bélgica, Países Bajos, etc. y no gola demasiado. El modelo la clasifica débil aunque ganó WC2018 y llegó a la final de WC2022.

Con shrinkage adaptativo + conf factor actual:
- `France alpha_adj = 1.52` (sf=0.60 con 20 partidos, × UEFA=1.0)
- `Argentina alpha_adj = 2.51` (sf=0.60 con 29 partidos, × CONMEBOL=1.30)

Resultado: France 12.9% win rate vs Argentina en sede neutral — lejos del 40-45% que sería intuitivo.

**No hay solución dentro del modelo actual sin datos de ranking FIFA o ELO.**

### CONMEBOL sobreestimado
`s_CONMEBOL=1.30` fue aprendido de WC2018+WC2022 donde Argentina ganó el torneo. Con más datos cross-confederación, este valor debería estabilizarse.

### Teams con pocos partidos (sf=0.85)
Los siguientes equipos del WC tienen <20 partidos en el dataset y usan sf=0.85 (85% shrinkage hacia la media):
- Norway (8), Turkey (8), Canada (3), USA (4), New Zealand (5), Scotland (6), Mexico (7)
- Germany (12), Netherlands (13), DR Congo (13), Ghana (13), Panama (13)
- Spain (14), Switzerland (14), Portugal (15), Tunisia (16), Jordan (16), Uzbekistan (16)

Esto significa que para ~60% de los equipos del WC, el modelo confía poco en sus datos individuales y usa principalmente la media WC ajustada por confederación.

### Bracket no determinista para terceros
Los 8 slots de terceros se resuelven con backtracking, que es correcto pero puede ser lento en casos muy restringidos.

---

## Convenciones del proyecto

### Correr localmente
```bash
# Pipeline completo (datos → MLE → predicción → Monte Carlo)
python main.py

# Validación retrospectiva
python validar.py

# App Streamlit
streamlit run app.py
```

### Qué NO subir al repo
```
data/*.csv    ← descargados automáticamente al iniciar
data/*.xlsx   ← descargados automáticamente al iniciar
__pycache__/
*.pyc
.env
```
El directorio `data/` tiene un `.gitkeep` para que git lo rastree vacío.

### Convenciones de commits
- `feat:` — nueva funcionalidad
- `fix:` — bug corregido
- `refactor:` — cambio interno sin cambio de comportamiento
- Nunca hacer commit de archivos `.csv` o `.xlsx`
- Siempre incluir qué cambió en el mensaje de commit (output numérico si aplica)

### Python 3.10+ requerido
El código usa `X | Y` para type unions y `tuple[float, float]` sin `from __future__ import annotations`.

---

## Próximos pasos acordados

### Pendientes inmediatos
1. **France/UEFA calibración** — explorar inicialización con FIFA rankings o ELO como prior para `estimar_selecciones`. Alternativa: agregar partidos de Nations League y amistosos FIFA al dataset.
2. **CONMEBOL s_k** — monitorear si s_CONMEBOL=1.30 se estabiliza o reduce con más datos cross-conf de WC2026.
3. **Validación con n=10000 y CI más ajustados** — actualmente hay alta varianza por n=64 en el test set.

### Post-validación Streamlit — Migración FastAPI + React
Una vez que la app Streamlit esté estable y validada:
- Backend: `FastAPI` exponiendo los endpoints `POST /predecir` y `POST /simular`
- Frontend: React con recharts o similar para los gráficos
- El core del modelo (src/*.py) no cambia — FastAPI lo importa directamente
- Hosting: Railway/Render para FastAPI, Vercel/Netlify para React

### Ideas futuras (no comprometidas)
- Agregar Nations League y amistosos FIFA al training set
- Modelo bayesiano jerárquico con prior ELO
- Calibración separada para "fase de grupo" vs "eliminatoria"
- Histórico de simulaciones guardado en SQLite

---

## Fuentes de datos

| Dataset | URL | Qué contiene |
|---|---|---|
| Club leagues | `football-data.co.uk/mmz4281/{season}/{liga}.csv` | Premier League + La Liga, temporadas 21/22–23/24, 2280 partidos |
| International | `football-data.co.uk/WorldCup2026.xlsx` | WC2018 (64), WC2022 (64), Clasificatorias 2026 (889) — con odds de mercado en WC2022 |
| Old qualifiers | `football-data.co.uk/internationals.xlsx` | WC2018 qualifiers (860) + torneos 2014-2017 (220) — usado solo en `validar.py` |

**Grupos FIFA 2026:** sorteo oficial del 5 de diciembre de 2025 (hardcoded en `GRUPOS`).
**Bracket:** Annex C del reglamento de competición FIFA 2026 (495 combinaciones de terceros — implementadas via backtracking bipartite matching).
