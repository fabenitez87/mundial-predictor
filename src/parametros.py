import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

XI_CLUBES      = 0.0018
XI_SELECCIONES = 0.004

# ── Confederaciones ────────────────────────────────────────────────────────
# Índice 0 = UEFA (referencia fija en 1.0); 1-5 son libres en la optimización
CONFS = ["UEFA", "CONMEBOL", "CONCACAF", "AFC", "CAF", "OFC"]
CONF_IDX = {c: i for i, c in enumerate(CONFS)}

# Mapeo completo: todos los equipos del dataset internacional → confederación
SELECCION_CONFEDERACION: dict[str, str] = {
    # ── UEFA ──────────────────────────────────────────────────────────────
    "Albania": "UEFA", "Andorra": "UEFA", "Armenia": "UEFA", "Austria": "UEFA",
    "Azerbaijan": "UEFA", "Belarus": "UEFA", "Belgium": "UEFA",
    "Bosnia and Herzegovina": "UEFA", "Bulgaria": "UEFA", "Croatia": "UEFA",
    "Cyprus": "UEFA", "Czechia": "UEFA", "Denmark": "UEFA", "England": "UEFA",
    "Estonia": "UEFA", "Faroe Islands": "UEFA", "Finland": "UEFA", "France": "UEFA",
    "Georgia": "UEFA", "Germany": "UEFA", "Gibraltar": "UEFA", "Greece": "UEFA",
    "Hungary": "UEFA", "Iceland": "UEFA", "Ireland": "UEFA", "Israel": "UEFA",
    "Italy": "UEFA", "Kazakhstan": "UEFA", "Kosovo": "UEFA", "Latvia": "UEFA",
    "Liechtenstein": "UEFA", "Lithuania": "UEFA", "Luxembourg": "UEFA",
    "Malta": "UEFA", "Moldova": "UEFA", "Montenegro": "UEFA",
    "Netherlands": "UEFA", "North Macedonia": "UEFA", "Northern Ireland": "UEFA",
    "Norway": "UEFA", "Poland": "UEFA", "Portugal": "UEFA", "Romania": "UEFA",
    "Russia": "UEFA", "San Marino": "UEFA", "Scotland": "UEFA", "Serbia": "UEFA",
    "Slovakia": "UEFA", "Slovenia": "UEFA", "Spain": "UEFA", "Sweden": "UEFA",
    "Switzerland": "UEFA", "Turkey": "UEFA", "Ukraine": "UEFA", "Wales": "UEFA",
    # ── CONMEBOL ─────────────────────────────────────────────────────────
    "Argentina": "CONMEBOL", "Bolivia": "CONMEBOL", "Brazil": "CONMEBOL",
    "Chile": "CONMEBOL", "Colombia": "CONMEBOL", "Ecuador": "CONMEBOL",
    "Paraguay": "CONMEBOL", "Peru": "CONMEBOL", "Uruguay": "CONMEBOL",
    "Venezuela": "CONMEBOL",
    # ── CONCACAF ─────────────────────────────────────────────────────────
    "Anguilla": "CONCACAF", "Antigua and Barbuda": "CONCACAF", "Aruba": "CONCACAF",
    "Bahamas": "CONCACAF", "Barbados": "CONCACAF", "Belize": "CONCACAF",
    "Bermuda": "CONCACAF", "British Virgin Islands": "CONCACAF",
    "Canada": "CONCACAF", "Cayman Islands": "CONCACAF", "Costa Rica": "CONCACAF",
    "Cuba": "CONCACAF", "Curacao": "CONCACAF", "Dominica": "CONCACAF",
    "Dominican Republic": "CONCACAF", "El Salvador": "CONCACAF",
    "Grenada": "CONCACAF", "Guatemala": "CONCACAF", "Guyana": "CONCACAF",
    "Haiti": "CONCACAF", "Honduras": "CONCACAF", "Jamaica": "CONCACAF",
    "Mexico": "CONCACAF", "Montserrat": "CONCACAF", "Nicaragua": "CONCACAF",
    "Panama": "CONCACAF", "Puerto Rico": "CONCACAF",
    "Saint Kitts and Nevis": "CONCACAF", "Saint Lucia": "CONCACAF",
    "Saint Vincent and the Grenadines": "CONCACAF", "Suriname": "CONCACAF",
    "Trinidad & Tobago": "CONCACAF", "USA": "CONCACAF",
    # ── AFC ──────────────────────────────────────────────────────────────
    "Afghanistan": "AFC", "Australia": "AFC", "Bahrain": "AFC",
    "Bangladesh": "AFC", "Bhutan": "AFC", "Brunei": "AFC", "Cambodia": "AFC",
    "China": "AFC", "Chinese Taipei": "AFC", "Guam": "AFC", "Hong Kong": "AFC",
    "India": "AFC", "Indonesia": "AFC", "Iran": "AFC", "Iraq": "AFC",
    "Japan": "AFC", "Jordan": "AFC", "Kuwait": "AFC", "Kyrgyzstan": "AFC",
    "Laos": "AFC", "Lebanon": "AFC", "Macau": "AFC", "Malaysia": "AFC",
    "Maldives": "AFC", "Mongolia": "AFC", "Myanmar": "AFC", "Nepal": "AFC",
    "North Korea": "AFC", "Oman": "AFC", "Pakistan": "AFC", "Palestine": "AFC",
    "Philippines": "AFC", "Qatar": "AFC", "Saudi Arabia": "AFC",
    "Singapore": "AFC", "South Korea": "AFC", "Sri Lanka": "AFC",
    "Syria": "AFC", "Tajikistan": "AFC", "Thailand": "AFC",
    "Timor-Leste": "AFC", "Turkmenistan": "AFC", "United Arab Emirates": "AFC",
    "Uzbekistan": "AFC", "Vietnam": "AFC", "Yemen": "AFC",
    # ── CAF ──────────────────────────────────────────────────────────────
    "Algeria": "CAF", "Angola": "CAF", "Benin": "CAF", "Botswana": "CAF",
    "Burkina Faso": "CAF", "Burundi": "CAF", "Cameroon": "CAF",
    "Cape Verde": "CAF", "Central Africa": "CAF", "Chad": "CAF",
    "Comoros": "CAF", "Congo": "CAF", "DR Congo": "CAF", "Djibouti": "CAF",
    "Egypt": "CAF", "Equatorial Guinea": "CAF", "Eswatini": "CAF",
    "Ethiopia": "CAF", "Gabon": "CAF", "Gambia": "CAF", "Ghana": "CAF",
    "Guinea": "CAF", "Guinea Bissau": "CAF", "Ivory Coast": "CAF",
    "Kenya": "CAF", "Lesotho": "CAF", "Liberia": "CAF", "Libya": "CAF",
    "Madagascar": "CAF", "Malawi": "CAF", "Mali": "CAF", "Mauritania": "CAF",
    "Mauritius": "CAF", "Morocco": "CAF", "Mozambique": "CAF",
    "Namibia": "CAF", "Niger": "CAF", "Nigeria": "CAF", "Rwanda": "CAF",
    "Sao Tome and Principe": "CAF", "Senegal": "CAF", "Seychelles": "CAF",
    "Sierra Leone": "CAF", "Somalia": "CAF", "South Africa": "CAF",
    "South Sudan": "CAF", "Sudan": "CAF", "Tanzania": "CAF", "Togo": "CAF",
    "Tunisia": "CAF", "Uganda": "CAF", "Zambia": "CAF", "Zimbabwe": "CAF",
    # ── OFC ──────────────────────────────────────────────────────────────
    "Cook Islands": "OFC", "Fiji": "OFC", "New Caledonia": "OFC",
    "New Zealand": "OFC", "Papua New Guinea": "OFC", "Samoa": "OFC",
    "Solomon Islands": "OFC", "Tahiti": "OFC", "Tonga": "OFC",
    "Vanuatu": "OFC",
}


# ── Utilidades compartidas ─────────────────────────────────────────────────

def escalar_elo_a_alpha(elo_val: float, elo_mean: float, elo_std: float,
                        alpha_mean: float = 1.88,
                        alpha_std: float = 0.50) -> float:
    """
    Escala un ELO rating al espacio de alpha del modelo Poisson.

    Usa z-score del ELO para mapear linealmente en log-espacio:
      z     = (elo - elo_mean) / elo_std
      k     = log(alpha_mean + alpha_std) - log(alpha_mean)
      alpha = exp(log(alpha_mean) + k * z)

    Propiedades:
      elo = elo_mean  → alpha = alpha_mean  (equipo promedio)
      elo = elo_mean + elo_std → alpha = alpha_mean + alpha_std
      elo = elo_mean - elo_std → alpha = alpha_mean - alpha_std (aprox)
    """
    z = (elo_val - elo_mean) / elo_std if elo_std > 0 else 0.0
    k = np.log(alpha_mean + alpha_std) - np.log(alpha_mean)
    return float(np.exp(np.log(alpha_mean) + k * z))


def _pesos_temporales(df: pd.DataFrame,
                      fecha_ref: pd.Timestamp | None = None,
                      xi: float = XI_CLUBES) -> np.ndarray:
    if fecha_ref is None:
        fecha_ref = pd.Timestamp.today()
    if "Date" not in df.columns:
        return np.ones(len(df))
    fechas = pd.to_datetime(df["Date"], errors="coerce")
    dias = (fecha_ref - fechas).dt.days.fillna(0).clip(lower=0).values
    return np.exp(-xi * dias)


def _log_verosimilitud(params: np.ndarray, equipos: list[str],
                       home_idx: np.ndarray, away_idx: np.ndarray,
                       fthg: np.ndarray, ftag: np.ndarray,
                       pesos: np.ndarray) -> float:
    n = len(equipos)
    alpha = np.exp(params[:n])
    beta  = np.exp(params[n:2*n])
    gamma = np.exp(params[2*n])

    lambda_h = gamma * alpha[home_idx] * beta[away_idx]
    lambda_a = alpha[away_idx] * beta[home_idx]

    ll = pesos * (
        fthg * np.log(lambda_h + 1e-10) - lambda_h - gammaln(fthg + 1) +
        ftag * np.log(lambda_a + 1e-10) - lambda_a - gammaln(ftag + 1)
    )
    return -ll.sum()


# ── Estimación sin ajuste confederación (ligas de clubes) ─────────────────

def estimar_parametros(df: pd.DataFrame,
                       xi: float = XI_CLUBES) -> tuple[dict, float]:
    """MLE L-BFGS-B estándar. Retorna (fuerzas, gamma)."""
    equipos = sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))
    idx = {e: i for i, e in enumerate(equipos)}
    n = len(equipos)

    home_idx = np.array([idx[e] for e in df["HomeTeam"]])
    away_idx = np.array([idx[e] for e in df["AwayTeam"]])
    fthg = df["FTHG"].values.astype(float)
    ftag = df["FTAG"].values.astype(float)
    pesos = _pesos_temporales(df, xi=xi)

    resultado = minimize(
        _log_verosimilitud,
        np.zeros(2 * n + 1),
        args=(equipos, home_idx, away_idx, fthg, ftag, pesos),
        method="L-BFGS-B",
        options={"maxiter": 2000, "ftol": 1e-9},
    )
    params = resultado.x
    alpha = np.exp(params[:n])
    beta  = np.exp(params[n:2*n])
    gamma = float(np.exp(params[2*n]))

    fuerzas = {e: (float(alpha[i]), float(beta[i])) for i, e in enumerate(equipos)}
    return fuerzas, gamma


# ── Estimación con ajuste por confederación (selecciones) ─────────────────

def estimar_selecciones(df: pd.DataFrame,
                        elo_series: "pd.Series | None" = None,
                        lambda_prior: float = 25.0) -> tuple[dict, float, dict]:
    """
    MLE L-BFGS-B con decay xi=0.004 + 5 parámetros de confederación.

    Si elo_series no es None, usa MAP estimation con prior ELO:
      penalidad = lambda_prior * sum((log_alpha_i - log_alpha_prior_i)^2)
    donde log_alpha_prior_i se deriva del ELO rating via escalar_elo_a_alpha().

    Esto ancla los alphas a la fortaleza real de cada seleccion, corrigiendo
    el sesgo que tiene el MLE puro para equipos con pocos datos o clasificatorias
    de nivel desigual (ej: France 2062 ELO calibra mejor que alpha_raw=1.02).

    Normalización post-estimación:
      α_norm = α · s_conf
      β_norm = β / s_conf

    Retorna (fuerzas_normalizadas, gamma, conf_strengths).
    """
    xi = XI_SELECCIONES
    equipos = sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))
    idx = {e: i for i, e in enumerate(equipos)}
    n = len(equipos)

    # Índice de confederación por equipo (default UEFA=0 para desconocidos)
    conf_of = np.array([
        CONF_IDX.get(SELECCION_CONFEDERACION.get(e, "UEFA"), 0)
        for e in equipos
    ])

    home_idx  = np.array([idx[e] for e in df["HomeTeam"]])
    away_idx  = np.array([idx[e] for e in df["AwayTeam"]])
    home_conf = conf_of[home_idx]
    away_conf = conf_of[away_idx]
    fthg  = df["FTHG"].values.astype(float)
    ftag  = df["FTAG"].values.astype(float)
    pesos = _pesos_temporales(df, xi=xi)

    # Vector de parámetros: [log_alpha×n | log_beta×n | log_gamma | log_s×5]
    n_conf_free = len(CONFS) - 1  # 5

    # ── Prior ELO para MAP estimation ──────────────────────────────────────
    log_alpha_prior: np.ndarray | None = None
    if elo_series is not None:
        elo_vals = np.array([
            float(elo_series.get(e, elo_series.mean())) for e in equipos
        ])
        elo_mean = float(elo_vals.mean())
        elo_std  = float(elo_vals.std())
        alpha_priors = np.array([
            escalar_elo_a_alpha(v, elo_mean, elo_std) for v in elo_vals
        ])
        log_alpha_prior = np.log(alpha_priors)

    def objetivo(params: np.ndarray) -> float:
        log_alpha  = params[:n]
        log_beta   = params[n:2*n]
        log_gamma  = params[2*n]
        log_s_free = params[2*n+1:]

        log_s = np.empty(len(CONFS))
        log_s[0] = 0.0
        log_s[1:] = log_s_free

        alpha = np.exp(log_alpha)
        beta  = np.exp(log_beta)
        gamma = np.exp(log_gamma)
        s     = np.exp(log_s)

        s_h = s[home_conf]
        s_a = s[away_conf]

        lambda_h = gamma * alpha[home_idx] * s_h * beta[away_idx] / s_a
        lambda_a = alpha[away_idx] * s_a * beta[home_idx] / s_h

        ll = pesos * (
            fthg * np.log(lambda_h + 1e-10) - lambda_h - gammaln(fthg + 1) +
            ftag * np.log(lambda_a + 1e-10) - lambda_a - gammaln(ftag + 1)
        )
        # L2 sobre parámetros libres de confederación (evita divergencia con poco cross-conf)
        reg = 1.0 * np.sum(log_s_free ** 2)

        # MAP: penalidad L2 hacia prior ELO
        # F(θ) = -logL + λ_conf·||log_s||² + λ_prior·||log_α - log_α_prior||²
        map_pen = 0.0
        if log_alpha_prior is not None:
            map_pen = lambda_prior * np.sum((log_alpha - log_alpha_prior) ** 2)

        return -ll.sum() + reg + map_pen

    # Inicializar en el prior ELO mejora convergencia significativamente
    x0 = np.zeros(2 * n + 1 + n_conf_free)
    if log_alpha_prior is not None:
        x0[:n] = log_alpha_prior
    resultado = minimize(
        objetivo,
        x0,
        method="L-BFGS-B",
        options={"maxiter": 3000, "ftol": 1e-9},
    )

    params = resultado.x
    alpha      = np.exp(params[:n])
    beta       = np.exp(params[n:2*n])
    gamma      = float(np.exp(params[2*n]))
    log_s_free = params[2*n+1:]

    log_s    = np.zeros(len(CONFS))
    log_s[1:] = log_s_free
    s = np.exp(log_s)

    # Normalizar a escala global: α_norm=α·s, β_norm=β/s
    fuerzas: dict[str, tuple[float, float]] = {}
    for i, equipo in enumerate(equipos):
        c = conf_of[i]
        fuerzas[equipo] = (float(alpha[i] * s[c]), float(beta[i] / s[c]))

    conf_strengths = {CONFS[i]: float(s[i]) for i in range(len(CONFS))}
    return fuerzas, gamma, conf_strengths
