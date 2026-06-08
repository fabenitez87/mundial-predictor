"""
Modulo 2 + 3: Simulacion Monte Carlo del Mundial 2026.
Formato real: 12 grupos de 4 equipos, Ronda de 32, R16, Cuartos, Semis, Final.
Bracket oficial segun FIFA (Annex C del reglamento de competicion).
Calibracion de selecciones con datos reales de partidos internacionales.
"""
import numpy as np
import pandas as pd
from itertools import combinations

# Aliases para nombres alternativos en los datos de calibracion
# clave: nombre en GRUPOS -> aliases posibles en fuerzas_selecciones
_ALIASES: dict[str, list[str]] = {
    "Bosnia and Herzegovina": ["Bosnia & Herzegovina", "Bosnia-Herzegovina"],
    "DR Congo":               ["D.R. Congo", "Congo DR", "Congo"],
    "Czechia":                ["Czech Republic", "Czechia"],
    "Ivory Coast":            ["Cote d'Ivoire", "Ivory Coast"],
    "South Korea":            ["Korea Republic", "South Korea"],
    "USA":                    ["United States", "USA"],
}

# ── Grupos oficiales FIFA 2026 (sorteo 5-dic-2025) ────────────────────────
GRUPOS: dict[str, list[str]] = {
    "A": ["Mexico",      "South Africa",          "South Korea",  "Czechia"],
    "B": ["Canada",      "Switzerland",            "Qatar",        "Bosnia and Herzegovina"],
    "C": ["Brazil",      "Morocco",                "Haiti",        "Scotland"],
    "D": ["USA",         "Paraguay",               "Australia",    "Turkey"],
    "E": ["Germany",     "Curacao",                "Ivory Coast",  "Ecuador"],
    "F": ["Netherlands", "Japan",                  "Sweden",       "Tunisia"],
    "G": ["Belgium",     "Egypt",                  "Iran",         "New Zealand"],
    "H": ["Spain",       "Cape Verde",             "Saudi Arabia", "Uruguay"],
    "I": ["France",      "Senegal",                "Norway",       "Iraq"],
    "J": ["Argentina",   "Algeria",                "Austria",      "Jordan"],
    "K": ["Portugal",    "DR Congo",               "Uzbekistan",   "Colombia"],
    "L": ["England",     "Croatia",                "Ghana",        "Panama"],
}

# ── Slots del bracket para los 8 mejores terceros (Ronda de 32) ───────────
# match_id -> grupos de los que puede provenir el tercer clasificado
TERCEROS_SLOTS: dict[int, frozenset] = {
    74: frozenset("ABCDF"),
    77: frozenset("CDFGH"),
    79: frozenset("CEFHI"),
    80: frozenset("EHIJK"),
    81: frozenset("BEFIJ"),
    82: frozenset("AEHIJ"),
    85: frozenset("EFGIJ"),
    87: frozenset("DEIJL"),
}


# ── Fuerzas de respaldo ────────────────────────────────────────────────────

def _media_fuerzas(fuerzas: dict) -> tuple[float, float]:
    """Media global de ataque y defensa: fallback para equipos sin datos."""
    alphas = [v[0] for v in fuerzas.values()]
    betas  = [v[1] for v in fuerzas.values()]
    return float(np.mean(alphas)), float(np.mean(betas))


def _lookup(equipo: str, fuerzas: dict, media: tuple[float, float]) -> tuple[float, float]:
    """
    Busca las fuerzas de un equipo en el dict.
    Si no está por nombre exacto, prueba aliases conocidos.
    Si tampoco, devuelve la media global.
    """
    if equipo in fuerzas:
        return fuerzas[equipo]
    for alias in _ALIASES.get(equipo, []):
        if alias in fuerzas:
            return fuerzas[alias]
    return media


def _ajustar_beta(beta: float, s_atacante: float, s_defensor: float,
                  beta_media: float) -> float:
    """
    Corrección de regresión a la media para el beta defensivo.

    Si el rival atacante es de confederación más fuerte que el defensor,
    el beta (debilidad defensiva) se encoge hacia la media global:

        shrinkage = max(0, s_rival - s_equipo) * 0.4
        beta_adj  = beta * (1 - shrinkage) + beta_media * shrinkage

    Intuición: una defensa que se forjó contra rivales débiles (AFC bajo)
    no puede rendir igual contra atacantes de mayor nivel (CONMEBOL/UEFA).
    """
    shrinkage = max(0.0, s_atacante - s_defensor) * 0.4
    if shrinkage == 0.0:
        return beta
    return beta * (1.0 - shrinkage) + beta_media * shrinkage


def calcular_lambdas(local: str, visitante: str, fuerzas: dict, gamma: float,
                     media: tuple[float, float],
                     conf_ctx: dict | None = None) -> tuple[float, float]:
    """
    Retorna (lambda_h, lambda_a) para análisis y tests.
    Aplica dos correcciones si conf_ctx está presente:
      1. Beta shrinkage  — la defensa regresa a la media cuando el atacante
                           viene de una confederación más fuerte.
      2. Alpha shrinkage — el ataque de TODOS los equipos se encoge hacia la
                           media WC (alpha_media), con factor alpha_shrinkage.
                           Corrección para alphas inflados por clasificatorias
                           débiles (ej. Senegal 3.78 en CAF vs France 1.02 en UEFA).
    """
    ah, bh = _lookup(local,    fuerzas, media)
    aa, ba = _lookup(visitante, fuerzas, media)

    if conf_ctx is not None:
        # ── 1. Alpha shrinkage global (regresion a la media WC) ────────────
        # Corrige alphas inflados por clasificatorias debiles
        # (ej: Senegal 3.78 en CAF vs France 1.02 en UEFA competitiva).
        # alpha_adj = (1-sf)*alpha + sf*alpha_media
        if "alpha_shrinkage" in conf_ctx:
            am = conf_ctx["alpha_media"]      # media alpha de los 48 equipos del WC
            sf = conf_ctx["alpha_shrinkage"]  # factor: 0=sin cambio, 1=todo a la media
            ah = (1.0 - sf) * ah + sf * am
            aa = (1.0 - sf) * aa + sf * am

        # ── 2. Beta shrinkage global (regresion a la media WC) ─────────────
        # Corrige betas extremos: Ivory Coast beta=0.037, Ecuador beta=0.075
        # calibrados en clasificatorias donde nadie les anota.
        # beta_adj = (1-bs)*beta + bs*beta_media_wc
        if "beta_shrinkage" in conf_ctx:
            bm_wc = conf_ctx["beta_media_wc"]  # media beta de los 48 equipos del WC
            bs    = conf_ctx["beta_shrinkage"]
            ba = (1.0 - bs) * ba + bs * bm_wc
            bh = (1.0 - bs) * bh + bs * bm_wc

        # ── 3. Beta shrinkage por confederacion (contexto cross-conf) ───────
        # Cuando el atacante viene de una confederacion mas fuerte, la defensa
        # del rival se debilita adicionalmente (encogimiento hacia beta_media global).
        eq_conf   = conf_ctx["equipo_conf"]
        strengths = conf_ctx["strengths"]
        bm_global = conf_ctx["beta_media"]   # media global (207 equipos)
        s_h = strengths.get(eq_conf.get(local,    "UEFA"), 1.0)
        s_a = strengths.get(eq_conf.get(visitante, "UEFA"), 1.0)
        ba = _ajustar_beta(ba, s_h, s_a, bm_global)
        bh = _ajustar_beta(bh, s_a, s_h, bm_global)

    return gamma * ah * ba, aa * bh


# ── Simulacion de partido ──────────────────────────────────────────────────

def simular_partido(local: str, visitante: str, fuerzas: dict, gamma: float,
                    media: tuple[float, float], fase_grupos: bool = True,
                    conf_ctx: dict | None = None):
    """
    Fase de grupos: retorna (goles_local, goles_visitante).
    Fase eliminatoria: retorna el equipo ganador.
      Empate -> penales con leve ventaja al equipo con mayor fuerza atacante.

    conf_ctx (opcional): activa la corrección de shrinkage defensivo por conf.
      Requiere keys: 'strengths', 'equipo_conf', 'beta_media'.
    """
    ma, md = media
    ah, bh = _lookup(local,    fuerzas, media)
    aa, ba = _lookup(visitante, fuerzas, media)

    lambda_h, lambda_a = calcular_lambdas(local, visitante, fuerzas, gamma,
                                          media, conf_ctx=conf_ctx)

    gh = np.random.poisson(lambda_h)
    ga = np.random.poisson(lambda_a)

    if fase_grupos:
        return gh, ga

    if gh != ga:
        return local if gh > ga else visitante

    # Penales: ventaja leve al equipo con mayor ataque (ah/aa sin ajuste de conf)
    total = ah + aa
    p = 0.5 + (0.1 * (ah - aa) / total if total > 0 else 0.0)
    p = max(0.2, min(0.8, p))
    return local if np.random.random() < p else visitante


# ── Simulacion de grupo ────────────────────────────────────────────────────

def simular_grupo(equipos: list[str], fuerzas: dict, gamma: float,
                  media: tuple[float, float],
                  conf_ctx: dict | None = None) -> tuple[list[str], dict]:
    """
    6 partidos de ida simple.
    tabla[equipo] = [pts, gf, gc]
    Desempate: pts -> dif. goles -> goles a favor.
    Retorna (clasificacion_ordenada, tabla).
    """
    tabla: dict[str, list[int]] = {e: [0, 0, 0] for e in equipos}

    for eq1, eq2 in combinations(equipos, 2):
        # Sede neutral: sorteo de quien es "local" en el papel
        local, visitante = (eq1, eq2) if np.random.random() < 0.5 else (eq2, eq1)
        gh, ga = simular_partido(local, visitante, fuerzas, gamma, media,
                                 conf_ctx=conf_ctx)
        tabla[local][1]    += gh
        tabla[local][2]    += ga
        tabla[visitante][1] += ga
        tabla[visitante][2] += gh

        if gh > ga:
            tabla[local][0] += 3
        elif gh < ga:
            tabla[visitante][0] += 3
        else:
            tabla[local][0]    += 1
            tabla[visitante][0] += 1

    clasificacion = sorted(
        equipos,
        key=lambda e: (tabla[e][0], tabla[e][1] - tabla[e][2], tabla[e][1]),
        reverse=True,
    )
    return clasificacion, tabla


# ── Asignacion de terceros al bracket ─────────────────────────────────────

def _asignar_terceros(mejor_8: list[tuple[str, str]],
                      slots: dict[int, frozenset]) -> dict[int, str]:
    """
    Bipartite matching via backtracking (mas restringido primero).
    mejor_8: [(grupo, equipo), ...] ordenado de mejor a peor.
    Retorna {match_id: equipo}.
    """
    grupos_disp: dict[str, str] = dict(mejor_8)

    # Mas restringido primero: slot con menos grupos elegibles disponibles
    slot_ids = sorted(slots, key=lambda m: len(slots[m] & set(grupos_disp)))

    def bt(idx: int, disp: dict[str, str]):
        if idx == len(slot_ids):
            return {}
        mid = slot_ids[idx]
        elegibles = slots[mid] & set(disp)
        for g in sorted(elegibles):
            res = bt(idx + 1, {k: v for k, v in disp.items() if k != g})
            if res is not None:
                return {mid: disp[g], **res}
        return None

    result = bt(0, grupos_disp)
    if result is None:
        # Fallback extremo: no deberia ocurrir con datos FIFA validos
        eq = list(grupos_disp.values())
        np.random.shuffle(eq)
        result = {mid: eq[i % len(eq)] for i, mid in enumerate(slots)}
    return result


# ── Simulacion del torneo completo ────────────────────────────────────────

def simular_torneo(grupos: dict[str, list[str]], fuerzas: dict, gamma: float,
                   media: tuple[float, float],
                   conf_ctx: dict | None = None) -> dict:
    """
    Una iteracion Monte Carlo del Mundial 2026.
    Retorna dict con conjuntos de equipos por etapa:
      cuartos (8), semis (4), final (2), campeon (str).
    """
    # ── Fase de grupos ────────────────────────────────────────────────────
    primeros: dict[str, str] = {}
    segundos: dict[str, str] = {}
    terceros_data: dict[str, tuple] = {}  # grupo -> (pts, dif, gf, equipo)

    for letra, equipos in grupos.items():
        clasif, tabla = simular_grupo(equipos, fuerzas, gamma, media,
                                      conf_ctx=conf_ctx)
        primeros[letra] = clasif[0]
        segundos[letra] = clasif[1]
        e3 = clasif[2]
        t = tabla[e3]
        terceros_data[letra] = (t[0], t[1] - t[2], t[1], e3)

    # Mejores 8 terceros (pts, dif de goles, goles a favor)
    sorted_t = sorted(terceros_data.items(), key=lambda x: x[1], reverse=True)
    mejor_8  = [(g, data[3]) for g, data in sorted_t[:8]]
    asig     = _asignar_terceros(mejor_8, TERCEROS_SLOTS)

    p = primeros
    s = segundos

    def ko(a: str, b: str) -> str:
        # Sede neutral: sorteo de quien es "local"
        if np.random.random() < 0.5:
            a, b = b, a
        return simular_partido(a, b, fuerzas, gamma, media,
                               fase_grupos=False, conf_ctx=conf_ctx)

    # ── Ronda de 32 — bracket oficial FIFA ────────────────────────────────
    w73 = ko(s["A"],  s["B"])
    w74 = ko(p["E"],  asig[74])
    w75 = ko(p["F"],  s["C"])
    w76 = ko(p["C"],  s["F"])
    w77 = ko(p["I"],  asig[77])
    w78 = ko(s["E"],  s["I"])
    w79 = ko(p["A"],  asig[79])
    w80 = ko(p["L"],  asig[80])
    w81 = ko(p["D"],  asig[81])
    w82 = ko(p["G"],  asig[82])
    w83 = ko(s["K"],  s["L"])
    w84 = ko(p["H"],  s["J"])
    w85 = ko(p["B"],  asig[85])
    w86 = ko(p["J"],  s["H"])
    w87 = ko(p["K"],  asig[87])
    w88 = ko(s["D"],  s["G"])

    # ── Ronda de 16 ───────────────────────────────────────────────────────
    w89 = ko(w74, w77)
    w90 = ko(w73, w75)
    w91 = ko(w76, w78)
    w92 = ko(w79, w80)
    w93 = ko(w83, w84)
    w94 = ko(w81, w82)
    w95 = ko(w86, w88)
    w96 = ko(w85, w87)

    # ── Cuartos de final ──────────────────────────────────────────────────
    w97  = ko(w89, w90)
    w98  = ko(w93, w94)
    w99  = ko(w91, w92)
    w100 = ko(w95, w96)

    # ── Semifinales ───────────────────────────────────────────────────────
    w101 = ko(w97,  w98)
    w102 = ko(w99,  w100)

    # ── Final ─────────────────────────────────────────────────────────────
    campeon = ko(w101, w102)

    return {
        # "llego a cuartos" = jugo en QF = los 8 ganadores de R16
        "cuartos": {w89, w90, w91, w92, w93, w94, w95, w96},
        # "llego a semis"   = jugo en SF = los 4 ganadores de QF
        "semis":   {w97, w98, w99, w100},
        # "llego a la final"= jugo la final = los 2 ganadores de SF
        "final":   {w101, w102},
        "campeon": campeon,
    }


# ── Monte Carlo ────────────────────────────────────────────────────────────

def monte_carlo(grupos: dict[str, list[str]], fuerzas: dict, gamma: float,
                n: int = 10_000,
                conf_ctx: dict | None = None) -> pd.DataFrame:
    """
    Corre n simulaciones del torneo completo.
    Retorna DataFrame con probabilidades (%) por etapa para los 48 equipos,
    ordenado de mayor a menor probabilidad de campeon.

    conf_ctx (opcional): activa shrinkage defensivo por confederacion.
    """
    media = _media_fuerzas(fuerzas)
    todos = [e for equipos in grupos.values() for e in equipos]
    # conteos[equipo] = [campeon, final, semis, cuartos]
    conteos: dict[str, list[int]] = {e: [0, 0, 0, 0] for e in todos}

    for _ in range(n):
        res = simular_torneo(grupos, fuerzas, gamma, media, conf_ctx=conf_ctx)
        conteos[res["campeon"]][0] += 1
        for e in res["final"]:
            conteos[e][1] += 1
        for e in res["semis"]:
            conteos[e][2] += 1
        for e in res["cuartos"]:
            conteos[e][3] += 1

    rows = [
        {
            "equipo":      e,
            "campeon_%":   round(100 * conteos[e][0] / n, 2),
            "final_%":     round(100 * conteos[e][1] / n, 2),
            "semifinal_%": round(100 * conteos[e][2] / n, 2),
            "cuartos_%":   round(100 * conteos[e][3] / n, 2),
        }
        for e in todos
    ]

    return (
        pd.DataFrame(rows)
        .sort_values("campeon_%", ascending=False)
        .reset_index(drop=True)
    )
