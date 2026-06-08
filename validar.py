"""
Validacion retrospectiva del modelo sobre el Mundial 2022.

Metodologia (sin data leakage):
  Entrenamiento : partidos ANTERIORES al WC2022
    - Previous_Tournaments 2014-2017  (internationals.xlsx)
    - World_Cup_qualifiers 2015-2017  (internationals.xlsx)
    - World Cup 2018                  (WorldCup2026.xlsx, 64 partidos)
  Test          : 64 partidos del World Cup 2022
  Baselines     : uniforme (1/3), favorito del mercado (H-Avg odds),
                  probabilidades del mercado

Metricas: accuracy 1X2, Brier score, log-loss
"""
import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import poisson

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Normalizar nombres a los que usa nuestro modelo
NOMBRE_MAP = {
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "D.R. Congo":           "DR Congo",
    "Czech Republic":       "Czechia",
    "Korea Republic":       "South Korea",
    "IR Iran":              "Iran",
    "United States":        "USA",
}


# ── Carga de datos ─────────────────────────────────────────────────────────

def _cargar_training() -> pd.DataFrame:
    """
    Combina fuentes pre-WC2022 en un DataFrame con:
    Date, HomeTeam, AwayTeam, FTHG, FTAG
    """
    frames = []

    # 1. Datos del viejo internationals.xlsx (pre-2018)
    ruta_int = os.path.join(DATA_DIR, "internacionales.xlsx")
    if os.path.exists(ruta_int):
        sheets = pd.read_excel(ruta_int, sheet_name=None)

        # Previous_Tournaments (2014-2017)
        if "_Previous_Tournaments_data" in sheets:
            s = sheets["_Previous_Tournaments_data"]
            s = s[["Date", "Home", "Away", "HGFT", "AGFT"]].rename(
                columns={"Home": "HomeTeam", "Away": "AwayTeam",
                         "HGFT": "FTHG", "AGFT": "FTAG"}
            )
            frames.append(s)

        # WC2018 qualifiers (2015-2017)
        if "World_Cup_qualifiers" in sheets:
            s = sheets["World_Cup_qualifiers"]
            s = s[["Date", "Home", "Away", "HG", "AG"]].rename(
                columns={"Home": "HomeTeam", "Away": "AwayTeam",
                         "HG": "FTHG", "AG": "FTAG"}
            )
            frames.append(s)

    # 2. WC2018 completo del WorldCup2026.xlsx (64 partidos)
    ruta_wc26 = os.path.join(DATA_DIR, "WorldCup2026.xlsx")
    if os.path.exists(ruta_wc26):
        s = pd.read_excel(ruta_wc26, sheet_name="WorldCup2018")
        s = s[["Date", "Home", "Away", "HGFT", "AGFT"]].rename(
            columns={"Home": "HomeTeam", "Away": "AwayTeam",
                     "HGFT": "FTHG", "AGFT": "FTAG"}
        )
        frames.append(s)

    df = pd.concat(frames, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "FTHG", "FTAG"])
    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)
    df["HomeTeam"] = df["HomeTeam"].replace(NOMBRE_MAP)
    df["AwayTeam"]  = df["AwayTeam"].replace(NOMBRE_MAP)
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def _cargar_wc2022() -> pd.DataFrame:
    """WC2022 con columnas de resultado y odds de mercado."""
    ruta = os.path.join(DATA_DIR, "WorldCup2026.xlsx")
    df = pd.read_excel(ruta, sheet_name="WorldCup2022")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Home"] = df["Home"].replace(NOMBRE_MAP)
    df["Away"] = df["Away"].replace(NOMBRE_MAP)
    df["FTHG"] = df["HGFT"].astype(int)
    df["FTAG"] = df["AGFT"].astype(int)
    return df


# ── Modelo Poisson ─────────────────────────────────────────────────────────

def _estimar_basico(df: pd.DataFrame, xi: float = 0.004):
    """Wraper liviano para no importar el modulo completo."""
    from src.parametros import estimar_parametros
    return estimar_parametros(df, xi=xi)


def _estimar_conf(df: pd.DataFrame):
    """Con ajuste de confederacion."""
    from src.parametros import estimar_selecciones
    return estimar_selecciones(df)     # (fuerzas, gamma, conf_strengths)


def probs_poisson(lambda_h: float, lambda_a: float, max_g: int = 12):
    """Calcula (p_local, p_empate, p_visitante) analiticamente."""
    if lambda_h <= 0 or lambda_a <= 0:
        return 1/3, 1/3, 1/3
    ph = np.array([poisson.pmf(i, lambda_h) for i in range(max_g + 1)])
    pa = np.array([poisson.pmf(j, lambda_a) for j in range(max_g + 1)])
    mat = np.outer(ph, pa)
    p_h = float(np.tril(mat, -1).sum())
    p_d = float(np.trace(mat))
    p_a = float(np.triu(mat,  1).sum())
    # Clip para evitar log(0)
    total = p_h + p_d + p_a
    return max(0.005, p_h/total), max(0.005, p_d/total), max(0.005, p_a/total)


def predecir(home: str, away: str, fuerzas: dict, gamma: float,
             media: tuple, conf_ctx=None) -> tuple:
    from src.mundial import calcular_lambdas, _media_fuerzas
    lh, la = calcular_lambdas(home, away, fuerzas, gamma, media, conf_ctx)
    # WC sede neutral: promediamos H como local y A como local
    la2, lh2 = calcular_lambdas(away, home, fuerzas, gamma, media, conf_ctx)
    lh_neu = (lh + lh2) / 2
    la_neu = (la + la2) / 2
    return probs_poisson(lh_neu, la_neu)


# ── Metricas ───────────────────────────────────────────────────────────────

def metricas(probs_list: list, outcomes: list) -> dict:
    """
    probs_list : [(p_h, p_d, p_a), ...]
    outcomes   : ['H'/'D'/'A', ...]
    """
    n = len(outcomes)
    acc = brier = logloss = 0.0

    for (ph, pd_, pa), out in zip(probs_list, outcomes):
        ih = float(out == "H")
        id_ = float(out == "D")
        ia = float(out == "A")

        # Accuracy
        pred = "H" if ph >= pd_ and ph >= pa else ("D" if pd_ >= pa else "A")
        acc += pred == out

        # Brier score (suma cuadratica por outcome)
        brier   += (ph - ih)**2 + (pd_ - id_)**2 + (pa - ia)**2

        # Log-loss
        p_ok = ph * ih + pd_ * id_ + pa * ia
        logloss -= np.log(max(p_ok, 1e-10))

    return {
        "N":        n,
        "Accuracy": acc / n,
        "Brier":    brier / n,
        "LogLoss":  logloss / n,
    }


def odds_a_probs(h_avg, d_avg, a_avg):
    """Convierte odds decimales a probabilidades normalizadas (sin margen)."""
    try:
        p = np.array([1/h_avg, 1/d_avg, 1/a_avg], dtype=float)
        p = p / p.sum()
        return tuple(np.clip(p, 0.005, 0.995))
    except Exception:
        return (1/3, 1/3, 1/3)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=== Validacion retrospectiva — WC2022 ===\n")

    # ── Datos ──────────────────────────────────────────────────────────────
    print("Cargando datos de entrenamiento (pre-WC2022) ...")
    df_train = _cargar_training()
    print(f"  {len(df_train)} partidos | "
          f"{df_train['Date'].min().strftime('%Y-%m-%d')} - "
          f"{df_train['Date'].max().strftime('%Y-%m-%d')}")
    train_teams = set(df_train["HomeTeam"]) | set(df_train["AwayTeam"])
    print(f"  {len(train_teams)} equipos distintos en training\n")

    print("Cargando WC2022 (test) ...")
    df_test = _cargar_wc2022()
    n_test = len(df_test)
    print(f"  {n_test} partidos")

    # Resultado real
    outcomes = []
    for _, r in df_test.iterrows():
        if r["FTHG"] > r["FTAG"]:   outcomes.append("H")
        elif r["FTHG"] < r["FTAG"]: outcomes.append("A")
        else:                         outcomes.append("D")
    h_cnt = outcomes.count("H")
    d_cnt = outcomes.count("D")
    a_cnt = outcomes.count("A")
    print(f"  Resultados: H={h_cnt} ({100*h_cnt/n_test:.0f}%) "
          f"D={d_cnt} ({100*d_cnt/n_test:.0f}%) "
          f"A={a_cnt} ({100*a_cnt/n_test:.0f}%)\n")

    # Equipos del test cubiertos por el training
    test_teams = set(df_test["Home"]) | set(df_test["Away"])
    covered = test_teams & train_teams
    fallback = test_teams - train_teams
    print(f"  Equipos cubiertos por training: {len(covered)}/{len(test_teams)}")
    if fallback:
        print(f"  Usan fallback (media): {sorted(fallback)}")

    # ── Modelo basico (Poisson sin conf) ───────────────────────────────────
    print("\nEntrenando modelo Poisson basico (xi=0.004) ...")
    fuerzas_b, gamma_b = _estimar_basico(df_train)
    media_b = (float(np.mean([v[0] for v in fuerzas_b.values()])),
               float(np.mean([v[1] for v in fuerzas_b.values()])))
    print(f"  {len(fuerzas_b)} equipos calibrados | gamma={gamma_b:.3f}")

    probs_basico = []
    for _, r in df_test.iterrows():
        ph, pd_, pa = predecir(r["Home"], r["Away"],
                               fuerzas_b, gamma_b, media_b)
        probs_basico.append((ph, pd_, pa))

    # ── Modelo con ajuste de confederacion ─────────────────────────────────
    print("Entrenando modelo con ajuste de confederacion (MLE conf) ...")
    fuerzas_c, gamma_c, conf_s = _estimar_conf(df_train)
    from src.parametros import SELECCION_CONFEDERACION
    from src.mundial import _media_fuerzas
    beta_media = float(np.mean([v[1] for v in fuerzas_c.values()]))
    media_c = _media_fuerzas(fuerzas_c)
    conf_ctx = {"strengths": conf_s, "equipo_conf": SELECCION_CONFEDERACION,
                "beta_media": beta_media}
    print(f"  {len(fuerzas_c)} equipos | gamma={gamma_c:.3f} | "
          f"conf: " + " ".join(f"{k}={v:.2f}" for k, v in conf_s.items()))

    probs_conf = []
    for _, r in df_test.iterrows():
        ph, pd_, pa = predecir(r["Home"], r["Away"],
                               fuerzas_c, gamma_c, media_c, conf_ctx)
        probs_conf.append((ph, pd_, pa))

    # ── Baselines ──────────────────────────────────────────────────────────
    # 1. Uniforme
    probs_uni = [(1/3, 1/3, 1/3)] * n_test

    # 2. Base rate historica WC (goles reales en WC: ~46% H, ~23% D, ~31% A)
    BASE_H, BASE_D, BASE_A = 0.453, 0.234, 0.313
    probs_base_rate = [(BASE_H, BASE_D, BASE_A)] * n_test

    # 3. Mercado (odds decimales -> probabilidades sin margen de la casa)
    probs_mkt = []
    for _, r in df_test.iterrows():
        probs_mkt.append(odds_a_probs(r["H-Avg"], r["D-Avg"], r["A-Avg"]))

    # 4. Favorito siempre gana (odds -> 100% al menor odds)
    probs_fav = []
    for _, r in df_test.iterrows():
        ph, pd_, pa = odds_a_probs(r["H-Avg"], r["D-Avg"], r["A-Avg"])
        max_p = max(ph, pd_, pa)
        if ph == max_p:     probs_fav.append((0.99, 0.005, 0.005))
        elif pd_ == max_p:  probs_fav.append((0.005, 0.99, 0.005))
        else:               probs_fav.append((0.005, 0.005, 0.99))

    # ── Comparacion ────────────────────────────────────────────────────────
    modelos = {
        "Uniforme (1/3)":       probs_uni,
        "Base rate WC":         probs_base_rate,
        "Favorito siempre":     probs_fav,
        "Mercado (odds avg)":   probs_mkt,
        "Poisson basico":       probs_basico,
        "Poisson + conf MLE":   probs_conf,
    }

    print(f"\n{'='*68}")
    print(f"{'Modelo':<24} {'N':>4}  {'Accuracy':>9}  {'Brier':>7}  "
          f"{'LogLoss':>8}  {'vs Mkt Acc':>10}")
    print(f"{'-'*68}")

    m_acc = metricas(probs_mkt, outcomes)["Accuracy"]

    resultados = {}
    for nombre, probs in modelos.items():
        m = metricas(probs, outcomes)
        resultados[nombre] = m
        delta_acc = m["Accuracy"] - m_acc
        print(f"  {nombre:<22} {m['N']:>4}  "
              f"{m['Accuracy']:>8.1%}  "
              f"{m['Brier']:>7.4f}  "
              f"{m['LogLoss']:>8.4f}  "
              f"{delta_acc:>+9.1%}")
    print(f"{'='*68}\n")

    # ── Intervalo de confianza (accuracy binomial) ─────────────────────────
    n = n_test
    acc_poisson = resultados["Poisson basico"]["Accuracy"]
    se = np.sqrt(acc_poisson * (1 - acc_poisson) / n)
    print(f"Intervalo 95% accuracy Poisson basico: "
          f"{acc_poisson:.1%} ± {1.96*se:.1%}  "
          f"(n={n}, SE={se:.1%})")

    acc_mkt = resultados["Mercado (odds avg)"]["Accuracy"]
    se_mkt = np.sqrt(acc_mkt * (1 - acc_mkt) / n)
    print(f"Intervalo 95% accuracy Mercado:        "
          f"{acc_mkt:.1%} ± {1.96*se_mkt:.1%}\n")

    # ── Breakdown fase de grupos vs eliminatoria ───────────────────────────
    # WC2022 grupo = primeros 48 partidos, eliminatoria = últimos 16
    print("Breakdown por fase:")
    for fase, idx in [("Grupos (48)", slice(0, 48)), ("Eliminatoria (16)", slice(48, 64))]:
        outs_f = outcomes[idx]
        pb_f   = probs_basico[idx]
        pm_f   = probs_mkt[idx]
        mb = metricas(pb_f, outs_f)
        mm = metricas(pm_f, outs_f)
        print(f"  {fase:<22}  "
              f"Poisson acc={mb['Accuracy']:.1%} Brier={mb['Brier']:.4f}  |  "
              f"Mkt acc={mm['Accuracy']:.1%} Brier={mm['Brier']:.4f}")

    # ── Analisis de calibracion ────────────────────────────────────────────
    print("\nCalibracion del modelo Poisson basico (predicted p_H vs real freq):")
    bins = [(0.0, 0.33), (0.33, 0.50), (0.50, 0.65), (0.65, 1.0)]
    labels = ["bajo (<33%)", "medio (33-50%)", "alto (50-65%)", "muy alto (>65%)"]
    for (lo, hi), lbl in zip(bins, labels):
        sub_ph   = [probs_basico[i][0] for i in range(n) if lo <= probs_basico[i][0] < hi]
        sub_out  = [outcomes[i] for i in range(n) if lo <= probs_basico[i][0] < hi]
        if not sub_ph:
            continue
        real_freq = sub_out.count("H") / len(sub_out)
        pred_mean = np.mean(sub_ph)
        print(f"  P(H) {lbl:<18}: pred={pred_mean:.2f}  real={real_freq:.2f}  "
              f"n={len(sub_ph)}")

    # ── Conclusion ─────────────────────────────────────────────────────────
    print()
    print("=" * 68)
    print("CONCLUSION PARA LA WEB DE PREDICCIONES:")
    print("=" * 68)

    acc_uni = resultados["Uniforme (1/3)"]["Accuracy"]
    gap_vs_uni = acc_poisson - acc_uni
    gap_vs_mkt = acc_poisson - acc_mkt
    brier_p = resultados["Poisson basico"]["Brier"]
    brier_m = resultados["Mercado (odds avg)"]["Brier"]

    print(f"  Accuracy:  modelo={acc_poisson:.1%}  mercado={acc_mkt:.1%}  "
          f"uniforme={acc_uni:.1%}")
    print(f"  Brier:     modelo={brier_p:.4f}  mercado={brier_m:.4f}")
    print(f"  Gap vs uniforme: {gap_vs_uni:+.1%}")
    print(f"  Gap vs mercado:  {gap_vs_mkt:+.1%}")
    print()

    brier_ratio = brier_p / brier_m
    if brier_ratio < 1.05:
        veredicto = "COMPETITIVO con el mercado"
    elif brier_ratio < 1.15:
        veredicto = "RAZONABLE para una web (dentro del 15% del mercado)"
    elif brier_ratio < 1.30:
        veredicto = "ACEPTABLE — suficiente para contenido web, no para apuestas"
    else:
        veredicto = "LIMITADO — mejor usar el modelo como contexto narrativo"

    print(f"  Veredicto: {veredicto}")
    print()
    print("  CAVEAT: training pre-WC2022 carece de clasificatorias 2019-2022")
    print("  (gap de 4 anos de datos). Para WC2026 tenemos clasificatorias")
    print("  2023-2026 -> el modelo actual deberia ser MEJOR que en esta validacion.")


if __name__ == "__main__":
    main()
