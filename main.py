import os
import time

import numpy as np
import pandas as pd

from src.carga_datos import cargar_partidos, cargar_internacionales
from src.parametros import (estimar_parametros, estimar_selecciones,
                             CONFS, SELECCION_CONFEDERACION)
from src.prediccion import predecir_partido, marcador_mas_probable
from src.mundial import GRUPOS, monte_carlo, calcular_lambdas, _media_fuerzas

LOCAL     = "Man City"
VISITANTE = "Arsenal"
N_SIMS    = 10_000
CSV_V3    = os.path.join("data", "resultados_v3.csv")


def _mostrar_partido_test(local, visitante, fuerzas, gamma, media,
                          conf_ctx, conf_strengths):
    """Muestra lambdas de un partido con y sin shrinkage."""
    from src.parametros import SELECCION_CONFEDERACION as SC
    lh_sin, la_sin = calcular_lambdas(local, visitante, fuerzas, gamma, media)
    lh_con, la_con = calcular_lambdas(local, visitante, fuerzas, gamma, media,
                                      conf_ctx=conf_ctx)

    c_l = SC.get(local,    "?")
    c_v = SC.get(visitante, "?")
    s_l = conf_strengths.get(c_l, 1.0)
    s_v = conf_strengths.get(c_v, 1.0)

    print(f"   {local} ({c_l}, s={s_l:.3f})  vs  {visitante} ({c_v}, s={s_v:.3f})")
    print(f"   {'':24} {'SIN ajuste':>14}  {'CON ajuste':>14}  {'delta':>8}")
    print(f"   {'lambda '+local:24} {lh_sin:>14.4f}  {lh_con:>14.4f}  "
          f"{lh_con-lh_sin:>+8.4f}")
    print(f"   {'lambda '+visitante:24} {la_sin:>14.4f}  {la_con:>14.4f}  "
          f"{la_con-la_sin:>+8.4f}")

    # Probabilidades analíticas simples (sin Monte Carlo)
    from scipy.stats import poisson
    MAX_G = 10
    def probs(lh, la):
        ph = [poisson.pmf(i, lh) for i in range(MAX_G)]
        pa = [poisson.pmf(i, la) for i in range(MAX_G)]
        pl = sum(ph[i]*pa[j] for i in range(MAX_G) for j in range(MAX_G) if i > j)
        pe = sum(ph[i]*pa[i] for i in range(MAX_G))
        pv = 1 - pl - pe
        return pl, pe, pv

    pl_s, pe_s, pv_s = probs(lh_sin, la_sin)
    pl_c, pe_c, pv_c = probs(lh_con, la_con)
    print(f"   {'P('+local+' gana)':24} {pl_s:>14.2%}  {pl_c:>14.2%}  "
          f"{pl_c-pl_s:>+8.2%}")
    print(f"   {'P(Empate)':24} {pe_s:>14.2%}  {pe_c:>14.2%}  "
          f"{pe_c-pe_s:>+8.2%}")
    print(f"   {'P('+visitante+' gana)':24} {pv_s:>14.2%}  {pv_c:>14.2%}  "
          f"{pv_c-pv_s:>+8.2%}")
    print()


def main():
    print("=== Mundial Predictor - Modulo 4: Shrinkage defensivo ===\n")

    # ── 1-3. Pipeline base (datos + MLE clubes + prediccion) ───────────────
    print("1. Cargando datos de clubes ...")
    df = cargar_partidos()
    print(f"   {len(df)} partidos\n")

    print("2. MLE clubes (xi=0.0018) ...")
    fuerzas, gamma = estimar_parametros(df)
    print(f"   {len(fuerzas)} equipos | gamma={gamma:.4f}\n")

    print(f"3. Prediccion: {LOCAL} vs {VISITANTE}")
    pred = predecir_partido(LOCAL, VISITANTE, fuerzas, gamma)
    print(f"   P(local)={pred['prob_local']:.2%}  P(empate)={pred['prob_empate']:.2%}  "
          f"P(visit)={pred['prob_visitante']:.2%}")
    gl, gv = marcador_mas_probable(pred["matriz"])
    print(f"   Marcador mas probable: {LOCAL} {gl}-{gv} {VISITANTE}\n")

    # ── 4. Calibracion de selecciones con prior ELO (MAP, lambda=25) ──────────
    print("4. Calibrando selecciones (MLE xi=0.004 + prior ELO MAP, lambda=25) ...")
    df_int = cargar_internacionales()
    from src.carga_datos import cargar_elo as _cargar_elo
    from src.parametros import escalar_elo_a_alpha as _esc
    elo_series = _cargar_elo()
    fuerzas_con, gamma_con, conf_strengths = estimar_selecciones(
        df_int, elo_series=elo_series, lambda_prior=25
    )
    print(f"   {len(fuerzas_con)} selecciones | gamma={gamma_con:.4f} | "
          f"ELO prior: {'activo' if elo_series is not None else 'desactivado'}")

    print("\n   Niveles de confederacion (UEFA=1.0 referencia):")
    for conf in CONFS:
        s = conf_strengths[conf]
        bar = "#" * int(s * 20)
        print(f"   {conf:<10} s={s:.4f}  {bar}")

    # Medias globales y WC
    beta_media = float(np.mean([v[1] for v in fuerzas_con.values()]))
    media_global = _media_fuerzas(fuerzas_con)
    todos_wc = [e for g in GRUPOS.values() for e in g]
    beta_media_wc = float(np.mean([fuerzas_con.get(e, media_global)[1] for e in todos_wc]))

    # Conteo de partidos por equipo (shrinkage adaptativo)
    n_partidos = pd.concat([df_int["HomeTeam"], df_int["AwayTeam"]]).value_counts().to_dict()

    if elo_series is not None:
        _elo_vals = np.array([float(elo_series.get(e, elo_series.mean())) for e in todos_wc])
        _elo_m, _elo_s = float(_elo_vals.mean()), float(_elo_vals.std())
        elo_alpha_prior = {e: _esc(float(elo_series.get(e, _elo_m)), _elo_m, _elo_s)
                           for e in todos_wc}
        alpha_media_wc = 1.88
        skip_conf_alpha = True
        print(f"   ELO cargado: {len(elo_series)} equipos | prior ELO activo")
    else:
        elo_alpha_prior = {}
        alpha_media_wc = float(np.mean([fuerzas_con.get(e, media_global)[0] for e in todos_wc]))
        skip_conf_alpha = False
        print("   ELO no disponible — shrinkage clasico")

    print(f"   alpha_media WC: {alpha_media_wc:.4f}  beta_media WC: {beta_media_wc:.4f}")

    # Contexto de confederacion
    conf_ctx = {
        "strengths":       conf_strengths,
        "equipo_conf":     SELECCION_CONFEDERACION,
        "beta_media":      beta_media,
        "alpha_media":     alpha_media_wc,
        "beta_media_wc":   beta_media_wc,
        "n_partidos":      n_partidos,
        "elo_alpha_prior": elo_alpha_prior,
        "skip_conf_alpha": skip_conf_alpha,
    }

    # ── 5. Diagnostico: 48 equipos con parametros ajustados ──────────────────
    print("\n5. Diagnostico de parametros (48 equipos del Mundial)")
    from src.mundial import calcular_lambdas, _shrinkage_adaptivo

    print(f"   {'Equipo':<28} {'ELO':>5} {'N':>4} {'SF':>5} {'a_raw':>7} {'a_adj':>7} {'Conf':>9}")
    print("   " + "-"*73)

    todos_wc_sorted = sorted(todos_wc)
    for e in todos_wc_sorted:
        n     = n_partidos.get(e, 0)
        sf    = _shrinkage_adaptivo(n)
        a_raw = fuerzas_con.get(e, (alpha_media_wc, beta_media_wc))[0]
        conf_name = SELECCION_CONFEDERACION.get(e, "?")
        elo_v = int(elo_series.get(e, 0)) if elo_series is not None else 0

        # alpha_adj: usa ELO prior como target si disponible
        am_e = elo_alpha_prior.get(e, alpha_media_wc)
        a_adj = (1.0 - sf) * a_raw + sf * am_e
        # conf factor solo si no hay ELO
        if not skip_conf_alpha:
            s = conf_strengths.get(SELECCION_CONFEDERACION.get(e, "UEFA"), 1.0)
            a_adj *= s

        print(f"   {e:<28} {elo_v:>5} {n:>4} {sf:>5.2f} {a_raw:>7.3f} {a_adj:>7.3f} {conf_name:>9}")

    # ── 6. Test de partidos clave ──────────────────────────────────────────
    print("\n6. Test de partidos (sede neutral, probabilities analiticas)")
    from scipy.stats import poisson as _poisson

    def _test_partido(e1, e2):
        """Muestra probabilidades 1X2 para un partido neutral."""
        lh1, la1 = calcular_lambdas(e1, e2, fuerzas_con, gamma_con, media_global, conf_ctx)
        lh2, la2 = calcular_lambdas(e2, e1, fuerzas_con, gamma_con, media_global, conf_ctx)
        # Neutral: promedia goles propios de cada equipo en ambas asignaciones
        lh = (lh1 + la2) / 2   # e1 como local + e1 como visitante
        la = (la1 + lh2) / 2   # e2 como visitante + e2 como local
        MAX = 12
        mat = np.outer(
            [_poisson.pmf(i, lh) for i in range(MAX)],
            [_poisson.pmf(j, la) for j in range(MAX)],
        )
        p1 = float(np.tril(mat, -1).sum())
        pd = float(np.trace(mat))
        p2 = 1.0 - p1 - pd
        print(f"   {e1} vs {e2}:")
        print(f"     lambda={lh:.3f} / {la:.3f}   "
              f"P({e1})={p1:.1%}  P(Empate)={pd:.1%}  P({e2})={p2:.1%}")

    _test_partido("Argentina", "Haiti")    # target: ~70-75% Argentina
    _test_partido("France",    "Argentina") # target: ~38-42% France
    _test_partido("Brazil",    "Germany")   # target: ~35-40% cada uno
    _test_partido("Spain",     "France")    # target: ~40-45% Spain
    print()

    # ── 7. Monte Carlo CON shrinkage adaptativo ────────────────────────────
    print(f"7. Monte Carlo CON shrinkage adaptativo ({N_SIMS:,} sims) ...")
    t0 = time.time()
    df_v3 = monte_carlo(GRUPOS, fuerzas_con, gamma_con, n=N_SIMS,
                        conf_ctx=conf_ctx)
    print(f"   Completado en {time.time()-t0:.1f}s\n")

    print("   === TOP 10 CANDIDATOS A CAMPEON — v3 (con shrinkage) ===")
    print(f"   {'Equipo':<28} {'Campeon':>8} {'Final':>8} {'Semis':>7} {'Cuartos':>8} {'Conf':>9}")
    print("   " + "-" * 72)
    for _, row in df_v3.head(10).iterrows():
        e = row["equipo"]
        conf = SELECCION_CONFEDERACION.get(e, "?")
        print(f"   {e:<28} "
              f"{row['campeon_%']:>7.2f}% "
              f"{row['final_%']:>7.2f}% "
              f"{row['semifinal_%']:>6.2f}% "
              f"{row['cuartos_%']:>7.2f}%  "
              f"{conf:>9}")

    # ── 7. Comparacion sin/con shrinkage ───────────────────────────────────
    print("\n   === COMPARACION v2 (sin shrinkage) vs v3 (con shrinkage) ===")
    print("   Corriendo v2 sin shrinkage para comparacion ...")
    t0 = time.time()
    df_v2 = monte_carlo(GRUPOS, fuerzas_con, gamma_con, n=N_SIMS)
    print(f"   v2 completado en {time.time()-t0:.1f}s\n")

    # Union top-14 de ambas listas
    equipos_cmp = list(dict.fromkeys(
        list(df_v2["equipo"].head(14)) + list(df_v3["equipo"].head(14))
    ))
    v2_map = dict(zip(df_v2["equipo"], df_v2["campeon_%"]))
    v3_map = dict(zip(df_v3["equipo"], df_v3["campeon_%"]))

    filas = sorted(
        [(v3_map.get(e, 0), e, v2_map.get(e, 0), v3_map.get(e, 0))
         for e in equipos_cmp],
        reverse=True,
    )

    print(f"   {'Equipo':<28} {'v2 (sin)':>9} {'v3 (con)':>9} {'Cambio':>8} {'Conf':>9}")
    print("   " + "-" * 67)
    for _, e, v2, v3 in filas:
        cambio = v3 - v2
        fl = "^" if cambio > 0.3 else ("v" if cambio < -0.3 else "~")
        conf = SELECCION_CONFEDERACION.get(e, "?")
        print(f"   {e:<28} {v2:>8.2f}% {v3:>8.2f}%  {cambio:>+7.2f}% {fl}  {conf:>9}")

    # Guardar
    df_v3.to_csv(CSV_V3, index=False)
    print(f"\n   Resultados guardados en: {CSV_V3}")


if __name__ == "__main__":
    main()
