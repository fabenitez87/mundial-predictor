"""
Mundial 2026 Predictor — Streamlit app
Modelo Poisson bivariado + Monte Carlo + ajuste por confederacion
"""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.stats import poisson

# ── Configuracion de pagina ────────────────────────────────────────────────
st.set_page_config(
    page_title="Mundial 2026 Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports del proyecto ───────────────────────────────────────────────────
from src.mundial import GRUPOS, monte_carlo, calcular_lambdas, _media_fuerzas
from src.parametros import SELECCION_CONFEDERACION, CONFS

# ── Lista de equipos ───────────────────────────────────────────────────────
ALL_TEAMS = sorted(e for g in GRUPOS.values() for e in g)

# ── Carga del modelo (una sola vez) ───────────────────────────────────────
@st.cache_resource(
    show_spinner="⚙ Calibrando modelo Poisson bivariado... (primera carga ~20s)"
)
def cargar_modelo():
    """Descarga datos y estima parametros MLE. Se ejecuta UNA sola vez."""
    from src.carga_datos import cargar_internacionales
    from src.parametros import estimar_selecciones

    df_int = cargar_internacionales()
    fuerzas, gamma, conf_strengths = estimar_selecciones(df_int)
    beta_media = float(np.mean([v[1] for v in fuerzas.values()]))
    media = _media_fuerzas(fuerzas)
    conf_ctx = {
        "strengths":   conf_strengths,
        "equipo_conf": SELECCION_CONFEDERACION,
        "beta_media":  beta_media,
    }
    return fuerzas, gamma, conf_strengths, conf_ctx, media


# ── Funciones auxiliares ───────────────────────────────────────────────────

def lambdas_neutral(eq1, eq2, fuerzas, gamma, media, conf_ctx):
    """Lambdas en sede neutral (promedia ambas asignaciones de localía)."""
    lh1, la1 = calcular_lambdas(eq1, eq2, fuerzas, gamma, media, conf_ctx)
    lh2, la2 = calcular_lambdas(eq2, eq1, fuerzas, gamma, media, conf_ctx)
    return (lh1 + lh2) / 2, (la1 + la2) / 2


def matriz_marcadores(lh, la, max_g=5):
    """Matriz (max_g+1)×(max_g+1) de probabilidades de marcadores exactos."""
    ph = np.array([poisson.pmf(i, max(lh, 1e-6)) for i in range(max_g + 1)])
    pa = np.array([poisson.pmf(j, max(la, 1e-6)) for j in range(max_g + 1)])
    return np.outer(ph, pa)


def probs_1x2(mat):
    p_h = float(np.tril(mat, -1).sum())
    p_d = float(np.trace(mat))
    p_a = float(1.0 - p_h - p_d)
    return max(0.001, p_h), max(0.001, p_d), max(0.001, p_a)


# ── Cargar modelo ──────────────────────────────────────────────────────────
fuerzas, gamma_m, conf_strengths, conf_ctx, media = cargar_modelo()


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚽ Mundial 2026")
    st.caption("Predictor · Modelo Poisson bivariado")
    st.divider()

    seccion = st.radio(
        "Navegar",
        [
            "⚽  Predictor de partido",
            "🏆  Simulacion del torneo",
            "📊  Sobre el modelo",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption("**Niveles de confederacion estimados**")
    conf_sorted = sorted(conf_strengths.items(), key=lambda x: -x[1])
    max_s = max(v for _, v in conf_sorted)
    for conf, s in conf_sorted:
        bar = int(round(s / max_s * 16))
        st.markdown(
            f"`{conf:<9}` {'█' * bar}{'░' * (16 - bar)} `{s:.3f}`",
            unsafe_allow_html=False,
        )


# ══════════════════════════════════════════════════════════════════════════
# SECCION 1 — Predictor de partido
# ══════════════════════════════════════════════════════════════════════════
if seccion == "⚽  Predictor de partido":
    st.title("⚽ Predictor de Partido")
    st.markdown(
        "Predicciones basadas en el modelo **Poisson bivariado** calibrado "
        "con WC2018, WC2022 y clasificatorias 2026 · "
        "[football-data.co.uk](https://www.football-data.co.uk)"
    )
    st.divider()

    col_sel1, col_sep, col_sel2 = st.columns([5, 1, 5])
    idx1 = ALL_TEAMS.index("Argentina") if "Argentina" in ALL_TEAMS else 0
    idx2 = ALL_TEAMS.index("France") if "France" in ALL_TEAMS else 1

    with col_sel1:
        eq1 = st.selectbox("Equipo 1", ALL_TEAMS, index=idx1, key="eq1")
    with col_sep:
        st.markdown("<br><br><center>**vs**</center>", unsafe_allow_html=True)
    with col_sel2:
        eq2 = st.selectbox("Equipo 2", ALL_TEAMS, index=idx2, key="eq2")

    if eq1 == eq2:
        st.warning("Selecciona dos equipos distintos.")
        st.stop()

    predecir = st.button("⚽  Predecir partido", type="primary",
                         use_container_width=True)

    if predecir:
        lh, la = lambdas_neutral(eq1, eq2, fuerzas, gamma_m, media, conf_ctx)
        mat = matriz_marcadores(lh, la, max_g=5)
        p_h, p_d, p_a = probs_1x2(mat)

        # ── Probabilidades 1X2 ─────────────────────────────────────────────
        st.subheader("Probabilidades 1X2")
        m1, m2, m3 = st.columns(3)
        m1.metric(
            f"🏅 {eq1} gana",
            f"{p_h:.1%}",
            delta=f"{p_h - 1/3:+.0%} vs igual",
            delta_color="normal",
        )
        m2.metric(
            "🤝 Empate",
            f"{p_d:.1%}",
            delta=f"{p_d - 1/3:+.0%} vs igual",
            delta_color="off",
        )
        m3.metric(
            f"🏅 {eq2} gana",
            f"{p_a:.1%}",
            delta=f"{p_a - 1/3:+.0%} vs igual",
            delta_color="normal",
        )

        # ── Goles esperados ────────────────────────────────────────────────
        st.subheader("Goles esperados (lambdas Poisson)")
        g1, g2 = st.columns(2)
        g1.metric(f"λ {eq1}", f"{lh:.2f} goles/partido")
        g2.metric(f"λ {eq2}", f"{la:.2f} goles/partido")

        st.divider()

        # Precalcular candidatos (usados en top 5, top 10 y heatmap)
        candidatos = sorted(
            ((mat[i, j], i, j) for i in range(6) for j in range(6)),
            reverse=True,
        )

        # ── Top 5 + Heatmap ───────────────────────────────────────────────
        col_left, col_right = st.columns([1, 2])

        with col_left:
            st.subheader("Top 5 marcadores")
            df_top5 = pd.DataFrame(
                [
                    {"Marcador": f"{eq1} {i} – {j} {eq2}",
                     "Probabilidad": f"{p * 100:.2f}%"}
                    for p, i, j in candidatos[:5]
                ]
            )
            st.dataframe(df_top5, hide_index=True, use_container_width=True)

        with col_right:
            st.subheader("Heatmap de marcadores (goles 0–5)")
            st.caption(
                f"Filas = goles de **{eq1}**  ·  "
                f"Columnas = goles de **{eq2}**"
            )
            fig_heat = px.imshow(
                mat * 100,
                x=[str(j) for j in range(6)],
                y=[str(i) for i in range(6)],
                labels={"x": eq2, "y": eq1, "color": "Prob (%)"},
                color_continuous_scale="YlOrRd",
                text_auto=".2f",
                aspect="equal",
            )
            fig_heat.update_coloraxes(colorbar_title="Prob (%)")
            fig_heat.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                height=320,
            )
            st.plotly_chart(fig_heat, use_container_width=True)

        # ── Top 10 marcadores — barras horizontales ────────────────────────
        st.subheader("Top 10 marcadores mas probables")

        color_e1  = "#2196F3"   # azul → gana equipo 1
        color_draw = "#9E9E9E"  # gris → empate
        color_e2  = "#F44336"   # rojo → gana equipo 2

        df_bar = pd.DataFrame([
            {
                "Marcador":       f"{eq1} {i}–{j} {eq2}",
                "Prob (%)":       round(p * 100, 2),
                "Resultado":      (
                    f"{eq1} gana" if i > j
                    else "Empate" if i == j
                    else f"{eq2} gana"
                ),
            }
            for p, i, j in candidatos[:10]
        ])

        fig_bar = px.bar(
            df_bar.iloc[::-1].reset_index(drop=True),  # mayor prob arriba
            x="Prob (%)",
            y="Marcador",
            color="Resultado",
            orientation="h",
            text="Prob (%)",
            color_discrete_map={
                f"{eq1} gana": color_e1,
                "Empate":       color_draw,
                f"{eq2} gana": color_e2,
            },
            labels={"Prob (%)": "Probabilidad (%)", "Marcador": ""},
        )
        fig_bar.update_traces(
            texttemplate="%{x:.2f}%",
            textposition="outside",
        )
        fig_bar.update_layout(
            height=420,
            margin=dict(l=10, r=110, t=10, b=10),
            legend=dict(title="Resultado", orientation="h",
                        yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(title="Probabilidad (%)", range=[0, df_bar["Prob (%)"].max() * 1.35]),
        )
        st.plotly_chart(fig_bar, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# SECCION 2 — Simulacion del torneo
# ══════════════════════════════════════════════════════════════════════════
elif seccion == "🏆  Simulacion del torneo":
    st.title("🏆 Simulacion del Mundial 2026")
    st.markdown(
        "Monte Carlo sobre el **bracket oficial FIFA**:  \n"
        "12 grupos × 4 equipos → R32 → R16 → Cuartos → Semis → Final  \n"
        "Los 8 mejores terceros avanzan a la Ronda de 32."
    )
    st.divider()

    n_sims = st.slider(
        "Numero de simulaciones",
        min_value=1_000,
        max_value=50_000,
        value=10_000,
        step=1_000,
        format="%d",
    )
    secs_est = n_sims / 10_000 * 1.7
    st.caption(f"Tiempo estimado: ~{secs_est:.0f} segundos")

    simular = st.button(
        "▶  Simular Mundial 2026", type="primary", use_container_width=True
    )

    if simular:
        with st.spinner(f"Simulando {n_sims:,} torneos completos..."):
            df_mc = monte_carlo(
                GRUPOS, fuerzas, gamma_m, n=n_sims, conf_ctx=conf_ctx
            )
        st.session_state["df_mc"] = df_mc
        st.session_state["n_sims_run"] = n_sims
        st.success(f"Simulacion completa — {n_sims:,} torneos")

    if "df_mc" in st.session_state:
        df_mc   = st.session_state["df_mc"]
        n_run   = st.session_state.get("n_sims_run", n_sims)
        df_top  = df_mc.head(10).copy()

        st.subheader(f"Top 10 candidatos a campeon — {n_run:,} simulaciones")
        st.bar_chart(
            df_top.set_index("equipo")[["campeon_%"]],
            use_container_width=True,
            height=320,
        )

        st.subheader("Tabla completa — 48 equipos")
        df_display = df_mc.copy()
        for col in ["campeon_%", "final_%", "semifinal_%", "cuartos_%"]:
            df_display[col] = df_display[col].apply(lambda x: f"{x:.2f}%")
        df_display.columns = ["Equipo", "Campeon %", "Final %",
                               "Semifinal %", "Cuartos %"]

        st.dataframe(
            df_display.reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
            height=500,
        )

        csv_bytes = df_mc.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥  Descargar CSV completo",
            data=csv_bytes,
            file_name=f"mundial_2026_{n_run}sims.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ══════════════════════════════════════════════════════════════════════════
# SECCION 3 — Sobre el modelo
# ══════════════════════════════════════════════════════════════════════════
else:
    st.title("📊 Sobre el modelo")
    st.divider()

    col_doc, col_val = st.columns([3, 2])

    with col_doc:
        st.subheader("Como funciona")
        st.markdown("""
El predictor usa un modelo **Poisson bivariado** para estimar la probabilidad
de cada marcador en un partido de futbol.

### Parametros por equipo

| Parametro | Significado |
|-----------|-------------|
| **α (ataque)** | Cuantos goles por partido tiende a marcar el equipo |
| **β (defensa)** | Cuantos goles tiende a recibir por unidad de ataque rival |
| **γ (localia)** | Ventaja del equipo local (~1.35x en internacionales) |

### Formula

Para un partido **Local A vs Visitante B**:

- **Goles esperados A** = γ · α_A · β_B
- **Goles esperados B** = α_B · β_A

Cada equipo anota goles siguiendo una **distribucion de Poisson** independiente.
Eso da la probabilidad exacta de cualquier marcador (ej. 2–1 = 9.95%).

### Calibracion

Los parametros se estiman por **maxima verosimilitud (MLE)** con algoritmo
L-BFGS-B usando datos de partidos internacionales reales,
con decay temporal exponencial (**ξ = 0.004**): los partidos recientes
pesan mas que los viejos.

### Ajuste por confederacion

Los clasificatorias de Asia (AFC) y Oceania (OFC) son mas debiles que
las europeas. El modelo estima **5 parametros libres de nivel de confederacion**
(UEFA = 1.0 fija como referencia) usando los partidos cross-confederacion
de los Mundiales. Esto evita sobreestimar equipos que golean a rivales debiles.

### Monte Carlo del torneo

Para predecir el torneo completo, el modelo simula miles de veces el bracket
oficial FIFA 2026 (R32 → R16 → QF → SF → Final). Si hay empate en eliminatoria,
se simulan penales con leve ventaja al equipo de mayor fuerza atacante.
        """)

    with col_val:
        st.subheader("Validacion retrospectiva")
        st.markdown("**Test: WC2022 — 64 partidos**")

        c1, c2 = st.columns(2)
        c1.metric("Accuracy del modelo", "54.7%",
                  help="% de resultados 1X2 correctamente predichos en WC2022")
        c2.metric("Accuracy mercado", "53.1%",
                  help="Usando probabilidades implicitas de cuotas de apuestas")

        c1b, c2b = st.columns(2)
        c1b.metric("Accuracy eliminatorias", "62.5%",
                   "+6.3% vs mercado",
                   help="En los 16 partidos de fase eliminatoria del WC2022")
        c2b.metric("Brier Score", "0.61",
                   help="Mercado: 0.58 (menor es mejor)")

        st.info(
            "**Caveat importante:** el training pre-WC2022 carece de "
            "clasificatorias 2019–2022 (gap de 4 anos). Para WC2026 "
            "disponemos de clasificatorias **2023–2026**, por lo que "
            "se espera mejor performance."
        )

        st.subheader("Fuentes de datos")
        st.markdown("""
**Training (pre-entrenamiento):**
- Torneos previos 2014–2017 (220 partidos)
- World Cup 2018 qualifiers 2015–2017 (860 partidos)
- World Cup 2018 (64 partidos)
- World Cup 2022 (64 partidos)
- World Cup 2026 qualifiers 2023–2026 (889 partidos)

**Fuente:** [football-data.co.uk](https://www.football-data.co.uk)
**Grupos:** [Sorteo FIFA 5-dic-2025](https://www.fifa.com)
        """)

        st.subheader("Stack tecnico")
        st.markdown("""
`Python` · `pandas` · `numpy` · `scipy`
`Streamlit` · MLE L-BFGS-B · Poisson bivariado
Bracket oficial FIFA 2026 (Annex C — 495 combinaciones de terceros)
        """)
