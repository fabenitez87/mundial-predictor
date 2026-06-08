import os
import requests
import pandas as pd

COLUMNAS = ["HomeTeam", "AwayTeam", "FTHG", "FTAG"]

LIGAS = {
    "E0": "Premier League",
    "SP1": "La Liga",
}

TEMPORADAS = ["2122", "2223", "2324"]

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{liga}.csv"
URL_WC2026 = "https://www.football-data.co.uk/WorldCup2026.xlsx"

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)  # crea /data si no existe (necesario en cloud)

# Normalización: nombre en el xlsx → nombre en GRUPOS de mundial.py
NOMBRE_MAP_INT = {
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "D.R. Congo":           "DR Congo",
    "Czech Republic":       "Czechia",
}


# ── Ligas de clubes ────────────────────────────────────────────────────────

def _ruta_local(liga: str, temporada: str) -> str:
    return os.path.join(DATA_DIR, f"{liga}_{temporada}.csv")


def _descargar(liga: str, temporada: str) -> pd.DataFrame:
    url = BASE_URL.format(season=temporada, liga=liga)
    ruta = _ruta_local(liga, temporada)

    if os.path.exists(ruta):
        return pd.read_csv(ruta, usecols=lambda c: c in COLUMNAS)

    print(f"  Descargando {liga} {temporada} ...")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    with open(ruta, "wb") as f:
        f.write(resp.content)

    return pd.read_csv(ruta, usecols=lambda c: c in COLUMNAS)


def cargar_partidos() -> pd.DataFrame:
    frames = []
    for liga in LIGAS:
        for temp in TEMPORADAS:
            df = _descargar(liga, temp)
            df = df.dropna(subset=["FTHG", "FTAG"])
            df["FTHG"] = df["FTHG"].astype(int)
            df["FTAG"] = df["FTAG"].astype(int)
            df["liga"] = liga
            df["temporada"] = temp
            frames.append(df)

    return pd.concat(frames, ignore_index=True)


# ── Selecciones nacionales ─────────────────────────────────────────────────

def _leer_sheet_wc(ruta_xlsx: str, nombre: str,
                   col_h: str, col_a: str) -> pd.DataFrame:
    """Lee una hoja del xlsx y la devuelve con columnas normalizadas."""
    s = pd.read_excel(ruta_xlsx, sheet_name=nombre)
    s = s[["Date", "Home", "Away", col_h, col_a]].copy()
    s.columns = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    return s


def cargar_internacionales() -> pd.DataFrame:
    """
    Descarga WorldCup2026.xlsx de football-data.co.uk y extrae partidos
    internacionales desde 2018 (post-Rusia). Cachea en data/internacionales.csv.

    Fuentes combinadas:
      - WorldCup2018     : 64 partidos, jun-jul 2018
      - WorldCup2022     : 64 partidos, nov-dic 2022
      - WorldCup2026Qualifiers: clasificatorias 2023-abr 2026
    """
    ruta_csv  = os.path.join(DATA_DIR, "internacionales.csv")
    ruta_xlsx = os.path.join(DATA_DIR, "WorldCup2026.xlsx")

    if os.path.exists(ruta_csv):
        return pd.read_csv(ruta_csv, parse_dates=["Date"])

    if not os.path.exists(ruta_xlsx):
        print("  Descargando WorldCup2026.xlsx ...")
        resp = requests.get(URL_WC2026, timeout=60)
        resp.raise_for_status()
        with open(ruta_xlsx, "wb") as f:
            f.write(resp.content)

    frames = [
        _leer_sheet_wc(ruta_xlsx, "WorldCup2018",          "HGFT", "AGFT"),
        _leer_sheet_wc(ruta_xlsx, "WorldCup2022",          "HGFT", "AGFT"),
        _leer_sheet_wc(ruta_xlsx, "WorldCup2026Qualifiers","HG",   "AG"),
    ]

    df = pd.concat(frames, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "FTHG", "FTAG"])
    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)

    # Post-Rusia 2018
    df = df[df["Date"] >= "2018-01-01"].copy()

    # Normalizar nombres para que coincidan con GRUPOS
    df["HomeTeam"] = df["HomeTeam"].replace(NOMBRE_MAP_INT)
    df["AwayTeam"] = df["AwayTeam"].replace(NOMBRE_MAP_INT)

    df = df.sort_values("Date").reset_index(drop=True)
    df.to_csv(ruta_csv, index=False)
    print(f"  {len(df)} partidos internacionales procesados -> {ruta_csv}")
    return df
