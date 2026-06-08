import numpy as np
from scipy.stats import poisson


def predecir_partido(
    local: str,
    visitante: str,
    fuerzas: dict,
    gamma: float,
    max_goles: int = 8,
) -> dict:
    """
    Calcula lambdas, probabilidades de resultado y matriz de marcadores.

    Retorna dict con:
      lambda_h, lambda_a, prob_local, prob_empate, prob_visitante, matriz
    """
    alpha_h, beta_h = fuerzas[local]
    alpha_a, beta_a = fuerzas[visitante]

    lambda_h = gamma * alpha_h * beta_a
    lambda_a = alpha_a * beta_h

    # Distribución de goles: P(X=i) para i in 0..max_goles
    prob_h = np.array([poisson.pmf(i, lambda_h) for i in range(max_goles + 1)])
    prob_a = np.array([poisson.pmf(i, lambda_a) for i in range(max_goles + 1)])

    # Matriz de marcadores (i=goles local, j=goles visitante)
    matriz = np.outer(prob_h, prob_a)

    prob_local    = float(np.tril(matriz, -1).sum())
    prob_empate   = float(np.trace(matriz))
    prob_visitante = float(np.triu(matriz, 1).sum())

    return {
        "lambda_h": round(lambda_h, 4),
        "lambda_a": round(lambda_a, 4),
        "prob_local": round(prob_local, 4),
        "prob_empate": round(prob_empate, 4),
        "prob_visitante": round(prob_visitante, 4),
        "matriz": matriz,
    }


def marcador_mas_probable(matriz: np.ndarray) -> tuple[int, int]:
    idx = np.unravel_index(np.argmax(matriz), matriz.shape)
    return int(idx[0]), int(idx[1])
