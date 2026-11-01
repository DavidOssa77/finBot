"""Comparacion entre activos: coordenadas historicas y dominancia media-volatilidad"""

import pandas as pd

from .analytics import log_returns, descriptives, drawdown


def summary(prices, m):
    """Una fila por activo con sus coordenadas historicas.

    prices --> DataFrame ancho de precios (fechas x tickers)
    m --> periodos por año (252 diaria, 52 semanal, 12 mensual)

    Devuelve un DataFrame indexado por ticker con las once columnas que
    describen la muestra: fechas inicial y final, numero de rendimientos,
    ultimo precio, media y desviacion por periodo y anualizadas, los
    cuantiles 5 y 95 de los log-rendimientos, y la caida maxima.

    No depende del horizonte ni del modelo: son coordenadas del pasado,
    no proyecciones.
    """
    filas = {}
    for tk in prices.columns:
        s = prices[tk].dropna()
        g = log_returns(s)
        d = descriptives(g, m)
        _, dd_max = drawdown(s)
        filas[tk] = {
            "start_date": s.index[0],
            "end_date": s.index[-1],
            "return_observations": d["n"],
            "last_adjusted_close": s.iloc[-1],
            "mean_log_period": d["mean_period"],
            "sample_sd_period": d["sd_period"],
            "mean_log_annual": d["mean_annual"],
            "volatility_annual": d["sd_annual"],
            "q05_log": d["q05"],
            "q95_log": d["q95"],
            "max_drawdown": dd_max,
        }
    return pd.DataFrame(filas).T.rename_axis("ticker")


def _domina(a, b):
    """a domina a b si no es peor en ninguna dimension y es mejor en alguna."""
    no_peor = (a["mean_log_annual"] >= b["mean_log_annual"] and
               a["volatility_annual"] <= b["volatility_annual"])
    mejor = (a["mean_log_annual"] > b["mean_log_annual"] or
             a["volatility_annual"] < b["volatility_annual"])
    return no_peor and mejor

def preselect(tabla):
    """Añade razon retorno-volatilidad, dominancia y seleccion final

    tabla --> DataFrame como el que devuelve summary

    Devuelve una copia con tres columnas mas:
      individual_rvr --> retorno anual sobre volatilidad anual
      non_dominated --> True si ningun otro activo lo domina
      selected_max_rvr --> True solo para el de mayor rvr en la frontera
    """
    t = tabla.copy()
    t["individual_rvr"] = t["mean_log_annual"] / t["volatility_annual"]

    flags = []
    for i in t.index:
        dominado = any(_domina(t.loc[j], t.loc[i]) for j in t.index if j != i)
        flags.append(not dominado)
    t["non_dominated"] = flags

    frontera = t[t["non_dominated"]]
    t["selected_max_rvr"] = t.index == frontera["individual_rvr"].idxmax()
    return t