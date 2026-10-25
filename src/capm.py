"""CAPM: beta contra el indice de mercado, rendimiento requerido y desempeño.

Convencion: aqui se usan rendimientos SIMPLES, R_t = P_t / P_{t-1} - 1.
El CAPM es una relacion lineal entre rendimientos esperados y el
rendimiento de un portafolio es la combinacion lineal de los rendimientos
simples de sus activos; con log-rendimientos esa linealidad no se cumple.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .data import resample_prices, PERIODOS_POR_ANIO

BLUME = (0.67, 0.33)


def simple_returns(prices):
    """R_t = P_t / P_{t-1} - 1, descartando el primer nulo."""
    return prices.pct_change(fill_method=None).dropna()


def per_period(rate_annual, m):
    """Tasa anual efectiva convertida a tasa por periodo: (1+r)^(1/m) - 1."""
    return (1 + rate_annual) ** (1 / m) - 1


def equal_weight_index(prices):
    """Indice equiponderado con rebalanceo en cada periodo, base 100.

    prices --> DataFrame ancho de precios
    Es el proxy de mercado cuando no hay un indice observable (fixture):
    cada periodo rinde el promedio simple de los rendimientos de todos los
    activos con dato.
    """
    r = prices.pct_change(fill_method=None).mean(axis=1).fillna(0)
    return 100 * (1 + r).cumprod()


def excess_returns(asset, market, rf, freq):
    """Rendimientos en exceso del activo y del mercado, alineados.

    asset, market --> Series de precios (diarios o de menor frecuencia)
    rf --> tasa anual: un numero o una Serie diaria en decimales
    freq --> frecuencia de la regresion ("D", "W-FRI", "ME")

    Devuelve un DataFrame con columnas asset, market y rf (por periodo).
    La tasa que se resta es la vigente al INICIO de cada periodo: es la
    que se conocia cuando se tomo la decision de mantener el activo.
    """
    m = PERIODOS_POR_ANIO[freq]
    precios = pd.concat({"asset": asset, "market": market}, axis=1).dropna()
    precios = resample_prices(precios, freq).dropna()
    r = simple_returns(precios)

    if isinstance(rf, pd.Series):
        tasa = rf.reindex(precios.index, method="ffill").shift(1)
        tasa = per_period(tasa, m).reindex(r.index)
    else:
        tasa = pd.Series(per_period(rf, m), index=r.index)

    out = pd.DataFrame({"asset": r["asset"] - tasa,
                        "market": r["market"] - tasa,
                        "rf": tasa})
    return out.dropna()


def regress(ex, m):
    """Regresion del modelo de mercado en excesos: R_i - rf = a + b (R_m - rf) + e.

    ex --> DataFrame de excess_returns
    m --> periodos por año

    Devuelve un diccionario con beta cruda y ajustada (Blume), alfa de
    Jensen por periodo y anualizada con su p-valor, R^2, la
    descomposicion de la varianza anual en sistematica e idiosincratica,
    y la volatilidad idiosincratica anual.

    La fraccion sistematica de la varianza coincide con el R^2 (con
    intercepto, var(y) = var(ajuste) + var(residuo)); por eso no se
    reporta aparte: la volatilidad idiosincratica es la que agrega
    informacion, en puntos de volatilidad.
    """
    X = sm.add_constant(ex["market"])
    fit = sm.OLS(ex["asset"], X).fit()
    a, b = fit.params["const"], fit.params["market"]

    var_m = ex["market"].var(ddof=1)
    var_e = fit.resid.var(ddof=1)
    sistematica = b ** 2 * var_m * m
    idiosincratica = var_e * m

    return {
        "n": int(fit.nobs),
        "beta": b,
        "beta_se": fit.bse["market"],
        "beta_p": fit.pvalues["market"],
        "beta_adj": BLUME[0] * b + BLUME[1],
        "alpha_period": a,
        "alpha_annual": a * m,
        "alpha_p": fit.pvalues["const"],
        "r2": fit.rsquared,
        "var_systematic": sistematica,
        "var_idiosyncratic": idiosincratica,
        "vol_idiosyncratic": np.sqrt(idiosincratica),
        "resid": fit.resid,
    }


def market_return(market, rf_now, m, method, erp):
    """Rendimiento esperado anual del mercado.

    market --> Serie de precios del indice, ya en la frecuencia de m
    rf_now --> tasa libre de riesgo anual vigente
    method --> "prima": rf + erp; "historico": media aritmetica anualizada
               de los rendimientos simples del indice
    erp --> prima de riesgo de mercado anual
    """
    if method == "prima":
        return rf_now + erp
    if method == "historico":
        return simple_returns(market).mean() * m
    raise ValueError(f"metodo desconocido: {method}; use prima o historico")


def performance(ex, beta, m):
    """Sharpe, Treynor y media y volatilidad anuales del activo.

    ex --> DataFrame de excess_returns
    beta --> beta cruda de la regresion
    Las tres medidas usan el exceso de rendimiento medio anualizado.
    """
    exceso = ex["asset"].mean() * m
    vol = ex["asset"].std(ddof=1) * np.sqrt(m)
    return {
        "mean_annual": (ex["asset"] + ex["rf"]).mean() * m,
        "excess_annual": exceso,
        "volatility_annual": vol,
        "sharpe": exceso / vol,
        "treynor": exceso / beta,
    }


def analyze_asset(asset, market, rf, rf_now, params):
    """Corre el CAPM completo sobre un activo.

    asset, market --> Series de precios en la MISMA moneda
    rf --> tasa anual para los excesos (numero o Serie diaria)
    rf_now --> tasa anual vigente para el rendimiento requerido; su plazo
               deberia coincidir con el horizonte de inversion
    params --> CapmParams
    """
    m = PERIODOS_POR_ANIO[params.freq]
    ex = excess_returns(asset, market, rf, params.freq)
    reg = regress(ex, m)
    perf = performance(ex, reg["beta"], m)

    mercado = resample_prices(market.dropna(), params.freq).loc[ex.index[0]:]
    E_Rm = market_return(mercado, rf_now, m, params.market_method, params.erp)
    b = reg["beta_adj"] if params.beta_kind == "ajustada" else reg["beta"]

    return {
        **reg, **perf,
        "rf_now": rf_now,
        "E_Rm": E_Rm,
        "required_return": rf_now + b * (E_Rm - rf_now),
        "market_excess_annual": ex["market"].mean() * m,
        "excess": ex,
    }


COLUMNAS = ["market", "currency", "n", "beta", "beta_adj", "beta_p",
            "alpha_annual", "alpha_p", "r2", "vol_idiosyncratic",
            "mean_annual", "volatility_annual", "required_return",
            "sharpe", "treynor", "excess_annual", "market_excess_annual"]


def capm_table(resultados, mercados, monedas):
    """Una fila por activo con lo que la vista CAPM necesita.

    resultados --> dict {ticker: salida de analyze_asset}
    mercados, monedas --> dict {ticker: nombre del indice / moneda}
    """
    filas = {}
    for tk, r in resultados.items():
        fila = {k: r[k] for k in COLUMNAS if k in r}
        fila["market"] = mercados[tk]
        fila["currency"] = monedas[tk]
        filas[tk] = fila
    return pd.DataFrame(filas).T.rename_axis("ticker")[COLUMNAS]
