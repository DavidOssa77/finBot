"""Optimizacion de portafolios de Markowitz sin ventas en corto.

Convencion: rendimientos SIMPLES, igual que en capm.py. El rendimiento de
un portafolio es la combinacion lineal de los rendimientos simples de sus
activos

Restricciones de todos los optimizadores:
    sum(w) = 1        todo el capital invertido en activos riesgosos
    0 <= w <= w_max   sin ventas en corto y con tope por activo
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .capm import simple_returns
from .data import align, resample_prices


def returns_matrix(prices, freq):
    """Rendimientos simples de todos los activos sobre fechas comunes.

    prices --> DataFrame ancho de precios (diarios)
    freq --> frecuencia de los rendimientos ("D", "W-FRI", "ME")

    La covarianza exige que cada fecha tenga dato en todos los activos:
    se alinean los precios antes de calcular rendimientos.
    """
    return simple_returns(align(resample_prices(prices, freq)))


def moments(R, m):
    """Vector de medias y matriz de covarianzas anualizados.

    R --> DataFrame de rendimientos simples por periodo
    m --> periodos por año

    Devuelve (mu, Sigma) como Serie y DataFrame indexados por ticker.
    Media aritmetica por m y covarianza muestral (ddof=1) por m.
    """
    return R.mean() * m, R.cov(ddof=1) * m


def stats(w, mu, Sigma, rf=0.0):
    """Rendimiento esperado, volatilidad y Sharpe de un portafolio.

    w --> pesos (arreglo o Serie alineada con mu)
    mu, Sigma --> momentos anuales
    rf --> tasa libre de riesgo anual
    """
    w = np.asarray(w, dtype=float)
    ret = float(w @ np.asarray(mu))
    vol = float(np.sqrt(w @ np.asarray(Sigma) @ w))
    return {"return": ret, "volatility": vol, "sharpe": (ret - rf) / vol}


def _check(n, w_max):
    """Con n activos y tope w_max solo hay solucion si n * w_max >= 1."""
    if n * w_max < 1 - 1e-12:
        raise ValueError(
            f"con {n} activos y un tope de {w_max:.0%} por activo no se "
            f"puede invertir el 100 %: el tope minimo es {1 / n:.2%}")


def _suma_uno(w):
    """Restriccion de igualdad: sum(w) - 1 = 0."""
    return w.sum() - 1


def _grad_suma_uno(w):
    """Gradiente de la restriccion: un vector de unos."""
    return np.ones_like(w)


def _solve(objetivo, gradiente, n, w_max, extra=()):
    """Minimiza objetivo(w) con suma 1, sin cortos y con tope por activo.

    gradiente --> derivada analitica del objetivo; sin ella SLSQP la
                  aproxima por diferencias finitas y pierde precision
    extra --> restricciones adicionales (la frontera fija el rendimiento)
    Parte de pesos iguales, que siempre son factibles si _check pasa.
    Los pesos que el optimizador deja en 1e-10 o menos se llevan a cero:
    son ruido numerico, no posiciones.
    """
    _check(n, w_max)
    restricciones = [{"type": "eq", "fun": _suma_uno,
                      "jac": _grad_suma_uno}, *extra]
    res = minimize(objetivo, np.full(n, 1 / n), jac=gradiente,
                   method="SLSQP", bounds=[(0, w_max)] * n,
                   constraints=restricciones,
                   options={"ftol": 1e-12, "maxiter": 1000})
    if not res.success:
        raise RuntimeError(f"el optimizador no convergio: {res.message}")
    w = np.where(res.x < 1e-10, 0.0, res.x)
    return w / w.sum()


def min_variance(Sigma, w_max=1.0):
    """Portafolio de minima varianza global (PMVG).

    Sigma --> covarianzas anuales (DataFrame)
    w_max --> peso maximo por activo
    Devuelve una Serie de pesos indexada por ticker.

    El objetivo se divide por la traza de Sigma: con varianzas del orden
    de 1e-4, SLSQP da por resuelto el problema antes de tiempo.
    """
    S = Sigma.to_numpy()
    k = 1 / np.trace(S)

    def varianza(w):
        return k * w @ S @ w

    def gradiente(w):
        return 2 * k * S @ w

    w = _solve(varianza, gradiente, len(S), w_max)
    return pd.Series(w, index=Sigma.index)


def max_sharpe(mu, Sigma, rf, w_max=1.0):
    """Portafolio tangente: el de mayor razon de Sharpe.

    mu, Sigma --> momentos anuales
    rf --> tasa libre de riesgo anual
    Devuelve una Serie de pesos indexada por ticker.

    Si ningun activo rinde mas que rf, la tangente no existe (la recta
    que sale de rf no toca la frontera por arriba) y se lanza un error en
    lugar de devolver un portafolio sin sentido.
    """
    if (mu <= rf).all():
        raise ValueError("ningun activo supera la tasa libre de riesgo: "
                         "no hay portafolio tangente")
    m, S = mu.to_numpy(), Sigma.to_numpy()

    def menos_sharpe(w):
        return -(w @ m - rf) / np.sqrt(w @ S @ w)

    def gradiente(w):
        vol = np.sqrt(w @ S @ w)
        return -(m / vol - (w @ m - rf) * (S @ w) / vol ** 3)

    w = _solve(menos_sharpe, gradiente, len(m), w_max)
    return pd.Series(w, index=mu.index)

def max_return(mu, w_max=1.0):
    """Portafolio de maxima rentabilidad esperada.

    mu --> rendimientos esperados anuales (Serie)
    w_max --> peso maximo por activo
    Devuelve una Serie de pesos indexada por ticker.

    El objetivo w' mu es lineal y no depende de Sigma, asi que no hace
    falta optimizador: se llena hasta el tope el activo de mayor mu, luego
    el siguiente, y asi hasta completar el 100 %. Sin tope, es invertir
    todo en el activo de mayor rendimiento esperado. Es el extremo
    superior de la frontera eficiente.
    """
    _check(len(mu), w_max)
    w = pd.Series(0.0, index=mu.index)
    restante = 1.0
    for tk in mu.sort_values(ascending=False).index:
        w[tk] = min(w_max, restante)
        restante -= w[tk]
        if restante <= 1e-12:
            break
    return w

def key_portfolios(mu, Sigma, rf, w_max=1.0):
    """Los tres portafolios de referencia de la frontera eficiente.

    Devuelve (pesos, resumen):
      pesos --> DataFrame tickers x portafolio
      resumen --> DataFrame portafolio x (return, volatility, sharpe)

    El tangente se omite si ningun activo supera rf.
    """
    pesos = {"minima varianza": min_variance(Sigma, w_max),
             "maxima rentabilidad": max_return(mu, w_max)}
    if (mu > rf).any():
        pesos["tangente"] = max_sharpe(mu, Sigma, rf, w_max)
    pesos = pd.DataFrame(pesos)
    resumen = pd.DataFrame({k: stats(w, mu, Sigma, rf)
                            for k, w in pesos.items()}).T
    return pesos, resumen

def efficient_frontier(mu, Sigma, n_points=40, w_max=1.0):
    """Frontera eficiente sin ventas en corto.

    mu, Sigma --> momentos anuales
    n_points --> puntos de la curva
    w_max --> peso maximo por activo
    Devuelve un DataFrame con columnas return y volatility, de la minima
    varianza a la maxima rentabilidad.

    Para cada rendimiento objetivo entre esos dos extremos se busca la
    minima varianza que lo alcanza. El extremo superior no se optimiza:
    es max_return, que es exacto.
    """
    S = Sigma.to_numpy()
    m = mu.to_numpy()
    k = 1 / np.trace(S)
    inferior = stats(min_variance(Sigma, w_max), mu, Sigma)["return"]
    superior = stats(max_return(mu, w_max), mu, Sigma)

    def varianza(w):
        return k * w @ S @ w

    def gradiente(w):
        return 2 * k * S @ w

    def grad_objetivo(w):
        return m

    puntos = []
    for objetivo in np.linspace(inferior, superior["return"], n_points)[:-1]:

        def alcanza_objetivo(w, objetivo=objetivo):
            return w @ m - objetivo

        w = _solve(varianza, gradiente, len(m), w_max,
                   extra=[{"type": "eq", "fun": alcanza_objetivo,
                           "jac": grad_objetivo}])
        puntos.append(stats(w, mu, Sigma))
    puntos.append(superior)
    return pd.DataFrame(puntos)[["return", "volatility"]]


def max_utility(mu, Sigma, A, w_max=1.0):
    """Portafolio 100 % riesgoso de maxima utilidad media-varianza.

    mu, Sigma --> momentos anuales
    A --> aversion al riesgo del inversionista
    w_max --> peso maximo por activo
    Devuelve una Serie de pesos indexada por ticker.

    Maximiza U = w' mu - A/2 w' Sigma w sin activo libre de riesgo. Es
    un punto de la frontera eficiente: con A grande se acerca a la
    minima varianza y con A pequeño a la maxima rentabilidad.
    """
    m, S = mu.to_numpy(), Sigma.to_numpy()

    def menos_utilidad(w):
        return -(w @ m - A / 2 * w @ S @ w)

    def gradiente(w):
        return -(m - A * S @ w)

    w = _solve(menos_utilidad, gradiente, len(m), w_max)
    return pd.Series(w, index=mu.index)


def optimal_allocation(mu, Sigma, rf, A, w_max=1.0):
    """Asignacion del inversionista entre activos y renta fija.

    mu, Sigma --> momentos anuales
    rf --> tasa libre de riesgo anual (la renta fija)
    A --> aversion al riesgo del inversionista
    w_max --> peso maximo por activo riesgoso
    Devuelve un diccionario con
      y --> y* = (E[Rt] - rf) / (A sigma_t^2), sin recortar
      weights --> Serie de pesos en los activos riesgosos
      w_rf --> peso en renta fija
      return, volatility, utility --> del portafolio completo

    Separacion de Tobin sin apalancamiento: si y* <= 1 se invierte y* en
    el tangente y el resto en renta fija (un punto de la linea del
    mercado de capitales). Si y* > 1 haria falta endeudarse, asi que se
    elige el portafolio 100 % riesgoso de maxima utilidad, que esta en
    la frontera entre el tangente y la maxima rentabilidad. Con y* = 1
    los dos casos coinciden en el tangente. Si ningun activo supera rf,
    todo va a renta fija.
    """
    if (mu <= rf).all():
        y = 0.0
        w = pd.Series(0.0, index=mu.index)
    else:
        tangente = max_sharpe(mu, Sigma, rf, w_max)
        t = stats(tangente, mu, Sigma, rf)
        y = (t["return"] - rf) / (A * t["volatility"] ** 2)
        if y <= 1:
            w = y * tangente
        else:
            w = max_utility(mu, Sigma, A, w_max)

    # max evita un -1e-16 de redondeo cuando todo va a activos riesgosos
    w_rf = max(0.0, 1 - w.sum())
    ret = float(w @ mu) + w_rf * rf
    vol = float(np.sqrt(w @ Sigma.to_numpy() @ w))
    return {"y": y, "weights": w, "w_rf": w_rf, "return": ret,
            "volatility": vol, "utility": ret - A / 2 * vol ** 2}
