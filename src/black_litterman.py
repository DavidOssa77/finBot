"""Modelo de Black-Litterman: equilibrio del mercado mas visiones.

Convencion: Pi, Q y mu_BL son EXCESOS anuales sobre la tasa libre de
riesgo, en rendimientos simples. Para llevarlos a Markowitz se les suma
rf: el optimizador trabaja con rendimientos totales.

    Pi    = delta Sigma w_mkt                         equilibrio (CAPM)
    mu_BL = Pi + tau Sigma P' (P tau Sigma P' + Omega)^-1 (Q - P Pi)
    S_BL  = Sigma + tau Sigma - tau Sigma P' (...)^-1 P tau Sigma
"""

import numpy as np
import pandas as pd

TIPOS = ("absoluta", "relativa")


def market_weights(caps):
    """Pesos de mercado a partir de capitalizaciones.

    caps --> Serie de capitalizaciones en una misma moneda, por ticker
    Devuelve una Serie de pesos que suman 1.
    """
    if (caps <= 0).any() or caps.isna().any():
        raise ValueError("toda capitalizacion debe ser un numero positivo")
    return caps / caps.sum()


def risk_aversion(erp, w_mkt, Sigma):
    """Aversion al riesgo implicita del mercado: delta = prima / var(m).

    erp --> prima de riesgo de mercado anual, E[Rm] - rf
    w_mkt --> pesos de mercado (Serie alineada con Sigma)
    Sigma --> covarianzas anuales
    """
    w = w_mkt.reindex(Sigma.index).to_numpy()
    return erp / (w @ Sigma.to_numpy() @ w)


def implied_returns(delta, Sigma, w_mkt):
    """Excesos de equilibrio: Pi = delta Sigma w_mkt (optimizacion inversa).

    Son los rendimientos que hacen optimo al portafolio de mercado.
    Equivalen al CAPM: Pi_i = beta_i (E[Rm] - rf), con beta medida
    contra ese mismo portafolio.
    """
    w = w_mkt.reindex(Sigma.index).to_numpy()
    return pd.Series(delta * Sigma.to_numpy() @ w, index=Sigma.index)


def views_matrices(vistas, tickers, Sigma, tau):
    """Traduce una lista de visiones a las matrices P, Q y Omega.

    vistas --> lista de diccionarios con las claves
                 tipo      "absoluta" o "relativa"
                 activo    ticker de la vision
                 contra    segundo ticker (solo en las relativas)
                 q         exceso anual esperado; en una relativa, cuanto
                           rinde activo por encima de contra
                 confianza entre 0 (excluido) y 1
    tickers --> orden de los activos (el de Sigma)
    tau --> incertidumbre del equilibrio

    Omega es diagonal y sale de la confianza (Idzorek simplificado):
        Omega_k = (1 - c) / c * P_k tau Sigma P_k'
    Con una sola vision, su rendimiento recorre exactamente la fraccion c
    del camino entre el equilibrio y Q. Con confianza 1, Omega_k = 0.
    """
    tickers = list(tickers)
    n, k = len(tickers), len(vistas)
    P, Q, c = np.zeros((k, n)), np.zeros(k), np.zeros(k)

    for i, v in enumerate(vistas):
        if v["tipo"] not in TIPOS:
            raise ValueError(f"vision {i + 1}: tipo desconocido {v['tipo']}")
        if v["activo"] not in tickers:
            raise ValueError(f"vision {i + 1}: {v['activo']} no esta en el "
                             "portafolio")
        if not 0 < v["confianza"] <= 1:
            raise ValueError(f"vision {i + 1}: la confianza debe estar en "
                             "(0, 1]")
        P[i, tickers.index(v["activo"])] = 1
        if v["tipo"] == "relativa":
            if v.get("contra") not in tickers or v["contra"] == v["activo"]:
                raise ValueError(f"vision {i + 1}: la relativa necesita un "
                                 "segundo activo distinto del portafolio")
            P[i, tickers.index(v["contra"])] = -1
        Q[i], c[i] = v["q"], v["confianza"]

    S = Sigma.reindex(index=tickers, columns=tickers).to_numpy()
    varianza_vision = np.diag(P @ (tau * S) @ P.T)
    Omega = np.diag((1 - c) / c * varianza_vision)
    return P, Q, Omega


def posterior(Pi, Sigma, P, Q, Omega, tau):
    """Rendimientos y covarianzas de Black-Litterman.

    Pi --> excesos de equilibrio (Serie)
    Sigma --> covarianzas anuales (DataFrame)
    P, Q, Omega --> visiones: matriz k x n, vector k, matriz k x k
    tau --> incertidumbre del equilibrio
    Devuelve (mu_BL como Serie, Sigma_BL como DataFrame).

    Usa la forma "equilibrio + correccion", que no invierte Omega y por
    eso admite confianza del 100 % (Omega = 0). Es algebraicamente igual
    a la formula maestra [(tS)^-1 + P'O^-1 P]^-1 [(tS)^-1 Pi + P'O^-1 Q].
    Sin visiones (k = 0) la correccion es cero: mu_BL = Pi y
    Sigma_BL = (1 + tau) Sigma.
    """
    S = Sigma.to_numpy()
    tS = tau * S
    pi = Pi.reindex(Sigma.index).to_numpy()

    if len(Q) == 0:
        mu, S_bl = pi, S + tS
    else:
        ganancia = tS @ P.T @ np.linalg.inv(P @ tS @ P.T + Omega)
        mu = pi + ganancia @ (Q - P @ pi)
        S_bl = S + tS - ganancia @ P @ tS

    return (pd.Series(mu, index=Sigma.index),
            pd.DataFrame(S_bl, index=Sigma.index, columns=Sigma.columns))


def black_litterman(Sigma, w_mkt, erp, vistas, tau=0.05):
    """Corre el modelo completo.

    Sigma --> covarianzas anuales de los activos del portafolio
    w_mkt --> pesos de mercado (Serie por ticker)
    erp --> prima de riesgo de mercado anual
    vistas --> lista de visiones, ver views_matrices
    tau --> incertidumbre del equilibrio

    Devuelve un diccionario con delta, Pi, P, Q, Omega, mu_BL y
    Sigma_BL. Los rendimientos son excesos sobre rf.
    """
    delta = risk_aversion(erp, w_mkt, Sigma)
    Pi = implied_returns(delta, Sigma, w_mkt)
    P, Q, Omega = views_matrices(vistas, Sigma.index, Sigma, tau)
    mu_bl, Sigma_bl = posterior(Pi, Sigma, P, Q, Omega, tau)
    return {"delta": delta, "Pi": Pi, "P": P, "Q": Q, "Omega": Omega,
            "mu_BL": mu_bl, "Sigma_BL": Sigma_bl}
