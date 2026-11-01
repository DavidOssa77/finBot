"""VaR individual, niveles de entrada y salida, y puerta de señal"""

import numpy as np
from scipy.stats import norm

def var(m, v, conf, capital):
    """VaR parametrico individual en el horizonte

    m, v --> momentos acumulados, como los devuelve cumulative_moments
    conf --> nivel de confianza; la guia admite 0.95 y 0.99
    capital --> capital hipotetico, para expresar la perdida en dinero

    Devuelve una tupla (fraccional, monetario):
      fraccional --> perdida como fraccion del capital, entre 0 y 1
      monetario  --> la misma perdida multiplicada por el capital

    Ambos son magnitudes positivas: un VaR de 0.05 significa "perder el
    5 %", no "-5 %". Nunca es negativo por el piso max(0, ...).

    El VaR no es la perdida maxima posible ni dice cuanto se pierde mas
    alla del cuantil; con colas gruesas puede subestimar el riesgo.
    """
    q = m + norm.ppf(1 - conf) * np.sqrt(v)
    frac = max(0.0, 1 - np.exp(q))
    return frac, capital * frac

def levels(P_t, m, v, p_L, p_U, cost_buy, cost_sell):
    """Entrada, niveles terminales y relacion beneficio-riesgo neta

    P_t --> ultimo precio ajustado, que es la entrada
    m, v --> momentos acumulados
    p_L, p_U --> probabilidades de cola para SL y TP
    cost_buy, cost_sell --> costos proporcionales de compra y venta

    Devuelve un diccionario con siete claves:
      E --> entrada: el ultimo precio, sin costos
      SL --> stop-loss terminal: cuantil p_L de la distribucion
      TP --> take-profit terminal: cuantil p_U
      P_BE --> precio de equilibrio: a cuanto hay que vender para no
                  perder nada, una vez pagadas las dos comisiones
      D_neto --> cuanto se arriesga: lo pagado al comprar menos lo que
                  se recibiria vendiendo en SL
      U_neto --> cuanto se gana: lo que se recibiria vendiendo en TP
                  menos lo pagado al comprar
      BR_neto --> U_neto / D_neto; None si D_neto no es positivo, porque
                  ahi la razon no significa nada

    Los cuatro primeros son precios, D_neto y U_neto son dinero por
    accion, y BR_neto es un cociente sin unidades.

    SL y TP son niveles TERMINALES en t+H, no barreras que se vigilen
    durante el camino: el precio puede cruzarlos antes y volver.
    """
    sd = np.sqrt(v)
    E = P_t
    SL = P_t * np.exp(m + norm.ppf(p_L) * sd)
    TP = P_t * np.exp(m + norm.ppf(p_U) * sd)

    entrada = E * (1 + cost_buy)
    P_BE = entrada / (1 - cost_sell)
    D_neto = entrada - SL * (1 - cost_sell)
    U_neto = TP * (1 - cost_sell) - entrada
    BR_neto = U_neto / D_neto if D_neto > 0 else None

    return {"E": E, "SL": SL, "TP": TP, "P_BE": P_BE,
            "D_neto": D_neto, "U_neto": U_neto, "BR_neto": BR_neto}

def terminal_probs(P_t, P_BE, m, v):
    """Probabilidades terminales de ganar, perder y quedar neutral.

    P_t --> ultimo precio
    P_BE --> precio de equilibrio con costos, el que devuelve levels
    m, v --> momentos acumulados

    Devuelve un diccionario con tres claves que suman uno:
      win --> probabilidad de terminar por encima de P_BE
      lose --> probabilidad de terminar por debajo
      neutral --> solo puede valer 1 en la rama determinista, cuando
                  v = 0 y el precio proyectado coincide con P_BE

    Son probabilidades TERMINALES: de estar por encima o por debajo en
    t+H, no de haber tocado esos niveles antes. Un activo puede cruzar
    el stop-loss por el camino y aun asi terminar ganando.
    """
    if v > 0:
        # z compara lo que se NECESITA para cubrir costos, ln(P_BE/P_t),
        # contra lo que el modelo ESPERA, m; el resultado va en desviaciones.
        # z negativo -> el rendimiento esperado supera al requerido -> ganar
        #               es lo probable, y por eso win = 1 - cdf(z)
        # z positivo -> los costos piden mas de lo que el modelo espera
        z = (np.log(P_BE / P_t) - m) / np.sqrt(v)
        return {"win": 1 - norm.cdf(z), "lose": norm.cdf(z), "neutral": 0.0}

    # rama determinista: sin varianza el precio futuro es un unico valor,
    # asi que la probabilidad solo puede ser 0 o 1
    P_estrella = P_t * np.exp(m)
    if P_estrella > P_BE:
        return {"win": 1.0, "lose": 0.0, "neutral": 0.0}
    if P_estrella < P_BE:
        return {"win": 0.0, "lose": 1.0, "neutral": 0.0}
    return {"win": 0.0, "lose": 0.0, "neutral": 1.0}

def signal_gate(niveles, diagnosticos, sufficient, BR_min):
    """Puerta de señal: decide si el activo pasa y por que

    niveles --> diccionario que devuelve levels
    diagnosticos --> diccionario que devuelve diagnostics
    sufficient --> booleano de walk_forward: si hubo historia suficiente
                   para validar el modelo
    BR_min --> relacion beneficio-riesgo minima exigida

    Devuelve un diccionario con cuatro claves:
      signal --> True solo si las siete reglas se cumplen
      failed --> lista con el texto de las reglas incumplidas; vacia
                 cuando hay señal. Es lo que se muestra al usuario
                 para justificar un rechazo
      warnings --> contrastes que rechazan y, de haberla, la muestra
                   insuficiente. No bloquean nunca
      label --> "condicional" si hay advertencias, "incondicional" si no
    """
    E, SL, TP = niveles["E"], niveles["SL"], niveles["TP"]
    P_BE, D_neto, U_neto, BR_neto = (niveles["P_BE"], niveles["D_neto"],
                                     niveles["U_neto"], niveles["BR_neto"])

    reglas = {
        "SL < E": SL < E,
        "E < TP": E < TP,
        "P_BE >= E": P_BE >= E,
        "P_BE < TP": P_BE < TP,
        "D_neto > 0": D_neto > 0,
        "U_neto > 0": U_neto > 0,
        "BR_neto >= BR_min": BR_neto is not None and BR_neto >= BR_min,
    }
    incumplidas = [nombre for nombre, ok in reglas.items() if not ok]

    advertencias = [nombre for nombre in ("jarque_bera", "ljung_box",
                                          "brown_forsythe", "arch_lm")
                    if diagnosticos[nombre]["reject"]]
    if not sufficient:
        advertencias.append("muestra insuficiente")

    return {
        "signal": not incumplidas,
        "failed": incumplidas,
        "warnings": advertencias,
        "label": "condicional" if advertencias else "incondicional",
    }