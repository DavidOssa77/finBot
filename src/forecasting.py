"""Modelos A y B (homocedasticos) y C (ARIMA-GARCH), trayectoria analitica
y validacion walk-forward."""

import warnings

import numpy as np
from .analytics import sample_stats
import pandas as pd
from arch import arch_model
from scipy.stats import norm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import arma2ma

MODELOS = ("A", "B", "C")
# Candidatos del modelo C: ARIMA(p, d, q) sobre log-precios. Con d = 1
# equivalen a ARMA(p, q) con constante sobre log-rendimientos.
ORDENES_C = ((1, 1, 1), (0, 1, 1), (1, 1, 0))

def fit(g_train, model):
    """Estima los parametros por periodo con el bloque de entrenamiento

    g_train --> Serie de log-rendimientos, SOLO del entrenamiento
    model --> "A" caminata aleatoria sin deriva, "B" caminata aleatoria lognormal con deriva
    Devuelve (mu, sigma) por periodo, nunca anualizados
    """
    if model not in ("A", "B"): #con esto valido que alguien no ingrese C o D o cualquier cosa
        raise ValueError(f"modelo desconocido: {model}; use 'A' o 'B'")
    mean, sigma = sample_stats(g_train)

    if(model == 'A'):
        mu = 0
    else:
        mu = mean
    return mu, sigma

def cumulative_moments(mu, sigma, h):

    """Media y varianza del log-rendimiento acumulado a h periodos
        Convierte los parámetros de un periodo (los obtenidos con fit) en parámetros 
        para H periodos

    mu, sigma --> estimadores por periodo, tal como los devuelve fit()
    h --> numero de periodos hacia adelante
    Devuelve (m, v) 
    """
    return h * mu, h * sigma ** 2

def fit_arima_garch(g_train):
    """Ajusta el modelo C: ARIMA elegido por AIC y GARCH(1,1) t-Student.

    g_train --> Serie de log-rendimientos, SOLO del entrenamiento
    Devuelve un diccionario con
      candidatos --> DataFrame por orden con log-verosimilitud y AIC
      orden --> (p, 1, q) elegido: el de menor AIC
      arima --> resultado de statsmodels del orden elegido
      garch --> resultado de arch sobre los residuos de ese ARIMA

    Cada ARIMA(p,1,q) sobre log-precios se estima como ARMA(p,q) con
    constante sobre log-rendimientos: es el mismo modelo y la constante
    es la deriva. Se elige por AIC y no por verosimilitud: el (1,1,1)
    contiene a los otros dos y su verosimilitud nunca es menor. El GARCH
    se estima en dos pasos, sobre los residuos en puntos porcentuales
    (x100), escala en la que arch converge mejor.
    """
    y = g_train.to_numpy()
    ajustes, filas = {}, []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for orden in ORDENES_C:
            p, _, q = orden
            res = ARIMA(y, order=(p, 0, q), trend="c").fit()
            ajustes[orden] = res
            filas.append({"orden": f"ARIMA{orden}",
                          "log-verosimilitud": res.llf, "AIC": res.aic})
        aics = {orden: res.aic for orden, res in ajustes.items()}
        elegido = min(aics, key=aics.get)
        garch = arch_model(ajustes[elegido].resid * 100, mean="Zero",
                           vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
    return {"candidatos": pd.DataFrame(filas).set_index("orden"),
            "orden": elegido, "arima": ajustes[elegido], "garch": garch}


def _moments_c(g_train, H):
    """Momentos acumulados de 1 a H con el modelo C.

    Media: suma de los rendimientos que pronostica el ARIMA. Varianza:
    los choques futuros e_{t+j} tienen la varianza que pronostica el
    GARCH, sigma^2_{t+j}, y cada uno se propaga a los rendimientos
    siguientes con los pesos psi del ARMA. En la suma de h rendimientos
    el choque j pesa Psi_{h-j} = psi_0 + ... + psi_{h-j}, asi que
        v_h = sum_{j=1..h} sigma^2_{t+j} Psi_{h-j}^2
    Devuelve (m, v, detalle) con m y v por periodo, no anualizados.
    """
    ajuste = fit_arima_garch(g_train)
    arima, garch = ajuste["arima"], ajuste["garch"]
    media = np.asarray(arima.forecast(H))
    var_paso = garch.forecast(horizon=H, reindex=False).variance
    var_paso = var_paso.to_numpy()[-1] / 1e4
    psi = arma2ma(np.r_[1, -arima.arparams], np.r_[1, arima.maparams], H)
    Psi = np.cumsum(psi)
    m = np.cumsum(media)
    v = np.array([np.sum(var_paso[:h] * Psi[h - 1::-1] ** 2)
                  for h in range(1, H + 1)])

    gp = garch.params
    persistencia = gp["alpha[1]"] + gp["beta[1]"]
    detalle = {
        "candidatos": ajuste["candidatos"], "orden": ajuste["orden"],
        "converge": (bool(arima.mle_retvals.get("converged", True))
                     and garch.convergence_flag == 0),
        "arima_params": pd.Series(arima.params,
                                  index=arima.param_names),
        "garch_params": gp, "persistencia": persistencia,
        "vol_condicional": np.sqrt(var_paso[0]),
        "vol_incondicional": (np.sqrt(gp["omega"] / (1 - persistencia))
                              / 100 if persistencia < 1 else None),
        "vol_muestral": g_train.std(ddof=1),
        "vol_historica": garch.conditional_volatility / 100,
    }
    return m, v, detalle


def moments(g_train, model, H):
    """Momentos acumulados del log-rendimiento de 1 a H periodos.

    g_train --> Serie de log-rendimientos, SOLO del entrenamiento
    model --> "A", "B" o "C" (ARIMA-GARCH)
    H --> horizonte maximo
    Devuelve (m, v, detalle): m y v son arreglos de largo H con la media
    y la varianza acumuladas a h = 1..H; detalle es None en A y B y, en
    C, el diccionario con el ajuste (ordenes, parametros, volatilidades
    por periodo).
    """
    if model not in MODELOS:
        raise ValueError(f"modelo desconocido: {model}; use A, B o C")
    if model == "C":
        return _moments_c(g_train, H)
    mu, sigma = fit(g_train, model)
    m, v = cumulative_moments(mu, sigma, np.arange(1, H + 1))
    return m, v, None


def terminal(P_t, m, v, params):
    """Distribucion del precio dados los momentos acumulados

    P_t --> ultimo precio observado
    m, v --> momentos acumulados, como los devuelve cumulative_moments
    params --> Params; se leen p_L y p_U para los cuantiles
    Devuelve un diccionario con mediana, media y los dos cuantiles

    Funciona con m y v escalares o como arrays: en el segundo caso
    devuelve un array por clave, que es lo que aprovecha path()
    """
    sd = np.sqrt(v)
    return {
        "median": P_t * np.exp(m),
        "mean": P_t * np.exp(m + v / 2),
        "q_L": P_t * np.exp(m + norm.ppf(params.p_L) * sd),
        "q_U": P_t * np.exp(m + norm.ppf(params.p_U) * sd),
    }

def path(P_t, m, v, params):
    """Trayectoria analitica de 1 a H.

    P_t --> ultimo precio observado
    m, v --> momentos acumulados a h = 1..H, como los devuelve moments()
    params --> Params; se leen p_L y p_U
    Devuelve un DataFrame con una fila por paso y columnas
    h, median, mean, q_L, q_U.
    """
    df = pd.DataFrame(terminal(P_t, m, v, params))
    df.insert(0, "h", np.arange(1, len(m) + 1))
    return df

def walk_forward(g, model, H, m, params):
    """Validacion temporal sin mirar el futuro (look ahead bias)

    Reestima el modelo en los diez ultimos origenes con ventana expansiva
    y mide el error contra lo que realmente paso.

    g --> Serie completa de log-rendimientos
    model --> "A", "B" o "C"; en C se reestima el ARIMA-GARCH completo,
              eleccion de orden incluida, en cada origen
    H --> horizonte
    m --> periodos por año, para el criterio de suficiencia
    params --> Params; se leen p_L y p_U para la cobertura
    Devuelve un diccionario con la suficiencia y, si se cumple, las metricas
    """
    T = len(g)
    n_min = max(2 * m, 5 * H)
    if T < n_min + H + 9:
        return {"sufficient": False, "T": T, "n_min": n_min,
                "required": n_min + H + 9}

    origins = [T - H - 9 + j for j in range(10)]
    errors, inside, dir_ok, dir_n = [], 0, 0, 0

    for o in origins:
        m_o, v_o, _ = moments(g.iloc[:o], model, H)
        m_H, v_H = m_o[-1], v_o[-1]
        Y = g.iloc[o:o + H].sum()
        errors.append(Y - m_H)

        sd = np.sqrt(v_H)
        lo = m_H + norm.ppf(params.p_L) * sd
        hi = m_H + norm.ppf(params.p_U) * sd
        inside += (lo <= Y <= hi)

        if abs(m_H) > 1e-12 and abs(Y) > 1e-12:
            dir_n += 1
            dir_ok += (np.sign(m_H) == np.sign(Y))

    errors = np.array(errors)
    return {
        "sufficient": True,
        "origins": origins,
        "errors": errors,
        "mae": np.abs(errors).mean(),
        "rmse": np.sqrt((errors ** 2).mean()),
        "coverage": inside / len(origins),
        "direction": None if dir_n == 0 else dir_ok / dir_n,
        "direction_n": dir_n,
    }