"""Carga, validacion y remuestreo de precios ajustados"""

import pandas as pd
import yfinance as yf
PERIODOS_POR_ANIO = {"D": 252, "W-FRI": 52, "ME": 12}

def load_fixture(path):
    """Carga el fixture del curso y devuelve la matriz de precios.

    path --> ruta al CSV en formato largo
    Devuelve un DataFrame ancho (fechas x tickers), ordenado y validado.
    """
    long_df = pd.read_csv(path, parse_dates=["date"])
    return validate_prices(wide_from_long(long_df))

def _download(ticker, start, end):
    """Descarga el cierre ajustado de un simbolo.

    Devuelve (serie, None) si respondio o (None, motivo) si fallo.
    """
    try:
        df = yf.download(ticker, start=start, end=end,
                         auto_adjust=True, progress=False)
    except Exception as e:
        return None, f"fallo la descarga: {e}"

    if df is None or df.empty:
        return None, "sin datos en el rango pedido"

    s = df["Close"].squeeze()
    if s.notna().sum() == 0:
        return None, "todos los precios llegaron vacios"
    return s, None


def fetch_prices(tickers, start, end):
    """Descarga precios ajustados de yfinance, aislando los fallos

    tickers --> lista de simbolos
    start, end --> fechas del rango
    Devuelve (precios, errores):
      precios --> DataFrame ancho con los tickers que respondieron
      errores --> dict {ticker: motivo} de los que fallaron

    Descarga uno a uno para que un simbolo invalido no arrastre a los
    demas
    """
    series, errores = {}, {}

    for t in tickers:
        s, motivo = _download(t, start, end)
        if motivo:
            errores[t] = motivo
        else:
            series[t] = s

    if not series:
        return pd.DataFrame(), errores

    return validate_prices(pd.DataFrame(series)), errores


# ------------------------------------------------------------- mercado
# Sufijo de Yahoo --> (moneda, nombre del mercado, indices candidatos).
# Los candidatos se prueban en orden: si Yahoo no publica el indice, se
# usa el ETF que lo replica. Londres cotiza en peniques (GBp): el nivel
# en dolares sale escalado por 100, pero los rendimientos no cambian.
MERCADOS = {
    "":    ("USD", "S&P 500",   ["^GSPC"]),
    ".CL": ("COP", "COLCAP",    ["^COLCAP", "ICOLCAP.CL"]),
    ".T":  ("JPY", "Nikkei 225", ["^N225"]),
    ".SA": ("BRL", "Bovespa",   ["^BVSP"]),
    ".MX": ("MXN", "IPC Mexico", ["^MXX"]),
    ".SN": ("CLP", "IPSA",      ["^IPSA"]),
    ".TO": ("CAD", "S&P/TSX",   ["^GSPTSE"]),
    ".L":  ("GBP", "FTSE 100",  ["^FTSE"]),
    ".DE": ("EUR", "DAX",       ["^GDAXI"]),
    ".PA": ("EUR", "CAC 40",    ["^FCHI"]),
    ".MC": ("EUR", "IBEX 35",   ["^IBEX"]),
    ".HK": ("HKD", "Hang Seng", ["^HSI"]),
}

# Bonos del Tesoro de EE.UU.; Yahoo los cotiza en puntos porcentuales
TASAS = {"13 semanas": ("^IRX", 0.25), "5 años": ("^FVX", 5),
         "10 años": ("^TNX", 10), "30 años": ("^TYX", 30)}


def suffix(ticker):
    """Sufijo de bolsa del simbolo de Yahoo ('' para EE.UU.)."""
    if ticker.startswith("^") or "." not in ticker:
        return ""
    s = "." + ticker.rsplit(".", 1)[1]
    return s if s in MERCADOS else ""


def market_info(ticker):
    """(moneda, nombre del mercado, indices candidatos) del activo."""
    return MERCADOS[suffix(ticker)]


def fetch_market(names, start, end):
    """Descarga el indice de cada mercado pedido, probando sus candidatos.

    names --> nombres de mercado (los de MERCADOS)
    Devuelve (indices, errores):
      indices --> dict {nombre: (simbolo usado, Serie de precios)}
      errores --> dict {nombre: motivo}
    """
    por_nombre = {v[1]: v[2] for v in MERCADOS.values()}
    indices, errores = {}, {}
    for n in names:
        motivos = []
        for cand in por_nombre[n]:
            s, motivo = _download(cand, start, end)
            if motivo is None:
                indices[n] = (cand, validate_prices(s.to_frame(cand))[cand])
                break
            motivos.append(f"{cand}: {motivo}")
        else:
            errores[n] = "; ".join(motivos)
    return indices, errores


def tenor_for(H, m):
    """Plazo del bono mas cercano al horizonte de inversion H/m años."""
    anios = H / m
    return min(TASAS, key=lambda k: abs(TASAS[k][1] - anios))


def fetch_rate(tenor, start, end):
    """Rendimiento anual del bono del Tesoro, en decimales.

    tenor --> clave de TASAS
    Devuelve (Serie, None) o (None, motivo). No pasa por validate_prices:
    una tasa en cero es un dato valido, no un faltante.
    """
    s, motivo = _download(TASAS[tenor][0], start, end)
    if motivo:
        return None, motivo
    return (s / 100).sort_index().ffill(), None


def fetch_fx(currencies, start, end):
    """Tipo de cambio de cada moneda contra el dolar.

    Yahoo cotiza 'XXX=X' como unidades de XXX por un dolar.
    Devuelve (DataFrame ancho por moneda, errores).
    """
    series, errores = {}, {}
    for c in sorted(set(currencies) - {"USD"}):
        s, motivo = _download(f"{c}=X", start, end)
        if motivo:
            errores[c] = motivo
        else:
            series[c] = s
    fx = validate_prices(pd.DataFrame(series)) if series else pd.DataFrame()
    return fx, errores


def to_usd(prices, currency, fx):
    """Expresa una serie de precios en dolares.

    prices --> Serie de precios en moneda local
    currency --> moneda de la serie
    fx --> DataFrame de fetch_fx

    Se usa el ultimo tipo de cambio disponible en cada fecha de precio,
    porque las bolsas y el mercado cambiario no comparten calendario.
    """
    if currency == "USD":
        return prices
    tc = fx[currency].reindex(prices.index, method="ffill")
    return (prices / tc).dropna()


def fetch_market_caps(tickers):
    """Capitalizacion bursatil actual de cada simbolo, en su moneda local.

    tickers --> lista de simbolos
    Devuelve (capitalizaciones, errores):
      capitalizaciones --> Serie por ticker con los que respondieron
      errores --> dict {ticker: motivo} de los que fallaron

    Es la capitalizacion de HOY, no la del periodo analizado. Para
    acciones preferenciales Yahoo puede reportar la de toda la empresa.
    """
    caps, errores = {}, {}
    for t in tickers:
        try:
            cap = yf.Ticker(t).fast_info["marketCap"]
        except Exception as e:
            errores[t] = f"fallo la consulta: {e}"
            continue
        if cap is None or not cap > 0:
            errores[t] = "Yahoo no reporta capitalizacion"
            continue
        caps[t] = float(cap)
    return pd.Series(caps, dtype=float), errores


def wide_from_long(long_df, value_col="adjusted_close"):
    """Convierte un DataFrame largo a uno ancho (fechas x tickers)

    long_df --> DataFrame con columnas 'date', 'ticker' y value_col
    value_col --> str, columna de precio a pivotear
    Devuelve DataFrame ancho, indice datetime, una columna por ticker.
    Duplicados de (date, ticker) se resuelven quedandose con el ultimo.
    """
    long_df = long_df.loc[:, ["date", "ticker", value_col]].copy()
    long_df["date"] = pd.to_datetime(long_df["date"])
    long_df = long_df.drop_duplicates(subset=["date", "ticker"], keep="last")
    wide = long_df.pivot(index="date", columns="ticker", values=value_col)
    return wide.rename_axis(columns=None)


def validate_prices(prices):
    """Aplica sobre un DataFrame ancho de precios (fechas x tickers)

    prices --> DataFrame, indice de fechas, una columna por ticker
    Devuelve el DataFrame validado; los precios no positivos quedan
    como NaN (faltante explicito), no se eliminan filas ni columnas.

     Ante fechas duplicadas se conserva el ultimo registro, aunque su precio
    sea invalido: en ese caso la observacion queda como faltante en lugar
    de recuperar el valor anterior
    """
    prices = prices.sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]
    prices = prices.mask(prices <= 0)
    return prices 

def resample_prices(prices, freq):
    """Remuestrea PRECIOS a la frecuencia pedida, antes de calcular rendimientos.

    prices --> DataFrame o Serie de precios, indice de fechas
    freq   --> "D" diaria, "W-FRI" semanal cierre viernes, "ME" fin de mes
    Devuelve los precios remuestreados, tomando el ultimo valido de cada
    periodo y descartando los periodos sin ningun dato.

    El orden importa: remuestrear precios y luego calcular log-rendimientos
    NO da lo mismo que calcular rendimientos diarios y luego agregarlos.
    """
    if freq not in PERIODOS_POR_ANIO:
        raise ValueError(f"frecuencia desconocida: {freq}; use D, W-FRI o ME")
    if freq == "D":
        return prices
    return prices.resample(freq).last().dropna(how="all")

def coverage(prices):
    """Cobertura temporal y porcentaje de faltantes por activo.

    prices --> DataFrame ancho de precios
    Devuelve un DataFrame con una fila por activo: primera y ultima fecha
    con dato, cuantas observaciones validas tiene y que porcentaje falta.
    """
    return pd.DataFrame({
        "start": prices.apply(lambda s: s.first_valid_index()),
        "end": prices.apply(lambda s: s.last_valid_index()),
        "n": prices.notna().sum(),
        "missing_pct": prices.isna().mean() * 100,
    })

def align(prices):
    """Deja solo las fechas donde todos los activos tienen dato.

    prices --> DataFrame ancho de precios, ya remuestreado
    Devuelve el DataFrame recortado a la interseccion comun de fechas.

    Solo se usa en la vista comparativa: para un activo individual los
    huecos los resuelve log_returns con su dropna, sin sacrificar fechas
    por culpa de los demas activos
    """
    return prices.dropna(how="any")