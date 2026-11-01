"""Lo que comparten todas las vistas: el contexto y las descargas cacheadas."""

from dataclasses import dataclass, field

import pandas as pd
import streamlit as st

import src.contracts as ct
import src.data as dt


@dataclass
class Contexto:
    """Estado comun que el panel lateral y la carga de datos entregan a
    cada vista.

    precios --> precios remuestreados a la frecuencia elegida
    precios_diarios --> precios antes de remuestrear (el CAPM fija su
                        propia frecuencia de regresion)
    ajustes --> lo que devolvieron los controles propios de la vista
    """

    fuente: str
    precios: pd.DataFrame
    precios_diarios: pd.DataFrame
    freq: str
    m: int
    H: int
    modelo: str
    params: ct.Params
    origen: str
    inicio: object = None
    fin: object = None
    ajustes: dict = field(default_factory=dict)


@st.cache_data
def cargar_fixture(path):
    """Lee el fixture una sola vez por sesion."""
    return dt.load_fixture(path)


def _sin_fallos(cacheada, *args):
    """Llama a una descarga cacheada sin dejar sus fallos en la cache.

    cacheada --> funcion con st.cache_data que devuelve (valor, fallos)
    args --> sus argumentos
    Un fallo puede ser pasajero (limite de peticiones de Yahoo, red). Si
    se guardara, la app lo repetiria sin volver a consultar mientras no
    cambien los argumentos. Se devuelve igual, con su aviso, pero se
    borra esa entrada para que la siguiente ejecucion reintente.
    """
    resultado = cacheada(*args)
    if resultado[1]:
        cacheada.clear(*args)
    return resultado


@st.cache_data
def _precios(tickers, inicio, fin):
    """Precios de yfinance, cacheados por combinacion de argumentos."""
    return dt.fetch_prices(list(tickers), inicio, fin)


def descargar(tickers, inicio, fin):
    """Precios de los activos; los fallos se reintentan."""
    return _sin_fallos(_precios, tickers, inicio, fin)


@st.cache_data
def _mercado(nombres, inicio, fin):
    """Indices de mercado, cacheados igual que los activos."""
    return dt.fetch_market(list(nombres), inicio, fin)


def descargar_mercado(nombres, inicio, fin):
    """Indices de mercado; los fallos se reintentan."""
    return _sin_fallos(_mercado, nombres, inicio, fin)


@st.cache_data
def _tasa(plazo, inicio, fin):
    """Rendimiento del bono del Tesoro al plazo pedido."""
    return dt.fetch_rate(plazo, inicio, fin)


def descargar_tasa(plazo, inicio, fin):
    """Tasa del Tesoro; si falla, se reintenta."""
    return _sin_fallos(_tasa, plazo, inicio, fin)


@st.cache_data
def _fx(monedas, inicio, fin):
    """Tipos de cambio contra el dolar."""
    return dt.fetch_fx(list(monedas), inicio, fin)


def descargar_fx(monedas, inicio, fin):
    """Tipos de cambio; los fallos se reintentan."""
    return _sin_fallos(_fx, monedas, inicio, fin)


@st.cache_data(ttl=3600)
def _capitalizacion(tickers):
    """Capitalizaciones de mercado; se refrescan cada hora."""
    return dt.fetch_market_caps(list(tickers))


def descargar_capitalizacion(tickers):
    """Capitalizaciones de mercado; los fallos se reintentan."""
    return _sin_fallos(_capitalizacion, tickers)
