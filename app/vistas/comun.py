"""Piezas que comparten varias vistas."""

import streamlit as st

import src.data as dt
import src.pipeline as pl


@st.cache_data(show_spinner="Ajustando el modelo...")
def _analizar(serie, params, modelo, H, m):
    """pipeline.analyze cacheado: el modelo C reestima un ARIMA-GARCH en
    cada origen del walk-forward y tarda varios segundos."""
    return pl.analyze(serie, params, modelo, H, m)


def moneda(ticker):
    """Moneda en que cotiza el activo, segun el sufijo del ticker.

    Londres cotiza en peniques (GBP), no en libras: un rotulo GBP haria
    pensar que el precio esta en libras.
    """
    if dt.suffix(ticker) == ".L":
        return "GBP"
    return dt.market_info(ticker)[0]


def analisis_activo(ctx):
    """Selector del activo y motor completo sobre su serie.

    Lo usan Resumen, Precio, Rendimientos y Proyeccion.
    Devuelve (ticker, serie de precios, salida de pipeline.analyze).
    """
    ticker = st.selectbox("Activo", list(ctx.precios.columns))
    serie = ctx.precios[ticker].dropna()
    r = _analizar(serie, ctx.params, ctx.modelo, ctx.H, ctx.m)  # punto de entrada del bot
    return ticker, serie, r
