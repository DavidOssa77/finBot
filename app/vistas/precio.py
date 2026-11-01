"""Vista Precio: precio ajustado y drawdown del activo."""

import streamlit as st

from app import charts
from .comun import analisis_activo, moneda


def render(ctx):
    """Dibuja la vista con el contexto comun."""
    ticker, serie, r = analisis_activo(ctx)
    st.plotly_chart(charts.price_chart(serie, r["drawdown"], ticker,
                                       moneda(ticker)),
                    width="stretch")
    st.caption(f"Caida maxima desde maximos: {r['max_drawdown']:.2%}. "
               "El drawdown se calcula sobre precios, no sobre rendimientos.")
