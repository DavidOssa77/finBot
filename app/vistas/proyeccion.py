"""Vista Proyeccion: banda de cuantiles a lo largo del horizonte y, con el
modelo C, el detalle del ARIMA-GARCH."""

import numpy as np
import pandas as pd
import streamlit as st

from app import charts
from .comun import analisis_activo, moneda


def render(ctx):
    """Dibuja la vista con el contexto comun."""
    ticker, _, r = analisis_activo(ctx)
    params = ctx.params
    st.plotly_chart(
        charts.projection_chart(r["P_t"], r["path"], ticker,
                                moneda(ticker), params.p_L, params.p_U),
        width="stretch")
    st.caption("Cuantiles terminales en cada h, no un camino posible: el "
               "precio puede salirse de la banda y volver a entrar.")

    with st.expander("Ver la tabla"):
        st.dataframe(r["path"], width="stretch")

    if r["model_detail"] is not None:
        _detalle_c(r["model_detail"], r["g"].index, ticker, ctx.m)


def _detalle_c(d, fechas, ticker, m):
    """Ajuste del modelo C: eleccion del ARIMA, GARCH y volatilidades.

    d --> model_detail de pipeline.analyze
    fechas --> indice de los log-rendimientos, para fechar la volatilidad
    m --> periodos por año, para anualizar
    """
    st.subheader("Modelo C: ARIMA-GARCH")
    if not d["converge"]:
        st.warning("El optimizador no convergio del todo en el ARIMA o en "
                   "el GARCH: toma los resultados con cautela.")
    if d["persistencia"] >= 0.999:
        st.warning(
            f"GARCH en la frontera (alfa + beta = {d['persistencia']:.3f}): "
            "los choques de volatilidad casi no se disipan y la varianza "
            "incondicional no esta bien definida. Pasa con series "
            "sinteticas como el fixture; con datos de mercado es raro.")

    c1, c2 = st.columns(2)
    candidatos = d["candidatos"].copy()
    candidatos["elegido"] = ["si" if o == f"ARIMA{d['orden']}" else ""
                             for o in candidatos.index]
    c1.markdown(f"**Media: ARIMA{d['orden']}** (menor AIC)")
    c1.dataframe(candidatos.style.format(
        {"log-verosimilitud": "{:.2f}", "AIC": "{:.2f}"}), width="stretch")
    c2.markdown("**Varianza: GARCH(1,1) con t-Student**")
    gp = d["garch_params"]
    c2.dataframe(pd.DataFrame({"valor": [
        gp["omega"] / 1e4, gp["alpha[1]"], gp["beta[1]"],
        d["persistencia"], gp["nu"]]},
        index=["omega", "alfa", "beta", "alfa + beta", "nu (grados de "
               "libertad)"]).style.format("{:.4g}"), width="stretch")

    anual = np.sqrt(m)
    incond = (None if d["vol_incondicional"] is None
              else d["vol_incondicional"] * anual)
    k1, k2, k3 = st.columns(3)
    k1.metric("Volatilidad condicional (siguiente periodo)",
              f"{d['vol_condicional'] * anual:.1%}")
    k2.metric("Volatilidad incondicional GARCH",
              "no definida" if incond is None else f"{incond:.1%}")
    k3.metric("Volatilidad muestral (A y B)",
              f"{d['vol_muestral'] * anual:.1%}")
    vol = pd.Series(np.asarray(d["vol_historica"]) * anual, index=fechas)
    st.plotly_chart(charts.cond_vol_chart(
        vol, incond, d["vol_muestral"] * anual, ticker), width="stretch")
    st.caption(
        "Los modelos A y B usan una sola volatilidad para toda la muestra. "
        "El GARCH la deja cambiar: sube tras choques grandes y vuelve "
        "despacio hacia la incondicional. La proyeccion del modelo C parte "
        "de la volatilidad condicional de hoy. Todo anualizado con raiz de "
        f"{m} periodos.")
    with st.expander("Ver los parametros del ARIMA"):
        st.dataframe(d["arima_params"].rename("valor").to_frame()
                     .style.format("{:.4g}"), width="stretch")
