"""Vista Resumen: señal, niveles, riesgo y validacion temporal del activo."""

import numpy as np
import streamlit as st

from .comun import analisis_activo


def render(ctx):
    """Dibuja la vista con el contexto comun."""
    _, _, r = analisis_activo(ctx)
    params = ctx.params
    puerta = r["gate"]
    niveles = r["levels"]

    if puerta["signal"]:
        st.success(f"SEÑAL — pronostico {puerta['label']}")
    else:
        st.error("SIN SEÑAL")
        st.write("Reglas incumplidas:")
        for regla in puerta["failed"]:
            st.write(f"- {regla}")

    if puerta["warnings"]:
        st.warning("Advertencias: " + ", ".join(puerta["warnings"]))

    st.subheader("Niveles")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Entrada", f"{niveles['E']:.2f}")
    c2.metric("Stop-loss", f"{niveles['SL']:.2f}",
              f"{niveles['SL'] / niveles['E'] - 1:.2%}")
    c3.metric("Take-profit", f"{niveles['TP']:.2f}",
              f"{niveles['TP'] / niveles['E'] - 1:.2%}")
    c4.metric("Equilibrio", f"{niveles['P_BE']:.2f}")

    st.subheader("Riesgo y recompensa")
    d1, d2, d3 = st.columns(3)
    br = niveles["BR_neto"]
    d1.metric("BR neto", f"{br:.3f}" if br is not None else "n/d")
    d2.metric(f"VaR {params.confidence:.0%}", f"{r['var_frac']:.2%}",
              f"-{r['var_money']:,.0f} de {params.capital:,.0f}",
              delta_color="inverse")
    d3.metric("Pr(ganar)", f"{r['probs']['win']:.1%}")

    st.subheader("Validacion temporal")
    wf = r["walk_forward"]
    if wf["sufficient"]:
        e1, e2, e3 = st.columns(3)
        e1.metric("Cobertura", f"{wf['coverage']:.2f}",
                  f"{len(wf['origins'])} origenes solapados",
                  delta_color="off",
                  help=f"Proporcion de origenes donde el rendimiento real cayo "
                       f"dentro de la banda {params.p_L:.0%}-{params.p_U:.0%}. "
                       "Los origenes son consecutivos, asi que sus ventanas "
                       "comparten casi todas sus observaciones: la metrica "
                       "tiende a 0 o a 1 y no mide calibracion.")
        sd_H = np.sqrt(r["v_H"])
        e2.metric("MAE", f"{wf['mae']:.2%}",
                  f"{wf['mae'] / sd_H:.2f}x la volatilidad a H",
                  delta_color="off",
                  help="Error absoluto medio del pronostico central contra el "
                       "rendimiento logaritmico acumulado que realmente ocurrio, "
                       "promediado sobre los origenes.")
        direccion = wf["direction"]
        n_dir = wf["direction_n"]
        e3.metric("Direccion",
                  f"{direccion:.0%}" if direccion is not None else "n/d",
                  f"{round(direccion * n_dir)} de {n_dir} origenes"
                  if direccion is not None else "el modelo A no tiene deriva",
                  delta_color="off",
                  help="Proporcion de origenes donde el pronostico central "
                       "acerto el signo del rendimiento acumulado. El modelo A "
                       "impone mu = 0, asi que no hay signo que acertar y la "
                       "metrica queda sin definir. En el modelo B el pronostico "
                       "apunta siempre en el sentido de la deriva estimada, asi "
                       "que la metrica termina midiendo cuantas ventanas se "
                       "movieron en ese sentido, no la capacidad predictiva del "
                       "modelo. En el modelo C el signo depende tambien de "
                       "los ultimos rendimientos (la parte ARIMA), asi que "
                       "si mide algo de capacidad predictiva.")
    else:
        st.info(f"Muestra insuficiente: {wf['T']} rendimientos, "
                f"hacen falta {wf['required']}.")
