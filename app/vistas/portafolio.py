"""Vista Portafolio: frontera eficiente de Markowitz sin ventas en corto."""

import math

import pandas as pd
import streamlit as st

import src.data as dt
import src.portfolio as pf
from app import charts
from app.contexto import descargar_fx, descargar_tasa

# Nombre en la interfaz --> nombre en portfolio.key_portfolios
PORTAFOLIOS = {
    "Tangente": "tangente",
    "Minima varianza": "minima varianza",
    "Maxima rentabilidad": "maxima rentabilidad",
}


def controles(fuente):
    """Controles propios del portafolio en el panel lateral.

    Devuelve un diccionario con freq, rf_fixed y, con Yahoo, plazo y
    en_usd.
    """
    plazo = en_usd = None
    st.sidebar.divider()
    st.sidebar.caption("Portafolio")
    freq = st.sidebar.selectbox(
        "Frecuencia de los rendimientos", list(dt.PERIODOS_POR_ANIO),
        index=1, key="freq_portafolio",
        help="Semanal por defecto: con datos diarios, los horarios distintos "
             "de cada bolsa subestiman la covarianza entre activos de "
             "paises diferentes.")
    if fuente == "Yahoo Finance":
        plazo = st.sidebar.selectbox(
            "Plazo de la tasa libre de riesgo",
            ["Automatico"] + list(dt.TASAS), key="plazo_portafolio",
            help="Bono del Tesoro de EE.UU. Automatico elige el plazo mas "
                 "cercano al horizonte H.")
        en_usd = st.sidebar.checkbox(
            "Expresar todo en USD", True, key="usd_portafolio",
            help="Convierte los activos extranjeros a dolares: un portafolio "
                 "solo tiene sentido en una moneda.")
    rf_fixed = st.sidebar.number_input(
        "Tasa libre de riesgo (respaldo)" if fuente == "Yahoo Finance"
        else "Tasa libre de riesgo", 0.0, 0.20, 0.04, 0.0025, "%.4f",
        key="rf_portafolio",
        help="Tasa anual constante. Con Yahoo solo se usa si la descarga "
             "del bono falla.")
    return {"freq": freq, "plazo": plazo, "en_usd": en_usd,
            "rf_fixed": rf_fixed}


def _en_usd(precios, ctx):
    """Convierte a dolares los activos que cotizan en otra moneda.

    Devuelve (precios, avisos). Los activos cuya moneda no tiene tipo de
    cambio se descartan con un aviso.
    """
    monedas = {t: dt.market_info(t)[0] for t in precios.columns}
    extranjeras = set(monedas.values()) - {"USD"}
    if not extranjeras:
        return precios, []
    if not ctx.ajustes["en_usd"]:
        return precios, [
            "Hay activos en " + ", ".join(sorted(extranjeras)) + " que se "
            "combinan sin convertir: los pesos mezclan monedas."]

    fx, errores = descargar_fx(tuple(sorted(extranjeras)),
                               str(ctx.inicio), str(ctx.fin))
    avisos = [f"Sin tipo de cambio {c}=X ({motivo}): se omiten los activos "
              f"en {c}." for c, motivo in errores.items()]
    convertidos = {t: dt.to_usd(precios[t].dropna(), monedas[t], fx)
                   for t in precios.columns if monedas[t] not in errores}
    return pd.DataFrame(convertidos), avisos


def _tasa_libre(ctx):
    """Tasa libre de riesgo anual vigente y una etiqueta de su origen."""
    ajustes = ctx.ajustes
    if ctx.fuente != "Yahoo Finance":
        return ajustes["rf_fixed"], "constante", None

    plazo = (dt.tenor_for(ctx.H, ctx.m) if ajustes["plazo"] == "Automatico"
             else ajustes["plazo"])
    serie, motivo = descargar_tasa(plazo, str(ctx.inicio), str(ctx.fin))
    if motivo:
        return (ajustes["rf_fixed"], "respaldo",
                f"No se pudo descargar el bono a {plazo} ({motivo}): se usa "
                "la tasa de respaldo.")
    return float(serie.dropna().iloc[-1]), f"bono a {plazo}", None


def _sincronizar(origen, destino):
    """Copia el valor de un control al otro: barra y casilla del tope."""
    st.session_state[destino] = st.session_state[origen]


def _tope(contenedor, n):
    """Peso maximo por activo, con barra y casilla sincronizadas.

    n --> numero de activos elegidos
    Devuelve el tope en porcentaje, entero entre ceil(100 / n) y 100.

    Ambos controles comparten los mismos limites, asi que ninguno deja
    escribir ni arrastrar un valor fuera de rango. Antes de dibujarlos,
    el valor guardado se recorta a [minimo, 100]: si el usuario quita
    activos, el minimo sube y el tope anterior puede quedar por debajo.
    """
    minimo = math.ceil(100 / n)
    for clave in ("tope_barra", "tope_casilla"):
        valor = st.session_state.get(clave, 100)
        st.session_state[clave] = min(100, max(minimo, valor))
    ayuda = (f"Con {n} activos el tope no puede bajar de {minimo} %: por "
             "debajo no se alcanza a invertir el 100 %. En 100 % no hay "
             "tope y el portafolio puede concentrarse en pocos activos.")

    barra, casilla = contenedor.columns([3, 1])
    barra.slider("Peso maximo por activo (%)", minimo, 100, step=1,
                 key="tope_barra", help=ayuda, on_change=_sincronizar,
                 args=("tope_barra", "tope_casilla"))
    casilla.number_input("Valor exacto (%)", minimo, 100, step=1,
                         key="tope_casilla", on_change=_sincronizar,
                         args=("tope_casilla", "tope_barra"))
    return st.session_state["tope_barra"]


def render(ctx):
    """Dibuja la vista con el contexto comun y los ajustes de controles."""
    st.subheader("Construccion del portafolio")
    activos = st.multiselect(
        "Activos del portafolio", list(ctx.precios_diarios.columns),
        help="Solo se puede elegir entre los activos cargados. Para usar "
             "otros, agregalos en el panel lateral.")
    if len(activos) < 2:
        st.info("Elige al menos dos activos para construir el portafolio.")
        return

    c1, c2 = st.columns(2)
    tipo = c1.radio("Portafolio", list(PORTAFOLIOS), horizontal=True)
    tope = _tope(c2, len(activos))
    w_max = tope / 100

    precios, avisos = ctx.precios_diarios[activos], []
    if ctx.fuente == "Yahoo Finance":
        precios, avisos = _en_usd(precios, ctx)
    rf, origen_rf, aviso_rf = _tasa_libre(ctx)
    if aviso_rf:
        avisos.append(aviso_rf)
    for a in avisos:
        st.warning(a)
    if precios.shape[1] < 2:
        st.error("Quedan menos de dos activos con datos.")
        return

    freq = ctx.ajustes["freq"]
    m = dt.PERIODOS_POR_ANIO[freq]
    R = pf.returns_matrix(precios, freq)
    if len(R) < 20:
        st.error(f"Solo hay {len(R)} periodos en que todos los activos "
                 "cotizan: no alcanza para estimar la covarianza.")
        return
    mu, Sigma = pf.moments(R, m)

    pesos, resumen = pf.key_portfolios(mu, Sigma, rf, w_max)
    nombre = PORTAFOLIOS[tipo]
    if nombre not in pesos:
        st.error("Ningun activo rinde mas que la tasa libre de riesgo: no "
                 "existe portafolio tangente. Elige otro portafolio.")
        return
    elegido = resumen.loc[nombre]

    k1, k2 = st.columns(2)
    k1.metric("Rendimiento esperado", f"{elegido['return']:.2%}")
    k2.metric("Volatilidad", f"{elegido['volatility']:.2%}")

    frontera = pf.efficient_frontier(mu, Sigma, w_max=w_max)
    individuales = pd.DataFrame({"return": mu,
                                 "volatility": Sigma.to_numpy().diagonal()
                                 ** 0.5}, index=mu.index)
    st.plotly_chart(charts.frontier_chart(
        frontera, individuales, resumen.loc[[nombre]], nombre,
        rf if nombre == "tangente" else None), width="stretch")
    st.caption(
        f"{len(R)} periodos comunes, frecuencia {freq}, rendimientos simples "
        "anualizados. La frontera va del portafolio de minima varianza al de "
        "maxima rentabilidad: para cada rendimiento, la menor volatilidad "
        "posible sin ventas en corto"
        + (f" y sin superar {tope} % por activo." if tope < 100 else ".")
        + (f" El tangente se calcula con una tasa libre de riesgo de "
           f"{rf:.2%} ({origen_rf})." if nombre == "tangente" else ""))

    w = pesos[nombre]
    st.plotly_chart(charts.weights_chart(w[w > 0], tipo), width="stretch")

    with st.expander("Comparar los tres portafolios"):
        st.dataframe(resumen[["return", "volatility"]].rename(columns={
            "return": "rendimiento", "volatility": "volatilidad"})
            .style.format("{:.2%}"), width="stretch")
        st.dataframe(pesos.style.format("{:.2%}"), width="stretch")