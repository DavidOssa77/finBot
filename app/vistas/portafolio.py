"""Vista Portafolio: frontera eficiente de Markowitz sin ventas en corto."""

import math

import pandas as pd
import streamlit as st

import src.black_litterman as bl
import src.capm as cm
import src.data as dt
import src.portfolio as pf
from app import charts
from app.contexto import (descargar_capitalizacion, descargar_fx,
                          descargar_mercado, descargar_tasa)

BENCHMARK = "S&P 500"
# Clave de performance_metrics, nombre en la tabla, formato y explicacion
METRICAS = (
    ("return", "Rendimiento anual", "{:.2%}", "Media historica anualizada"),
    ("volatility", "Volatilidad anual", "{:.2%}", "Riesgo total"),
    ("sharpe", "Sharpe", "{:.3f}", "Exceso por unidad de riesgo total"),
    ("sortino", "Sortino", "{:.3f}", "Exceso por unidad de riesgo a la baja"),
    ("beta", "Beta", "{:.3f}", "Sensibilidad al benchmark"),
    ("alpha", "Alfa de Jensen anual", "{:.2%}",
     "Rendimiento sobre lo que justifica su beta"),
    ("treynor", "Treynor", "{:.4f}",
     "Exceso por unidad de riesgo sistematico"),
    ("m2", "M2", "{:.2%}",
     "Rendimiento con la volatilidad del benchmark"),
    ("r2", "R2", "{:.3f}", "Parte sistematica de la varianza"),
    ("max_drawdown", "Caida maxima", "{:.2%}", "Peor caida desde un maximo"),
    ("var", "VaR 95 % (por periodo)", "{:.2%}",
     "Cuantil historico de perdida"),
    ("cvar", "CVaR 95 % (por periodo)", "{:.2%}",
     "Perdida media cuando se supera el VaR"),
)

FUENTES_MU = ("Historicos", "Black-Litterman")
COLUMNAS_VISTAS = ("tipo", "activo", "contra", "rendimiento (%)",
                   "confianza (%)")
# Controles que conservan su valor al cambiar de vista: clave --> valor
# inicial. app.py los vuelve a escribir en cada ejecucion para que
# Streamlit no los borre cuando esta vista no se dibuja; por eso los
# widgets no reciben valor inicial propio.
PERSISTENTES = {
    "freq_portafolio": "W-FRI",
    "plazo_portafolio": "Automatico",
    "usd_portafolio": True,
    "rf_portafolio": 0.04,
    "tope_barra": 100,
    "tope_casilla": 100,
    "activos_portafolio": [],
    "fuente_mu": FUENTES_MU[0],
    "pesos_mercado": "Iguales",
    "tau_bl": 0.05,
    "erp_bl": 0.1,
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
        key="freq_portafolio",
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
            "Expresar todo en USD", key="usd_portafolio",
            help="Convierte los activos extranjeros a dolares: un portafolio "
                 "solo tiene sentido en una moneda.")
    rf_fixed = st.sidebar.number_input(
        "Tasa libre de riesgo (respaldo)" if fuente == "Yahoo Finance"
        else "Tasa libre de riesgo", 0.0, 0.20, step=0.0025, format="%.4f",
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


def _pesos_mercado(activos, ctx):
    """Pesos de mercado de los activos del portafolio.

    Con Yahoo, el usuario elige entre capitalizacion (convertida a
    dolares) y pesos iguales. Si falta la capitalizacion o el tipo de
    cambio de algun activo, se usan pesos iguales para todos: mezclar
    capitalizaciones con pesos inventados no tiene sentido.
    Devuelve (Serie de pesos, descripcion del origen, aviso o None).
    """
    iguales = pd.Series(1 / len(activos), index=activos)
    if ctx.fuente != "Yahoo Finance":
        return (iguales, "pesos iguales (el fixture no tiene capitalizacion)",
                None)

    eleccion = st.radio("Pesos de mercado", ["Capitalizacion", "Iguales"],
                        horizontal=True, key="pesos_mercado",
                        help="Capitalizacion bursatil actual de Yahoo, "
                             "convertida a dolares.")
    if eleccion == "Iguales":
        return iguales, "pesos iguales", None

    caps, errores = descargar_capitalizacion(tuple(activos))
    if errores:
        return iguales, "pesos iguales", (
            "Sin capitalizacion para " + ", ".join(errores) + ": se usan "
            "pesos iguales para todos los activos.")

    monedas = {t: dt.market_info(t)[0] for t in activos}
    extranjeras = set(monedas.values()) - {"USD"}
    if extranjeras:
        fx, err_fx = descargar_fx(tuple(sorted(extranjeras)),
                                  str(ctx.inicio), str(ctx.fin))
        if err_fx:
            return iguales, "pesos iguales", (
                "Sin tipo de cambio para " + ", ".join(err_fx) + ": no se "
                "pueden comparar capitalizaciones; se usan pesos iguales.")
        for t in activos:
            if monedas[t] != "USD":
                caps[t] = caps[t] / fx[monedas[t]].dropna().iloc[-1]
    return bl.market_weights(caps[activos]), "capitalizacion en USD", None


def _leer_vistas(tabla, rf):
    """Convierte las filas del editor en visiones para black_litterman.

    En las absolutas el usuario escribe el rendimiento TOTAL esperado y
    aqui se le resta rf: el modelo trabaja con excesos. En las relativas
    escribe la diferencia entre los dos activos, que no cambia.
    Devuelve (vistas, avisos). Las filas vacias se ignoran sin aviso; las
    incompletas se ignoran con aviso.
    """
    vistas, avisos = [], []
    for i, fila in tabla.reset_index(drop=True).iterrows():
        if fila.isna().all():
            continue
        tipo, activo, contra = fila["tipo"], fila["activo"], fila["contra"]
        r, c = fila["rendimiento (%)"], fila["confianza (%)"]
        incompleta = (pd.isna(tipo) or pd.isna(activo) or pd.isna(r)
                      or pd.isna(c)
                      or (tipo == "relativa" and pd.isna(contra)))
        if incompleta:
            avisos.append(f"La vision {i + 1} esta incompleta y se ignora.")
            continue
        vistas.append({
            "tipo": tipo, "activo": activo,
            "contra": contra if tipo == "relativa" else None,
            "q": r / 100 - rf if tipo == "absoluta" else r / 100,
            "confianza": c / 100})
    return vistas, avisos


def _black_litterman(mu, Sigma, rf, ctx):
    """Seccion de Black-Litterman: parametros, visiones y resultado.

    mu, Sigma --> momentos historicos anuales de los activos elegidos
    Devuelve (mu, Sigma) de Black-Litterman en rendimientos totales,
    listos para el optimizador, o None si las visiones tienen errores.
    """
    st.subheader("Black-Litterman")
    activos = list(Sigma.index)
    w_mkt, origen_w, aviso_w = _pesos_mercado(activos, ctx)
    if aviso_w:
        st.warning(aviso_w)

    p1, p2 = st.columns(2)
    tau = p1.number_input(
        "Tau", 0.005, 1.0, step=0.005, format="%.3f", key="tau_bl",
        help="Incertidumbre del equilibrio. Valores tipicos: 0.025 a 0.05. "
             "No cambia los rendimientos (cuanto pesa cada vision lo fija "
             "su confianza); solo agranda la covarianza, (1 + tau) Sigma "
             "sin visiones.")
    erp = p2.number_input(
        "Prima de riesgo de mercado", 0.0, 0.15, step=0.0025,
        format="%.4f", key="erp_bl",
        help="E[Rm] - rf anual. Fija la aversion al riesgo del mercado y, "
             "con ella, los rendimientos de equilibrio.")

    st.markdown(
        "**Visiones.** Absoluta: rendimiento total anual esperado del "
        "activo. Relativa: cuanto rinde *activo* por encima de *contra*. "
        "La confianza dice que fraccion del camino se recorre desde el "
        "equilibrio hacia la vision.")
    # El estado del editor no se puede reescribir como el de los demas
    # controles. Se guarda la tabla resultante en "vistas_guardadas" y,
    # cuando Streamlit borro el editor (se volvio a la vista), se usa como
    # tabla de partida; mientras se edita, la partida no cambia.
    if "vistas_bl" not in st.session_state:
        st.session_state["vistas_base"] = st.session_state.get(
            "vistas_guardadas",
            pd.DataFrame({c: pd.Series(dtype="object")
                          for c in COLUMNAS_VISTAS[:3]}
                         | {c: pd.Series(dtype="float")
                            for c in COLUMNAS_VISTAS[3:]}))
    tabla = st.data_editor(
        st.session_state["vistas_base"], num_rows="dynamic",
        key="vistas_bl", width="stretch", hide_index=True,
        column_config={
            "tipo": st.column_config.SelectboxColumn(
                "tipo", options=list(bl.TIPOS), required=True),
            "activo": st.column_config.SelectboxColumn(
                "activo", options=activos, required=True),
            "contra": st.column_config.SelectboxColumn(
                "contra", options=activos,
                help="Solo en las visiones relativas"),
            "rendimiento (%)": st.column_config.NumberColumn(
                "rendimiento (%)", min_value=-100.0, max_value=200.0,
                step=0.5, format="%.2f"),
            "confianza (%)": st.column_config.NumberColumn(
                "confianza (%)", min_value=1.0, max_value=100.0, step=5.0,
                format="%.0f"),
        })
    st.session_state["vistas_guardadas"] = tabla.reset_index(drop=True)
    vistas, avisos = _leer_vistas(tabla, rf)
    for a in avisos:
        st.warning(a)

    try:
        r = bl.black_litterman(Sigma, w_mkt, erp, vistas, tau)
    except ValueError as e:
        st.error(f"Revisa las visiones: {e}")
        return None

    mu_bl = r["mu_BL"] + rf
    st.plotly_chart(charts.bl_returns_chart(pd.DataFrame({
        "historico": mu, "equilibrio": r["Pi"] + rf,
        "black_litterman": mu_bl})), width="stretch")
    st.caption(
        f"Pesos de mercado: {origen_w}. Aversion al riesgo implicita "
        f"delta = {r['delta']:.2f}. El equilibrio es el CAPM sobre estos "
        "activos; sin visiones, Black-Litterman coincide con el y el "
        "portafolio tangente es el de mercado. Todos los rendimientos son "
        f"totales: excesos mas rf = {rf:.2%}.")

    with st.expander("Ver el detalle"):
        st.dataframe(pd.DataFrame({
            "peso de mercado": w_mkt, "historico": mu,
            "equilibrio": r["Pi"] + rf, "Black-Litterman": mu_bl,
        }).style.format("{:.2%}"), width="stretch")

    return mu_bl, r["Sigma_BL"]


def _asignacion(mu, Sigma, rf, w_max, resumen):
    """Portafolio del inversionista con el A guardado: titulo y metricas.

    mu, Sigma --> momentos anuales con los que se construyo el portafolio
    resumen --> portafolios de referencia de key_portfolios
    Devuelve el diccionario de portfolio.optimal_allocation. El perfil lo
    guarda la vista Perfil de riesgo; app.py no deja llegar aqui sin el.
    """
    perfil = st.session_state["perfil"]
    A = perfil["A"]
    st.subheader(f"Tu portafolio (perfil {perfil['nombre']}, A = {A:.1f})")
    r = pf.optimal_allocation(mu, Sigma, rf, A, w_max)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Rendimiento esperado", f"{r['return']:.2%}")
    k2.metric("Volatilidad", f"{r['volatility']:.2%}")
    k3.metric("y*", f"{r['y']:.2f}")
    k4.metric("Renta fija", f"{r['w_rf']:.1%}")

    if "tangente" not in resumen.index:
        st.info("Ningun activo rinde mas que la tasa libre de riesgo: no "
                "existe portafolio tangente y todo el capital va a renta "
                "fija.")
        return r
    t = resumen.loc["tangente"]
    A_corte = (t["return"] - rf) / t["volatility"] ** 2
    if r["y"] > 1:
        st.info(
            f"y* = {r['y']:.2f} > 1: este inversionista querria endeudarse "
            "para invertir mas que su capital en el tangente. Sin "
            "apalancamiento, invierte el 100 % en activos riesgosos y se "
            "mueve por la frontera hacia la maxima rentabilidad.")
    st.caption(
        "Separacion de Tobin: tu aversion al riesgo decide cuanto va al "
        "portafolio tangente y cuanto a renta fija, sin endeudarse. "
        f"y* = (E[Rt] - rf) / (A sigma_t^2) con rf = {rf:.2%}; con estos "
        f"activos, y* = 1 cuando A = {A_corte:.1f}. Para cambiar A, vuelve "
        "a la vista Perfil de riesgo.")
    return r


def _benchmark(precios, ctx):
    """Precios del benchmark, su nombre y un aviso o None.

    Con Yahoo es el S&P 500; con el fixture, o si la descarga falla, un
    proxy equiponderado de los activos, en la moneda del portafolio.
    """
    if ctx.fuente == "Yahoo Finance":
        indices, errores = descargar_mercado(
            (BENCHMARK,), str(ctx.inicio), str(ctx.fin))
        if not errores:
            simbolo, serie = indices[BENCHMARK]
            return serie, f"{BENCHMARK} ({simbolo})", None
        return (cm.equal_weight_index(precios), "proxy equiponderado",
                f"No se pudo descargar el {BENCHMARK}: se compara contra un "
                "proxy equiponderado de los activos elegidos.")
    return (cm.equal_weight_index(ctx.precios_diarios),
            "proxy equiponderado del fixture", None)


def _texto(valor, formato):
    """Valor con su formato, o n/d si la razon no esta definida."""
    if pd.isna(valor):
        return "n/d"
    # Un residuo de redondeo como -1e-17 no debe mostrarse como -0.00 %
    return formato.format(0.0 if abs(valor) < 1e-12 else valor)


def _metricas(R, pesos, r, rf, m, precios, ctx):
    """Tabla de desempeño ex post de los portafolios contra el benchmark.

    R --> rendimientos simples historicos de los activos
    pesos --> pesos de los portafolios de referencia
    r --> asignacion del inversionista
    precios --> precios de los activos, para fechar el benchmark
    """
    st.subheader("Metricas de desempeño")
    serie, nombre, aviso = _benchmark(precios, ctx)
    if aviso:
        st.warning(aviso)
    Rm = pf.benchmark_returns(serie, precios, ctx.ajustes["freq"])

    columnas = {"tu portafolio": pf.portfolio_returns(
        R, r["weights"], rf / m)}
    for portafolio in pesos:
        columnas[portafolio] = pf.portfolio_returns(R, pesos[portafolio])
    columnas["benchmark"] = Rm
    resultado = {c: pf.performance_metrics(rp, Rm, rf, m)
                 for c, rp in columnas.items()}

    tabla = pd.DataFrame(
        {c: [_texto(x[clave], formato) for clave, _, formato, _ in METRICAS]
         for c, x in resultado.items()},
        index=[metrica[1] for metrica in METRICAS])
    tabla.insert(0, "que mide", [metrica[3] for metrica in METRICAS])
    # Alto de todas las filas mas el encabezado (35 px cada una), sin barra
    st.dataframe(tabla, width="stretch", height=35 * (len(tabla) + 1) + 3)
    st.caption(
        f"Benchmark: {nombre}. Medidas ex post: cada portafolio mantiene "
        "sus pesos fijos sobre los rendimientos historicos, rebalanceando "
        f"cada periodo sin costos, con rf = {rf:.2%}. Siempre se miden con "
        "lo que realmente ocurrio, aunque los pesos vengan de "
        "Black-Litterman. n/d: la razon no esta definida, como con todo el "
        "capital en renta fija.")


def render(ctx):
    """Dibuja la vista con el contexto comun y los ajustes de controles."""
    st.subheader("Construccion del portafolio")
    # Si cambiaron los tickers del panel, la eleccion guardada puede traer
    # activos que ya no estan cargados: se quitan antes de dibujar.
    cargados = list(ctx.precios_diarios.columns)
    st.session_state["activos_portafolio"] = [
        t for t in st.session_state["activos_portafolio"] if t in cargados]
    activos = st.multiselect(
        "Activos del portafolio", cargados, key="activos_portafolio",
        help="Solo se puede elegir entre los activos cargados. Para usar "
             "otros, agregalos en el panel lateral.")
    if len(activos) < 2:
        st.info("Elige al menos dos activos para construir el portafolio.")
        return

    fuente_mu = st.radio(
        "Rendimientos esperados", FUENTES_MU, horizontal=True,
        key="fuente_mu",
        help="Historicos: media de la muestra. Black-Litterman: equilibrio "
             "del mercado ajustado por las visiones que ingreses.")
    tope = _tope(st.container(), len(activos))
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
    with st.expander("Matriz de correlaciones"):
        st.plotly_chart(charts.correlation_chart(R.corr()), width="stretch")
        st.caption(
            f"Correlacion de Pearson de los rendimientos en los {len(R)} "
            f"periodos comunes, frecuencia {freq}.")
    if fuente_mu == "Black-Litterman":
        resultado = _black_litterman(mu, Sigma, rf, ctx)
        if resultado is None:
            return
        mu, Sigma = resultado

    pesos, resumen = pf.key_portfolios(mu, Sigma, rf, w_max)
    r = _asignacion(mu, Sigma, rf, w_max, resumen)

    # Un solo grafico: frontera, activos, los tres portafolios de
    # referencia, la linea del mercado de capitales y el inversionista.
    frontera = pf.efficient_frontier(mu, Sigma, w_max=w_max)
    individuales = pd.DataFrame({"return": mu,
                                 "volatility": Sigma.to_numpy().diagonal()
                                 ** 0.5}, index=mu.index)
    puntos = resumen.copy()
    puntos.loc["inversionista"] = {
        "return": r["return"], "volatility": r["volatility"],
        "sharpe": float("nan")}
    st.plotly_chart(charts.frontier_chart(
        frontera, individuales, puntos, "inversionista", rf),
        width="stretch")
    st.caption(
        f"{len(R)} periodos comunes, frecuencia {freq}, rendimientos simples "
        "anualizados"
        + (", esperados de Black-Litterman." if fuente_mu != "Historicos"
           else ".")
        + " La frontera va del portafolio de minima varianza al de "
        "maxima rentabilidad: para cada rendimiento, la menor volatilidad "
        "posible sin ventas en corto"
        + (f" y sin superar {tope} % por activo." if tope < 100 else ".")
        + f" Tasa libre de riesgo: {rf:.2%} ({origen_rf}).")

    final = pd.concat([r["weights"], pd.Series({"renta fija": r["w_rf"]})])
    st.plotly_chart(charts.weights_chart(final[final > 1e-9],
                                         "tu portafolio"), width="stretch")

    _metricas(R, pesos, r, rf, m, precios, ctx)

    with st.expander("Comparar los portafolios de referencia"):
        st.dataframe(resumen[["return", "volatility"]].rename(columns={
            "return": "rendimiento", "volatility": "volatilidad"})
            .style.format("{:.2%}"), width="stretch")
        st.dataframe(pesos.style.format("{:.2%}"), width="stretch")
