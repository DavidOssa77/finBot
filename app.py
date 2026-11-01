"""Interfaz Streamlit del robot analitico para preseleccion de activos.

Punto de entrada: arma el panel lateral comun, carga los datos y le pasa
el contexto a la vista elegida. Cada vista vive en app/vistas/.
"""

from datetime import date

import streamlit as st

import src.contracts as ct
import src.data as dt
from app.contexto import Contexto, cargar_fixture, descargar
from app.vistas import VISTAS, INICIAL, PERFIL, SIN_PERFIL

FIXTURE = "notebooks/Fixture_20_activos_sintetico_TBS_EIA.csv"

st.set_page_config(page_title="Robot analitico", layout="wide")


def ir_al_perfil():
    """Cambia el menu a la vista del perfil; corre antes de redibujar."""
    st.session_state["vista"] = PERFIL


# Streamlit borra el estado de los widgets que no se dibujan; reescribir
# sus claves en cada ejecucion conserva lo elegido al cambiar de vista.
for registrada in VISTAS.values():
    for clave, inicial in getattr(registrada, "PERSISTENTES", {}).items():
        st.session_state[clave] = st.session_state.get(clave, inicial)

# ---------------------------------------------------------------- sidebar
st.sidebar.title("Robot analitico")
st.session_state.setdefault("vista", INICIAL)
vista = st.sidebar.radio("Vista", list(VISTAS), key="vista")
modulo = VISTAS[vista]

# Punto de entrada: sin perfil de riesgo guardado solo se abren las vistas
# informativas y la del cuestionario.
if vista not in SIN_PERFIL and "perfil" not in st.session_state:
    st.title("Robot analitico para preseleccion de activos")
    st.warning("Antes de usar el robot, responde el cuestionario de perfil "
               "de riesgo: con el se decide cuanto de tu capital va a "
               "activos riesgosos y cuanto a renta fija.")
    st.button("Ir a Perfil de riesgo", type="primary", on_click=ir_al_perfil)
    st.stop()

inicio = fin = None

st.sidebar.divider()
fuente = st.sidebar.radio("Fuente", ["Fixture del curso", "Yahoo Finance"])

if fuente == "Yahoo Finance":
    texto = st.sidebar.text_input("Tickers", "AAPL, MSFT, KO")
    tickers = tuple(t.strip().upper() for t in texto.split(",") if t.strip())
    inicio = st.sidebar.date_input("Desde", date(2021, 1, 1))
    fin = st.sidebar.date_input("Hasta", date.today())

st.sidebar.divider()
st.sidebar.caption("Parametros")
freq = st.sidebar.selectbox("Frecuencia", list(dt.PERIODOS_POR_ANIO))
H = st.sidebar.number_input("Horizonte H", min_value=1, value=20,
                            help="Periodos hacia adelante, en la frecuencia "
                                 "elegida. Un horizonte largo frente a la "
                                 "historia disponible deja sin validar el "
                                 "walk-forward.")
modelo = st.sidebar.selectbox(
    "Modelo", ["A", "B", "C"],
    help="A: caminata aleatoria sin deriva. B: con deriva. C: ARIMA-GARCH, "
         "media que depende del pasado y volatilidad condicional; tarda "
         "unos segundos por activo.")

params = ct.Params(
    cost_buy=st.sidebar.number_input("Comision compra", 0.0, 0.05, 0.0, 0.0005, "%.4f"),
    cost_sell=st.sidebar.number_input("Comision venta", 0.0, 0.05, 0.0, 0.0005, "%.4f"),
    BR_min=st.sidebar.number_input("BR minimo", 0.0, 5.0, 1.0, 0.1),
    confidence=st.sidebar.select_slider("Confianza VaR", [0.95, 0.99], 0.95),
    capital=st.sidebar.number_input("Capital", 1000, 1_000_000, 10_000, 1000),
)

ajustes = modulo.controles(fuente) if hasattr(modulo, "controles") else {}

# ------------------------------------------------------------------ datos
if fuente == "Fixture del curso":
    precios = cargar_fixture(FIXTURE)
    errores = {}
    origen = "Fixture sintetico del curso — precios simulados, no datos de mercado"
else:
    precios, errores = descargar(tickers, str(inicio), str(fin))
    origen = f"Yahoo Finance — {inicio} a {fin}"

if precios.empty:
    st.error("No se pudo cargar ningun activo.")
    st.stop()

precios_diarios = precios
precios = dt.resample_prices(precios, freq)
m = dt.PERIODOS_POR_ANIO[freq]

# --------------------------------------------------------------- cabecera
st.title("Robot analitico para preseleccion de activos")
st.caption(origen)

if errores:
    st.warning(
        f"No se pudieron descargar {len(errores)} de "
        f"{len(errores) + precios.shape[1]} tickers: "
        + ", ".join(errores)
        + ". El analisis continua con los que si respondieron.")
    with st.expander("Ver el motivo de cada fallo"):
        for t, motivo in errores.items():
            st.write(f"**{t}** — {motivo}")
        st.caption(
            "yfinance no distingue un simbolo inexistente de uno sin datos "
            "en el rango: en ambos casos devuelve una tabla vacia. Revisa la "
            "escritura del ticker y el periodo pedido.")

# ----------------------------------------------------------------- vista
modulo.render(Contexto(
    fuente=fuente, precios=precios, precios_diarios=precios_diarios,
    freq=freq, m=m, H=H, modelo=modelo, params=params, origen=origen,
    inicio=inicio, fin=fin, ajustes=ajustes))
