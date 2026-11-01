"""Vista Glosario: simbolos de Yahoo Finance que el robot reconoce."""

import pandas as pd
import streamlit as st

import src.data as dt

# Nombre del mercado (el de dt.MERCADOS) --> (simbolo, nombre). Todos se
# verificaron contra Yahoo en octubre de 2026: dos anios de historia y la
# moneda que espera dt.MERCADOS.
TICKERS = {
    "S&P 500": [
        ("AAPL", "Apple"), ("MSFT", "Microsoft"),
        ("GOOGL", "Alphabet (Google), clase A"), ("AMZN", "Amazon"),
        ("NVDA", "Nvidia"), ("META", "Meta Platforms"), ("TSLA", "Tesla"),
        ("JPM", "JPMorgan Chase"), ("BAC", "Bank of America"),
        ("V", "Visa"), ("MA", "Mastercard"), ("KO", "Coca-Cola"),
        ("PEP", "PepsiCo"), ("JNJ", "Johnson & Johnson"), ("PFE", "Pfizer"),
        ("UNH", "UnitedHealth"), ("XOM", "Exxon Mobil"), ("CVX", "Chevron"),
        ("WMT", "Walmart"), ("COST", "Costco"), ("MCD", "McDonald's"),
        ("HD", "Home Depot"), ("DIS", "Disney"), ("NFLX", "Netflix"),
        ("EC", "Ecopetrol (ADR en Nueva York)"),
        ("CIB", "Grupo Cibest - Bancolombia (ADR en Nueva York)"),
        ("AVAL", "Grupo Aval (ADR en Nueva York)"),
        ("SPY", "ETF S&P 500"), ("QQQ", "ETF Nasdaq 100"),
        ("DIA", "ETF Dow Jones"), ("IWM", "ETF Russell 2000"),
        ("VTI", "ETF mercado total de EE.UU."),
        ("EEM", "ETF mercados emergentes"), ("GLD", "ETF oro"),
        ("TLT", "ETF bonos del Tesoro a mas de 20 anios"),
    ],
    "COLCAP": [
        ("ECOPETROL.CL", "Ecopetrol"),
        ("PFCIBEST.CL", "Grupo Cibest (Bancolombia), preferencial"),
        ("CIBEST.CL", "Grupo Cibest (Bancolombia), ordinaria"),
        ("ISA.CL", "ISA"), ("GEB.CL", "Grupo Energia Bogota"),
        ("GRUPOSURA.CL", "Grupo Sura, ordinaria"),
        ("PFGRUPSURA.CL", "Grupo Sura, preferencial"),
        ("CELSIA.CL", "Celsia"), ("PFAVAL.CL", "Grupo Aval, preferencial"),
        ("BOGOTA.CL", "Banco de Bogota"),
        ("PFDAVVNDA.CL", "Davivienda, preferencial"),
        ("CORFICOLCF.CL", "Corficolombiana"), ("PROMIGAS.CL", "Promigas"),
        ("TERPEL.CL", "Terpel"), ("MINEROS.CL", "Mineros"),
        ("ICOLCAP.CL", "ETF iShares COLCAP"),
    ],
    "Nikkei 225": [
        ("7203.T", "Toyota"), ("6758.T", "Sony"),
        ("9984.T", "SoftBank Group"), ("8306.T", "Mitsubishi UFJ"),
        ("6861.T", "Keyence"),
    ],
    "Bovespa": [
        ("PETR4.SA", "Petrobras, preferencial"), ("VALE3.SA", "Vale"),
        ("ITUB4.SA", "Itau Unibanco, preferencial"),
        ("BBDC4.SA", "Bradesco, preferencial"), ("ABEV3.SA", "Ambev"),
    ],
    "IPC Mexico": [
        ("WALMEX.MX", "Walmart de Mexico"), ("AMXB.MX", "America Movil"),
        ("GFNORTEO.MX", "Banorte"), ("FEMSAUBD.MX", "FEMSA"),
        ("CEMEXCPO.MX", "Cemex"),
    ],
    "IPSA": [
        ("SQM-B.SN", "SQM, serie B"), ("FALABELLA.SN", "Falabella"),
        ("COPEC.SN", "Empresas Copec"), ("CHILE.SN", "Banco de Chile"),
        ("ENELAM.SN", "Enel Americas"),
    ],
    "S&P/TSX": [
        ("SHOP.TO", "Shopify"), ("RY.TO", "Royal Bank of Canada"),
        ("TD.TO", "Toronto-Dominion Bank"), ("ENB.TO", "Enbridge"),
        ("CNR.TO", "Canadian National Railway"),
    ],
    "FTSE 100": [
        ("HSBA.L", "HSBC"), ("BP.L", "BP"), ("SHEL.L", "Shell"),
        ("AZN.L", "AstraZeneca"), ("ULVR.L", "Unilever"),
    ],
    "DAX": [
        ("SAP.DE", "SAP"), ("SIE.DE", "Siemens"), ("BMW.DE", "BMW"),
        ("ALV.DE", "Allianz"), ("VOW3.DE", "Volkswagen, preferencial"),
    ],
    "CAC 40": [
        ("MC.PA", "LVMH"), ("OR.PA", "L'Oreal"), ("TTE.PA", "TotalEnergies"),
        ("AIR.PA", "Airbus"), ("SAN.PA", "Sanofi"),
    ],
    "IBEX 35": [
        ("SAN.MC", "Banco Santander"), ("IBE.MC", "Iberdrola"),
        ("ITX.MC", "Inditex"), ("BBVA.MC", "BBVA"),
        ("TEF.MC", "Telefonica"),
    ],
    "Hang Seng": [
        ("0700.HK", "Tencent"), ("9988.HK", "Alibaba"),
        ("0005.HK", "HSBC Holdings"), ("1299.HK", "AIA Group"),
        ("0941.HK", "China Mobile"),
    ],
}


def _tabla_mercados():
    """Una fila por sufijo de dt.MERCADOS: moneda e indice de referencia."""
    filas = [{"sufijo": sufijo or "(ninguno)", "mercado": nombre,
              "moneda": moneda, "indice en Yahoo": " o ".join(candidatos)}
             for sufijo, (moneda, nombre, candidatos) in dt.MERCADOS.items()]
    return pd.DataFrame(filas)


def render(ctx):
    """Dibuja la vista; no depende de los datos cargados."""
    st.subheader("Glosario de tickers")
    st.markdown(
        "El robot descarga los precios de **Yahoo Finance** con `yfinance`, "
        "asi que cada activo se escribe con su simbolo de Yahoo, separados "
        "por comas en el panel lateral (por ejemplo `AAPL, ECOPETROL.CL, "
        "SAP.DE`). Las acciones de EE.UU. no llevan sufijo; las de otras "
        "bolsas llevan el sufijo de su mercado. El sufijo decide la moneda "
        "del activo y el indice contra el que el CAPM mide su beta.")

    st.warning(
        "**Puedes usar simbolos que no esten en esta lista**, pero existe el "
        "riesgo de que yfinance no los reconozca: el simbolo puede estar mal "
        "escrito, la empresa puede haberse retirado de bolsa o haber "
        "cambiado de simbolo (Bancolombia paso de `PFBCOLOM.CL` a "
        "`PFCIBEST.CL`), o puede no tener historia en el rango de fechas. "
        "Si falla, el activo se omite con un aviso y el analisis sigue con "
        "los demas. Para encontrar un simbolo, buscalo en "
        "finance.yahoo.com y copialo exactamente, con su sufijo.")
    st.warning(
        "**Si el sufijo no esta en la tabla de mercados, el activo se trata "
        "como de EE.UU.**: en dolares y medido contra el S&P 500. Para una "
        "accion de otra bolsa (por ejemplo `.AX` Australia o `.SW` Suiza) "
        "la conversion a dolares y la beta del CAPM quedan mal.")

    st.subheader("Mercados reconocidos")
    st.dataframe(_tabla_mercados(), hide_index=True, width="stretch")
    st.caption(
        "Si Yahoo no publica el indice, se usa el siguiente candidato: un "
        "ETF que lo replica. Londres cotiza en peniques (GBp); los "
        "rendimientos no cambian.")

    st.subheader("Simbolos verificados")
    st.caption(
        "Todos respondieron en Yahoo en octubre de 2026 con al menos dos "
        "años de historia. El mismo nombre puede ser otra empresa en otra "
        "bolsa: `SAN.MC` es Banco Santander y `SAN.PA` es Sanofi.")
    pestanas = st.tabs(list(TICKERS))
    for pestana, mercado in zip(pestanas, TICKERS):
        tabla = pd.DataFrame(TICKERS[mercado], columns=["ticker", "nombre"])
        pestana.dataframe(tabla, hide_index=True, width="stretch")
