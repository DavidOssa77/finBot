"""Vista CAPM: beta contra el indice de mercado, LMV y rendimiento requerido."""

import pandas as pd
import streamlit as st

import src.capm as cm
import src.contracts as ct
import src.data as dt
from app import charts
from app.contexto import descargar_mercado, descargar_tasa, descargar_fx


def controles(fuente):
    """Controles propios del CAPM en el panel lateral.

    Se dibujan antes de cargar los datos, debajo de los parametros comunes.
    Devuelve un diccionario con params_capm y, con Yahoo, plazo, indice
    y en_usd.
    """
    plazo = indice = en_usd = None
    st.sidebar.divider()
    st.sidebar.caption("CAPM")
    freq_capm = st.sidebar.selectbox(
        "Frecuencia de la regresion", list(dt.PERIODOS_POR_ANIO), index=1,
        help="Semanal por defecto: con datos diarios, la diferencia de "
             "horarios entre bolsas sesga la beta hacia abajo.")
    if fuente == "Yahoo Finance":
        plazo = st.sidebar.selectbox(
            "Plazo de la tasa libre de riesgo",
            ["Automatico"] + list(dt.TASAS),
            help="Bono del Tesoro de EE.UU. Automatico elige el plazo mas "
                 "cercano al horizonte H. Los excesos de la regresion usan "
                 "siempre la letra a 13 semanas.")
        indice = st.sidebar.selectbox(
            "Indice de mercado",
            ["Automatico por pais"] + [v[1] for v in dt.MERCADOS.values()],
            help="Automatico asigna a cada activo el indice de su bolsa "
                 "segun el sufijo del ticker (.CL Colombia, .T Japon...).")
        en_usd = st.sidebar.checkbox(
            "Expresar todo en USD", True,
            help="Convierte activos e indices extranjeros a dolares, para "
                 "que sean consistentes con la tasa del Tesoro.")
    metodo_rm = st.sidebar.radio(
        "Rendimiento esperado del mercado", ["prima", "historico"],
        format_func={"prima": "rf + prima de riesgo",
                     "historico": "media historica del indice"}.get)
    params_capm = ct.CapmParams(
        freq=freq_capm,
        erp=st.sidebar.number_input(
            "Prima de riesgo de mercado", 0.0, 0.15, 0.05, 0.0025, "%.4f",
            disabled=metodo_rm != "prima",
            help="E[Rm] - rf anual. Referencia: la prima implicita que "
                 "publica Damodaran para EE.UU. ronda 4-5 %."),
        market_method=metodo_rm,
        beta_kind=st.sidebar.radio("Beta para el rendimiento requerido",
                                   ["ajustada", "cruda"]),
        rf_fixed=st.sidebar.number_input(
            "Tasa libre de riesgo (respaldo)" if fuente == "Yahoo Finance"
            else "Tasa libre de riesgo", 0.0, 0.20, 0.04, 0.0025, "%.4f",
            help="Tasa anual constante. Con Yahoo solo se usa si la "
                 "descarga del bono falla."),
    )
    return {"params_capm": params_capm, "plazo": plazo, "indice": indice,
            "en_usd": en_usd}


def render(ctx):
    """Dibuja la vista con el contexto comun y los ajustes de controles."""
    fuente, precios_diarios = ctx.fuente, ctx.precios_diarios
    inicio, fin, H, m = ctx.inicio, ctx.fin, ctx.H, ctx.m
    params_capm = ctx.ajustes["params_capm"]
    plazo, indice, en_usd = (ctx.ajustes[k] for k in ("plazo", "indice", "en_usd"))
    avisos = []
    if fuente == "Fixture del curso":
        proxy = "Proxy equiponderado"
        activos = {t: precios_diarios[t].dropna() for t in precios_diarios}
        indice_de = {t: proxy for t in activos}
        moneda_de = {t: "USD" for t in activos}
        indices = {proxy: cm.equal_weight_index(precios_diarios)}
        simbolos = {proxy: "promedio de los activos"}
        rf_serie = rf_now = params_capm.rf_fixed
        plazo_usado = "constante"
        avisos.append(
            "El fixture no trae indice de mercado: se usa como proxy un "
            "portafolio equiponderado de los activos cargados, y cada activo "
            "forma parte del mercado contra el que se mide. La tasa libre de "
            "riesgo es la constante del panel.")
    else:
        activos = {t: precios_diarios[t].dropna() for t in precios_diarios}
        moneda_de = {t: dt.market_info(t)[0] for t in activos}
        indice_de = {t: dt.market_info(t)[1] if indice == "Automatico por pais"
                     else indice for t in activos}
        moneda_indice = {v[1]: v[0] for v in dt.MERCADOS.values()}

        descargados, err_idx = descargar_mercado(
            tuple(sorted(set(indice_de.values()))), str(inicio), str(fin))
        indices = {n: s for n, (_, s) in descargados.items()}
        simbolos = {n: sym for n, (sym, _) in descargados.items()}
        for n, motivo in err_idx.items():
            avisos.append(f"No se pudo descargar el indice {n} ({motivo}).")

        extranjeras = ({moneda_de[t] for t in activos}
                       | {moneda_indice[n] for n in indices}) - {"USD"}
        if en_usd and extranjeras:
            fx, err_fx = descargar_fx(tuple(sorted(extranjeras)),
                                      str(inicio), str(fin))
            for c, motivo in err_fx.items():
                avisos.append(f"Sin tipo de cambio {c}=X ({motivo}): se "
                              f"omiten los activos e indices en {c}.")
            activos = {t: dt.to_usd(s, moneda_de[t], fx)
                       for t, s in activos.items() if moneda_de[t] not in err_fx}
            indices = {n: dt.to_usd(s, moneda_indice[n], fx)
                       for n, s in indices.items()
                       if moneda_indice[n] not in err_fx}
            moneda_de = {t: "USD" for t in activos}
        elif extranjeras:
            avisos.append(
                "Hay activos o indices en " + ", ".join(sorted(extranjeras))
                + " que se analizan en moneda local contra una tasa del "
                "Tesoro en dolares: el exceso de rendimiento mezcla monedas.")

        plazo_usado = dt.tenor_for(H, m) if plazo == "Automatico" else plazo
        rf_serie, motivo = descargar_tasa("13 semanas", str(inicio), str(fin))
        if motivo:
            avisos.append(f"No se pudo descargar la letra a 13 semanas "
                          f"({motivo}): los excesos usan la tasa de respaldo.")
            rf_serie = params_capm.rf_fixed
        serie_plazo, motivo = descargar_tasa(plazo_usado, str(inicio), str(fin))
        if motivo:
            avisos.append(f"No se pudo descargar el bono a {plazo_usado} "
                          f"({motivo}): se usa la tasa de respaldo.")
            rf_now = params_capm.rf_fixed
        else:
            rf_now = float(serie_plazo.dropna().iloc[-1])

    resultados, omitidos = {}, {}
    for t, serie in activos.items():
        n = indice_de[t]
        if n not in indices:
            omitidos[t] = f"sin indice {n}"
            continue
        try:
            r = cm.analyze_asset(serie, indices[n], rf_serie, rf_now,
                                 params_capm)
        except Exception as e:
            omitidos[t] = f"la regresion fallo: {e}"
            continue
        if r["n"] < 20:
            omitidos[t] = f"solo {r['n']} periodos en comun con el indice"
            continue
        resultados[t] = r

    for a in avisos:
        st.warning(a)
    if omitidos:
        st.warning("Activos sin CAPM: " + "; ".join(
            f"{t} ({motivo})" for t, motivo in omitidos.items()))
    if not resultados:
        st.error("Ningun activo tiene datos suficientes para el CAPM.")
        st.stop()

    tabla = cm.capm_table(resultados, indice_de, moneda_de)
    tabla[tabla.columns[2:]] = tabla[tabla.columns[2:]].apply(pd.to_numeric)

    st.subheader("Tasa libre de riesgo y mercado")
    c1, c2, c3 = st.columns(3)
    c1.metric("Tasa libre de riesgo", f"{rf_now:.2%}",
              f"plazo {plazo_usado}", delta_color="off", delta_arrow="off")
    por_mercado = {indice_de[t]: r["E_Rm"] for t, r in resultados.items()}
    if len(set(por_mercado.values())) == 1:
        E_Rm = next(iter(por_mercado.values()))
        c2.metric("E[Rm]", f"{E_Rm:.2%}",
                  "rf + prima" if params_capm.market_method == "prima"
                  else "media historica",
                  delta_color="off", delta_arrow="off")
        c3.metric("Prima de mercado", f"{E_Rm - rf_now:.2%}")
    else:
        c2.metric("E[Rm]", "varia por indice")
        c3.write("  \n".join(f"**{n}**: {v:.2%}"
                              for n, v in por_mercado.items()))
    st.caption("Indices usados: " + ", ".join(
        f"{n} ({simbolos.get(n, '?')})" for n in sorted(set(tabla["market"]))))

    st.plotly_chart(charts.sml_chart(tabla), width="stretch")
    st.caption(
        "LMV ex post: exceso de rendimiento realizado contra beta. La recta "
        "une el activo libre de riesgo con el mercado realizado, asi que la "
        "distancia vertical de cada activo a su recta es su alfa de Jensen. "
        "Estar por encima significa que el activo rindio mas de lo que su "
        "riesgo sistematico justificaba en esta muestra, no que vaya a "
        "seguir haciendolo.")

    st.subheader("Tabla CAPM")
    nombres = {
        "market": "indice", "currency": "moneda", "n": "periodos",
        "beta": "beta", "beta_adj": "beta ajustada", "beta_p": "p beta",
        "alpha_annual": "alfa anual", "alpha_p": "p alfa", "r2": "R2",
        "vol_idiosyncratic": "vol. idiosincratica",
        "mean_annual": "rend. historico", "volatility_annual": "volatilidad",
        "required_return": "rend. requerido", "sharpe": "Sharpe",
        "treynor": "Treynor"}
    vista_tabla = tabla[list(nombres)].rename(columns=nombres)
    st.dataframe(
        vista_tabla.sort_values("beta", ascending=False).style.format({
            "beta": "{:.3f}", "beta ajustada": "{:.3f}", "p beta": "{:.4f}",
            "alfa anual": "{:.2%}", "p alfa": "{:.4f}", "R2": "{:.3f}",
            "vol. idiosincratica": "{:.2%}", "rend. historico": "{:.2%}",
            "volatilidad": "{:.2%}", "rend. requerido": "{:.2%}",
            "Sharpe": "{:.3f}", "Treynor": "{:.4f}"}),
        width="stretch")
    st.caption(
        f"Rendimientos simples, frecuencia {params_capm.freq}, anualizados. "
        f"El rendimiento requerido usa la beta {params_capm.beta_kind}: "
        "rf + beta (E[Rm] - rf). Un p-valor de alfa alto significa que no "
        "se puede distinguir el alfa de cero.")

    st.subheader("Detalle por activo")
    ticker = st.selectbox("Activo", list(resultados))
    r = resultados[ticker]
    st.plotly_chart(charts.regression_chart(
        r["excess"], r["alpha_period"], r["beta"], ticker, indice_de[ticker]),
        width="stretch")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Beta", f"{r['beta']:.3f}",
              f"ajustada {r['beta_adj']:.3f}",
              delta_color="off", delta_arrow="off",
              help="Ajuste de Blume: 0.67 beta + 0.33. Las betas tienden a "
                   "regresar a 1 con el tiempo.")
    d2.metric("Alfa de Jensen anual", f"{r['alpha_annual']:.2%}",
              f"p-valor {r['alpha_p']:.3f}",
              delta_color="off", delta_arrow="off")
    d3.metric("R2", f"{r['r2']:.3f}",
              f"vol. idiosincratica {r['vol_idiosyncratic']:.2%}",
              delta_color="off", delta_arrow="off",
              help="Fraccion de la varianza del activo que explica el "
                   "mercado: su riesgo sistematico. El resto es riesgo "
                   "idiosincratico, el que se puede eliminar diversificando "
                   "y que el CAPM no remunera; su volatilidad anual es la "
                   "desviacion de los residuos de la regresion.")
    d4.metric("Rendimiento requerido", f"{r['required_return']:.2%}",
              f"historico {r['mean_annual']:.2%}",
              delta_color="off", delta_arrow="off")
