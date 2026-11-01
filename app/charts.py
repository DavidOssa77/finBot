"""Figuras de Plotly para la interfaz."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

TEMA = "plotly_white"

def price_chart(prices, drawdown, ticker):
    """Precio y caida desde maximos, en dos paneles con eje de fechas comun

    prices --> Serie de precios ajustados
    drawdown --> Serie D_t, la primera salida de analytics.drawdown
    ticker --> nombre del activo, solo para el titulo

    Devuelve una figura de Plotly; no la muestra
    """
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3], vertical_spacing=0.07,
                        subplot_titles=("precio ajustado", "caida desde maximos (drawdown)"))

    fig.add_trace(go.Scatter(x=prices.index, y=prices, name="precio",
                             line=dict(width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown, name="drawdown",
                             fill="tozeroy", line=dict(width=1)), row=2, col=1)

    fig.update_yaxes(title_text="USD", row=1, col=1)
    fig.update_yaxes(title_text="caida", tickformat=".0%", row=2, col=1)
    fig.update_layout(template=TEMA, height=540, showlegend=False,
                      hovermode="x unified",
                      title=f"{ticker} - precio y drawdown")
    return fig

def comparison_map(comp):
    """Mapa media-volatilidad con la frontera de activos no dominados.

    comp --> DataFrame como el que devuelve compare.preselect, indexado
             por ticker y con las columnas volatility_annual,
             mean_log_annual, individual_rvr, non_dominated y
             selected_max_rvr
    """
    nd = comp["non_dominated"]
    fig = go.Figure()

    for sub, nombre, simbolo, tam in ((comp[~nd], "dominados", "circle-open", 9),
                                      (comp[nd], "no dominados", "circle", 13)):
        posiciones = ["middle right" if sel else "top center"
                      for sel in sub["selected_max_rvr"]]
        fig.add_trace(go.Scatter(
            x=sub["volatility_annual"], y=sub["mean_log_annual"],
            mode="markers+text", text=sub.index, textposition=posiciones,
            textfont=dict(size=10), name=nombre,
            marker=dict(size=tam, symbol=simbolo, line=dict(width=1.5)),
            customdata=sub["individual_rvr"],
            hovertemplate="<b>%{text}</b><br>volatilidad %{x:.2%}"
                          "<br>retorno %{y:.2%}<br>rvr %{customdata:.3f}"
                          "<extra></extra>"))

    elegido = comp[comp["selected_max_rvr"]]
    fig.add_trace(go.Scatter(
        x=elegido["volatility_annual"], y=elegido["mean_log_annual"],
        mode="markers", name="mayor rvr", hoverinfo="skip",
        marker=dict(size=26, symbol="star-open", line=dict(width=2))))

    fig.update_layout(
        template=TEMA, height=560,
        title="Mapa media-volatilidad - activos no dominados",
        xaxis=dict(title="volatilidad anual", tickformat=".1%"),
        yaxis=dict(title="retorno anual", tickformat=".1%"),
        legend=dict(orientation="h", y=-0.15))
    return fig

def projection_chart(P_t, trayectoria, ticker, p_L, p_U):
    """Banda de cuantiles, mediana y media a lo largo del horizonte.

    P_t --> ultimo precio, el punto de partida
    trayectoria --> DataFrame como el que devuelve forecasting.path,
                    con columnas h, median, mean, q_L y q_U
    ticker --> nombre del activo, para el titulo
    p_L, p_U --> probabilidades de cola, solo para rotular la banda
    """
    t = trayectoria
    banda = f"{p_U - p_L:.0%} central"
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=t["h"], y=t["q_U"], mode="lines",
                             line=dict(width=0), hoverinfo="skip",
                             showlegend=False))
    fig.add_trace(go.Scatter(x=t["h"], y=t["q_L"], mode="lines", name=banda,
                             line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(42,120,214,0.15)",
                             hoverinfo="skip"))

    fig.add_trace(go.Scatter(x=t["h"], y=t["median"], mode="lines",
                             name="mediana", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=t["h"], y=t["mean"], mode="lines", name="media",
                             line=dict(width=1.5, dash="dot")))

    fig.add_hline(y=P_t, line=dict(width=1, dash="dash", color="gray"),
                  annotation_text=f"P_t = {P_t:.2f}",
                  annotation_position="bottom right")

    fig.update_layout(
        template=TEMA, height=480, hovermode="x unified",
        title=f"{ticker} - proyeccion a {int(t['h'].max())} periodos",
        xaxis=dict(title="horizonte h"), yaxis=dict(title="precio (USD)"),
        legend=dict(orientation="h", y=-0.18))
    return fig

def returns_hist(g, ticker, bins=40):
    """Distribucion de los log-rendimientos contra la normal ajustada.

    g --> Serie de log-rendimientos
    ticker --> nombre del activo, para el titulo
    bins --> numero aproximado de barras

    Devuelve una figura de Plotly; no la muestra.

    La curva es la normal con la media y la desviacion de la propia
    muestra: hace visible el supuesto que contrasta Jarque-Bera. Si las
    barras se despegan de la curva en las colas o en el centro, ahi esta
    la razon por la que ese contraste rechaza.
    """
    media, sd = g.mean(), g.std(ddof=1)
    x = np.linspace(g.min(), g.max(), 200)
    normal = np.exp(-((x - media) ** 2) / (2 * sd ** 2)) / (sd * np.sqrt(2 * np.pi))

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=g, nbinsx=bins, histnorm="probability density",
                               name="observados", opacity=0.75,
                               hovertemplate="rendimiento %{x:.2%}"
                                             "<br>densidad %{y:.1f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=normal, mode="lines", name="gaussiana",
                             line=dict(width=2), hoverinfo="skip"))

    fig.add_vline(x=media, line=dict(width=1, dash="dash", color="gray"),
                  annotation_text=f"media {media:.3%}", annotation_position="top")

    fig.update_layout(
        template=TEMA, height=440, bargap=0.02,
        title=f"{ticker} - distribucion de los log-rendimientos",
        xaxis=dict(title="log-rendimiento", tickformat=".1%"),
        yaxis=dict(title="densidad"),
        legend=dict(orientation="h", y=-0.2))
    return fig
def sml_chart(tabla):
    """Linea del mercado de valores ex post, una por indice de mercado.

    tabla --> DataFrame de capm.capm_table, con beta, excess_annual,
              market_excess_annual, alpha_annual y market

    En el eje vertical va el exceso de rendimiento realizado. La recta
    pasa por el origen (beta 0, exceso 0) y por el mercado (beta 1, su
    exceso realizado), asi que la distancia vertical de cada activo a su
    recta es exactamente su alfa de Jensen.
    """
    fig = go.Figure()
    b_max = max(1.2, float(tabla["beta"].max()) * 1.1)
    b_min = min(0.0, float(tabla["beta"].min()) * 1.1)

    for nombre, sub in tabla.groupby("market"):
        pendiente = float(sub["market_excess_annual"].iloc[0])
        xs = np.array([b_min, b_max])
        fig.add_trace(go.Scatter(x=xs, y=pendiente * xs, mode="lines",
                                 name=f"LMV {nombre}", hoverinfo="skip",
                                 line=dict(width=1.5, dash="dash")))
        fig.add_trace(go.Scatter(
            x=[1.0], y=[pendiente], mode="markers", name=f"mercado {nombre}",
            marker=dict(size=14, symbol="diamond-open", line=dict(width=2)),
            hovertemplate=f"<b>{nombre}</b><br>exceso %{{y:.2%}}<extra></extra>"))

        encima = sub["alpha_annual"] >= 0
        for parte, rotulo, simbolo in ((sub[encima], "encima (alfa > 0)", "circle"),
                                       (sub[~encima], "debajo (alfa < 0)", "circle-open")):
            if parte.empty:
                continue
            fig.add_trace(go.Scatter(
                x=parte["beta"], y=parte["excess_annual"], mode="markers+text",
                text=parte.index, textposition="top center",
                textfont=dict(size=10), name=f"{nombre} - {rotulo}",
                marker=dict(size=11, symbol=simbolo, line=dict(width=1.5)),
                customdata=parte["alpha_annual"],
                hovertemplate="<b>%{text}</b><br>beta %{x:.2f}"
                              "<br>exceso %{y:.2%}<br>alfa %{customdata:.2%}"
                              "<extra></extra>"))

    fig.add_hline(y=0, line=dict(width=1, color="gray"))
    fig.update_layout(
        template=TEMA, height=560,
        title="Linea del mercado de valores (ex post)",
        xaxis=dict(title="beta"),
        yaxis=dict(title="exceso de rendimiento anual", tickformat=".1%"),
        legend=dict(orientation="h", y=-0.15))
    return fig


def regression_chart(ex, alpha, beta, ticker, mercado):
    """Dispersion de excesos del activo contra el mercado y recta ajustada.

    ex --> DataFrame de capm.excess_returns
    alpha, beta --> coeficientes por periodo de la regresion
    """
    x = np.linspace(ex["market"].min(), ex["market"].max(), 50)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ex["market"], y=ex["asset"], mode="markers", name="periodos",
        marker=dict(size=6, opacity=0.6),
        hovertemplate="mercado %{x:.2%}<br>activo %{y:.2%}<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=alpha + beta * x, mode="lines",
                             name=f"ajuste: beta = {beta:.2f}",
                             line=dict(width=2), hoverinfo="skip"))
    fig.update_layout(
        template=TEMA, height=460,
        title=f"{ticker} contra {mercado} - modelo de mercado en excesos",
        xaxis=dict(title="exceso del mercado", tickformat=".1%"),
        yaxis=dict(title=f"exceso de {ticker}", tickformat=".1%"),
        legend=dict(orientation="h", y=-0.2))
    return fig

def frontier_chart(frontera, activos, portafolios, elegido, rf=None):
    """Frontera eficiente, activos individuales y portafolios de referencia.

    frontera --> DataFrame de portfolio.efficient_frontier
    activos --> DataFrame por ticker con columnas return y volatility
    portafolios --> DataFrame portafolio x (return, volatility, sharpe)
    elegido --> nombre del portafolio que se resalta
    rf --> tasa libre de riesgo; si se pasa y el tangente existe, se
           dibuja la linea del mercado de capitales hasta el
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frontera["volatility"], y=frontera["return"], mode="lines",
        name="frontera eficiente", line=dict(width=2.5),
        hovertemplate="volatilidad %{x:.2%}<br>retorno %{y:.2%}"
                      "<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=activos["volatility"], y=activos["return"], mode="markers+text",
        text=activos.index, textposition="top center",
        textfont=dict(size=10), name="activos",
        marker=dict(size=9, symbol="circle-open", line=dict(width=1.5)),
        hovertemplate="<b>%{text}</b><br>volatilidad %{x:.2%}"
                      "<br>retorno %{y:.2%}<extra></extra>"))

    tangente = [p for p in portafolios.index if p.startswith("tangente")]
    if rf is not None and tangente:
        t = portafolios.loc[tangente[0]]
        x_fin = max(float(frontera["volatility"].max()),
                    float(activos["volatility"].max()))
        fig.add_trace(go.Scatter(
            x=[0, x_fin], y=[rf, rf + t["sharpe"] * x_fin], mode="lines",
            name="linea del mercado de capitales", hoverinfo="skip",
            line=dict(width=1.5, dash="dash")))

    for nombre, fila in portafolios.iterrows():
        resaltado = nombre == elegido
        fig.add_trace(go.Scatter(
            x=[fila["volatility"]], y=[fila["return"]], mode="markers",
            name=nombre,
            marker=dict(size=20 if resaltado else 12,
                        symbol="star" if resaltado else "diamond",
                        line=dict(width=1.5)),
            hovertemplate=f"<b>{nombre}</b><br>volatilidad %{{x:.2%}}"
                          "<br>retorno %{y:.2%}<extra></extra>"))

    fig.update_layout(
        template=TEMA, height=560,
        title="Frontera eficiente sin ventas en corto",
        xaxis=dict(title="volatilidad anual", tickformat=".1%",
                   rangemode="tozero"),
        yaxis=dict(title="rendimiento esperado anual", tickformat=".1%"),
        legend=dict(orientation="h", y=-0.15))
    return fig


def weights_chart(pesos, nombre):
    """Barras horizontales con los pesos del portafolio, de mayor a menor.

    pesos --> Serie de pesos indexada por ticker
    nombre --> nombre del portafolio, para el titulo
    """
    w = pesos.sort_values()
    fig = go.Figure(go.Bar(
        x=w, y=w.index, orientation="h", text=[f"{x:.1%}" for x in w],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>peso %{x:.2%}<extra></extra>"))
    fig.update_layout(
        template=TEMA, height=max(260, 40 * len(w) + 120),
        title=f"Pesos - {nombre}",
        xaxis=dict(title="peso", tickformat=".0%",
                   range=[0, max(1.0, float(w.max()) * 1.15)]),
        showlegend=False)
    return fig


def bl_returns_chart(tabla):
    """Rendimientos esperados por activo: historico, equilibrio y BL.

    tabla --> DataFrame por ticker con columnas historico, equilibrio y
              black_litterman (rendimientos totales anuales)
    """
    series = (("historico", "historico"), ("equilibrio", "equilibrio (CAPM)"),
              ("black_litterman", "Black-Litterman"))
    fig = go.Figure()
    for columna, nombre in series:
        fig.add_trace(go.Bar(
            x=tabla.index, y=tabla[columna], name=nombre,
            hovertemplate="<b>%{x}</b><br>" + nombre +
                          " %{y:.2%}<extra></extra>"))
    fig.update_layout(
        template=TEMA, height=440, barmode="group",
        title="Rendimientos esperados: historia, equilibrio y visiones",
        yaxis=dict(title="rendimiento esperado anual", tickformat=".1%"),
        legend=dict(orientation="h", y=-0.15))
    return fig
