"""Vista Metodologia: aviso, supuestos, modelos y limitaciones."""

import streamlit as st

AVISO = (
    "Esta aplicación fue desarrollada exclusivamente como actividad evaluativa "
    "del curso Teoría Moderna de Portafolios de Tech Business School - "
    "Universidad EIA. Sus datos, modelos y resultados tienen fines académicos "
    "y educativos. En ningún momento constituye asesoría financiera, "
    "recomendación de inversión ni una herramienta para tomar decisiones de "
    "inversión en la vida real. Es una herramienta académica que deberá "
    "revisarse, validarse y ajustarse, y puede contener errores, omisiones, "
    "rezagos o información incompleta. El sistema no ejecuta operaciones ni "
    "garantiza resultados.")


def render(ctx):
    """Dibuja la vista con el contexto comun."""
    params, modelo, origen = ctx.params, ctx.modelo, ctx.origen
    st.subheader("Aviso")
    st.warning(AVISO)

    st.subheader("Que calcula este robot")
    st.markdown(
        "Preselecciona activos: no recomienda comprar ni vender. Estima "
        "rendimiento y riesgo bajo un modelo declarado, comprueba si la "
        "operacion es viable una vez pagadas las comisiones, y muestra **que "
        "regla incumple** cuando no lo es. Con los activos elegidos "
        "construye portafolios de Markowitz, con rendimientos historicos o "
        "de Black-Litterman, y reparte el capital entre ellos y renta fija "
        "segun el perfil de riesgo del inversionista.")

    st.subheader("Convencion de rendimientos")
    st.latex(r"g_t = \ln\left(\frac{P_t}{P_{t-1}}\right)")
    st.markdown(
        "Log-rendimientos sobre precios **ajustados** por dividendos y splits. "
        "Los estadisticos usan varianza muestral (`ddof=1`), cuantiles de "
        "interpolacion lineal y asimetria y curtosis de Fisher sin sesgo.")

    st.subheader("Modelo")
    st.markdown(
        f"Se ajusta el modelo **{modelo}** sobre toda la muestra:\n\n"
        "- **A** — caminata aleatoria sin deriva: impone `mu = 0`\n"
        "- **B** — caminata con deriva: estima `mu` de la muestra\n"
        "- **C** — ARIMA-GARCH: la media depende de los rendimientos "
        "recientes y la volatilidad cambia en el tiempo\n\n"
        "A y B comparten la misma volatilidad, constante en el tiempo. Sus "
        "momentos se acumulan linealmente y la trayectoria es "
        "**analitica**, no simulada:")
    st.latex(r"m = h\,\mu \qquad v = h\,\sigma^2 \qquad "
             r"P_{t+h} = P_t\,e^{\,m + z_p\sqrt{v}}")
    st.markdown(
        "**Modelo C.** Los log-precios no son estacionarios, los "
        "log-rendimientos si; por eso la media se modela con un "
        "ARIMA(p,1,q) sobre el log-precio, que equivale a un ARMA(p,q) con "
        "constante (la deriva) sobre el log-rendimiento. Se comparan "
        "ARIMA(1,1,1), ARIMA(0,1,1) y ARIMA(1,1,0) y se elige el de menor "
        "AIC. Sobre sus residuos se estima un "
        "GARCH(1,1) con innovaciones t-Student, que captura las rachas de "
        "volatilidad y las colas gruesas:")
    st.latex(r"g_t = c + \phi\,(g_{t-1} - c) + \varepsilon_t + "
             r"\theta\,\varepsilon_{t-1} \qquad "
             r"\varepsilon_t = \sigma_t z_t,\; z_t \sim t_\nu")
    st.latex(r"\sigma_t^2 = \omega + \alpha\,\varepsilon_{t-1}^2 + "
             r"\beta\,\sigma_{t-1}^2")
    st.markdown(
        "La media acumulada suma los rendimientos que pronostica el ARIMA. "
        "En la varianza, cada choque futuro tiene la varianza que "
        "pronostica el GARCH y se propaga a los rendimientos siguientes con "
        "los pesos `psi` del ARMA:")
    st.latex(r"v_h = \sum_{j=1}^{h} \sigma_{t+j}^2\, \Psi_{h-j}^2 \qquad "
             r"\Psi_n = \sum_{k=0}^{n} \psi_k")
    st.markdown(
        "Con `alfa + beta < 1` la volatilidad condicional vuelve a la "
        "incondicional `raiz(omega / (1 - alfa - beta))`. Los niveles, el "
        "VaR y la señal se calculan igual que en A y B a partir de `m` y "
        "`v`. En el walk-forward el modelo completo, eleccion de orden "
        "incluida, se reestima en cada origen.")

    st.subheader("Niveles, costos y señal")
    st.markdown(
        f"Con colas {params.p_L:.0%} y {params.p_U:.0%} se obtienen el "
        "stop-loss y el take-profit **terminales** en `t+H`, no barreras que "
        "se vigilen durante el camino. El precio de equilibrio incorpora las "
        "dos comisiones:")
    st.latex(r"P_{BE} = P_t\,\frac{1 + c_b}{1 - c_s}")
    st.markdown(
        "La puerta exige **siete condiciones** de geometria y viabilidad "
        "economica. Si alguna falla, no hay señal y se reporta cual. Los "
        "diagnosticos y la insuficiencia de muestra **advierten pero no "
        "bloquean**: etiquetan el pronostico como condicional y dejan la "
        "decision en manos de quien opera.")

    st.subheader("CAPM")
    st.markdown(
        "La vista CAPM mide cada activo contra el indice de su mercado con "
        "el **modelo de mercado en excesos**, sobre rendimientos **simples** "
        "(el CAPM es lineal en rendimientos y los log-rendimientos no se "
        "suman entre activos):")
    st.latex(r"R_{i,t} - r_{f,t} = \alpha_i + \beta_i\,(R_{m,t} - r_{f,t})"
             r" + \varepsilon_{i,t}")
    st.markdown(
        "- **Mercado.** Con Yahoo, el indice de la bolsa del activo segun el "
        "sufijo del ticker (S&P 500 sin sufijo, COLCAP para `.CL`, Nikkei "
        "para `.T`...). Con el fixture, un proxy equiponderado de los "
        "activos cargados. Ningun indice es el portafolio de mercado del "
        "CAPM, que contiene todos los activos riesgosos y no es observable "
        "(critica de Roll).\n"
        "- **Moneda.** Activos e indices extranjeros se convierten a dolares "
        "con el tipo de cambio del dia, para ser consistentes con la tasa "
        "del Tesoro de EE.UU.\n"
        "- **Tasa libre de riesgo.** Los excesos de la regresion restan la "
        "letra del Tesoro a 13 semanas vigente al inicio de cada periodo, "
        "convertida a tasa por periodo. El rendimiento requerido usa el bono "
        "cuyo plazo se acerca mas al horizonte de inversion.\n"
        "- **Frecuencia.** Semanal por defecto: con datos diarios, los "
        "horarios distintos de cada bolsa sesgan la beta hacia cero.\n"
        "- **Beta ajustada de Blume**, `0.67 beta + 0.33`: corrige la "
        "tendencia de las betas a regresar a 1.")
    st.latex(r"E[R_i] = r_f + \beta_i\,(E[R_m] - r_f)")
    st.markdown(
        "`E[Rm]` sale de sumar a `rf` una prima de riesgo de mercado "
        "declarada, o de la media historica del indice, que es mucho mas "
        "ruidosa. La varianza del activo se separa en una parte "
        "**sistematica**, `beta^2 var(Rm)`, que el mercado remunera, y una "
        "**idiosincratica**, que se elimina diversificando.")

    st.subheader("Portafolios de Markowitz")
    st.markdown(
        "La vista Portafolio combina los activos elegidos con rendimientos "
        "**simples**, que si son lineales entre activos: el rendimiento del "
        "portafolio es el promedio ponderado de los rendimientos de sus "
        "activos. Media y covarianza muestral se anualizan multiplicando por "
        "los periodos del año:")
    st.latex(r"\mu = m\,\bar{R} \qquad \Sigma = m\,\widehat{\mathrm{Cov}}(R)"
             r"\qquad E[R_p] = w'\mu \qquad \sigma_p = \sqrt{w'\Sigma w}")
    st.markdown(
        "Todos los portafolios invierten el 100 % en activos riesgosos, "
        "**sin ventas en corto** y con un tope opcional por activo:")
    st.latex(r"\sum_i w_i = 1 \qquad 0 \le w_i \le w_{max}")
    st.markdown(
        "- **Minima varianza global:** minimiza `w' Sigma w`. Solo depende "
        "de las covarianzas, no de los rendimientos esperados.\n"
        "- **Tangente:** maximiza la razon de Sharpe, "
        "`(w' mu - rf) / sigma_p`. Es el punto donde la recta que sale de "
        "`rf` toca la frontera. No existe si ningun activo rinde mas que "
        "`rf`.\n"
        "- **Maxima rentabilidad:** maximiza `w' mu`. Sin tope es el activo "
        "de mayor media; con tope, llena los activos en orden de media.\n"
        "- **Frontera eficiente:** para cada rendimiento entre el de minima "
        "varianza y el de maxima rentabilidad, la menor volatilidad "
        "posible.\n\n"
        "Se resuelven con SLSQP y gradientes analiticos. Los precios se "
        "alinean a fechas comunes antes de calcular rendimientos, porque la "
        "covarianza necesita todos los activos en cada fecha. Con Yahoo, "
        "los activos extranjeros se pasan a dolares y `rf` es el bono del "
        "Tesoro de plazo mas cercano al horizonte. Los tres portafolios se "
        "muestran como referencia en la frontera; el portafolio final es "
        "el que corresponde al perfil de riesgo del inversionista.")

    st.subheader("Black-Litterman")
    st.markdown(
        "La media historica es un estimador muy ruidoso del rendimiento "
        "esperado, y Markowitz la amplifica: pequeños errores en `mu` "
        "producen portafolios concentrados en pocos activos. "
        "Black-Litterman parte de un punto de referencia neutral, el "
        "**equilibrio del mercado**, y lo corrige solo donde el usuario "
        "tiene una opinion. Trabaja con **excesos** sobre `rf`; al "
        "optimizador se le pasan rendimientos totales (exceso + `rf`).")
    st.markdown(
        "**Equilibrio.** Con los pesos de mercado `w_mkt` (capitalizacion "
        "de Yahoo en dolares, o pesos iguales) se buscan los rendimientos "
        "que harian optimo al portafolio de mercado (optimizacion inversa). "
        "Es el CAPM con beta medida contra ese portafolio:")
    st.latex(r"\delta = \frac{E[R_m] - r_f}{w_{mkt}'\,\Sigma\,w_{mkt}}"
             r"\qquad \Pi = \delta\,\Sigma\,w_{mkt}"
             r" = \beta\,(E[R_m] - r_f)")
    st.markdown(
        "**Visiones.** Cada fila de la tabla es una vision: absoluta (el "
        "activo rinde `q`) o relativa (el activo rinde `q` mas que otro). "
        "Se escriben como `P mu = Q`, con incertidumbre `Omega` diagonal. "
        "Omega sale de la confianza `c` (Idzorek simplificado):")
    st.latex(r"\Omega_k = \frac{1 - c_k}{c_k}\;P_k\,\tau\Sigma\,P_k'")
    st.markdown(
        "Con una sola vision, su rendimiento recorre exactamente la "
        "fraccion `c` del camino entre el equilibrio y `q`; con confianza "
        "del 100 % se adopta la vision tal cual. **Posterior:**")
    st.latex(r"\mu_{BL} = \Pi + \tau\Sigma P'\,(P\tau\Sigma P' + \Omega)^{-1}"
             r"\,(Q - P\,\Pi)")
    st.latex(r"\Sigma_{BL} = \Sigma + \tau\Sigma - \tau\Sigma P'\,"
             r"(P\tau\Sigma P' + \Omega)^{-1}\,P\,\tau\Sigma")
    st.markdown(
        "Sin visiones, `mu_BL = Pi` y el portafolio tangente es el de "
        "mercado. Como `Omega` es proporcional a `tau`, tau se cancela en "
        "`mu_BL`: los rendimientos solo dependen de la confianza, y tau "
        "solo escala la incertidumbre que se suma a `Sigma`.")

    st.subheader("Perfil de riesgo")
    st.markdown(
        "El inversionista maximiza una utilidad media-varianza, donde `A` "
        "es su aversion al riesgo:")
    st.latex(r"U = E[R] - \tfrac{1}{2}\,A\,\sigma^2")
    st.markdown(
        "**Separacion de Tobin.** El mejor portafolio riesgoso es el "
        "tangente para todos; el perfil solo decide cuanto capital se pone "
        "en el y cuanto en renta fija, que rinde `rf`. La fraccion optima "
        "en el tangente es:")
    st.latex(r"y^* = \frac{E[R_t] - r_f}{A\,\sigma_t^2}")
    st.markdown(
        "- **Si `y* <= 1`:** `y*` en el tangente y `1 - y*` en renta fija. "
        "El portafolio queda sobre la linea del mercado de capitales.\n"
        "- **Si `y* > 1`:** el inversionista querria endeudarse a `rf` para "
        "invertir mas que su capital. Sin apalancamiento, invierte el "
        "100 % en activos riesgosos y elige el de mayor utilidad: un punto "
        "de la frontera entre el tangente y la maxima rentabilidad. Con "
        "`y* = 1` los dos casos coinciden en el tangente.\n"
        "- **Si ningun activo supera `rf`:** no hay tangente y todo va a "
        "renta fija.")
    st.latex(r"\max_w\; w'\mu - \tfrac{A}{2}\,w'\Sigma w \quad "
             r"\text{s.a.}\; \textstyle\sum_i w_i = 1,\; "
             r"0 \le w_i \le w_{max}")
    st.markdown(
        "**Cuestionario.** Es el punto de entrada: hasta guardarlo en la "
        "vista Perfil de riesgo, solo se pueden abrir Metodologia y "
        "Glosario. Cinco preguntas: horizonte de inversion, capacidad de "
        "asumir perdidas, reaccion ante una caida del 20 % en un mes, "
        "preferencia entre alternativas de riesgo y experiencia. "
        "Cada respuesta vale de 1 a 3 puntos, de menos a mas tolerancia al "
        "riesgo. El puntaje, entre 5 y 15, se lleva linealmente a `A`, "
        "entre 10 (minimo puntaje, muy conservador) y 0.1 (maximo puntaje, "
        "agresivo, casi neutral al riesgo). El extremo bajo va por debajo "
        "del rango tipico de 2 a 4 de los textos para que un perfil "
        "agresivo llegue al tangente o lo supere aun cuando la razon de "
        "Sharpe de la muestra es baja. Por tercios del rango, el perfil es "
        "conservador, "
        "moderado o agresivo. La `A` se puede ajustar a mano. El perfil "
        "guardado se usa en la vista Portafolio y dura mientras la pestaña "
        "del navegador siga abierta.")

    st.subheader("Limitaciones conocidas")
    st.markdown(
        "- **La homocedasticidad no se cumple.** ARCH-LM rechaza en "
        "practicamente cualquier serie financiera: la volatilidad viene por "
        "rachas. Los modelos A y B usan una volatilidad unica, asi que sus "
        "intervalos son correctos en promedio pero mal calibrados en cada "
        "momento. El modelo C lo corrige con un GARCH.\n"
        "- **El modelo C aproxima la distribucion acumulada.** Usa la media "
        "y la varianza exactas del rendimiento acumulado, pero toma sus "
        "cuantiles de una normal: las colas t-Student de cada periodo se "
        "suavizan al sumar varios, pero a horizontes cortos el VaR puede "
        "quedarse corto. Ademas, con series sinteticas como el fixture el "
        "GARCH puede quedar en la frontera (alfa + beta cerca de 1).\n"
        "- **El VaR parametrico supone normalidad.** Con colas gruesas "
        "subestima la perdida extrema. No es la perdida maxima posible ni dice "
        "cuanto se pierde mas alla del cuantil.\n"
        "- **El walk-forward usa diez origenes solapados.** Comparten casi "
        "todas sus observaciones, asi que la cobertura tiende a salir cerca de "
        "0 o de 1 en vez de aproximarse al nivel prometido.\n"
        "- **El CAPM se estima con historia.** La beta cambia en el "
        "tiempo y el alfa realizado no predice el alfa futuro; con pocos "
        "años de datos casi ningun alfa es estadisticamente distinto de "
        "cero.\n"
        "- **El mercado de cada activo sale del sufijo del ticker.** Solo "
        "se reconocen los sufijos de la tabla de mercados (vista Glosario). "
        "Un activo con otro sufijo (por ejemplo `.AX` o `.SW`) se trata "
        "como de EE.UU.: se asume que cotiza en dolares y no se convierte, "
        "y el CAPM lo mide contra el S&P 500. Su beta, su conversion a "
        "dolares en Portafolio y su capitalizacion en Black-Litterman "
        "quedan mal, sin aviso.\n"
        "- **Markowitz es sensible a los rendimientos esperados.** Con "
        "medias historicas, cambios pequeños en la muestra mueven mucho los "
        "pesos del tangente y de la maxima rentabilidad. La frontera es "
        "eficiente dentro de la muestra, no necesariamente fuera de ella.\n"
        "- **Los pesos de mercado de Black-Litterman usan la capitalizacion "
        "de hoy**, no la del periodo analizado, y solo de los activos "
        "elegidos: es un mercado reducido, no el portafolio de mercado del "
        "CAPM. La prima de riesgo y la confianza de las visiones las fija "
        "el usuario.\n"
        "- **El paso del cuestionario a `A` es una convencion.** El rango "
        "de 0.1 a 10 y la escala lineal son supuestos razonables, no una "
        "medicion: dos personas con el mismo puntaje pueden tolerar riesgos "
        "distintos. Ademas, la utilidad media-varianza ignora las colas "
        "gruesas, y la renta fija se toma sin riesgo: la tasa del bono del "
        "Tesoro, sin riesgo de precio ni de moneda.")

    st.subheader("Origen de los datos")
    st.info(origen)
