# Robot Analítico para la Preselección de Activos

> Esta aplicación fue desarrollada exclusivamente como actividad evaluativa del
> curso Teoría Moderna de Portafolios de Tech Business School - Universidad EIA.
> Sus datos, modelos y resultados tienen fines académicos y educativos. En
> ningún momento constituye asesoría financiera, recomendación de inversión ni
> una herramienta para tomar decisiones de inversión en la vida real. Es una
> herramienta académica que deberá revisarse, validarse y ajustarse, y puede
> contener errores, omisiones, rezagos o información incompleta. El sistema no
> ejecuta operaciones ni garantiza resultados.

Herramienta que estima rendimiento y riesgo de un conjunto de activos, comprueba
si una operación es viable una vez pagadas las comisiones y reporta **qué regla
incumple** cuando no lo es. Con los activos elegidos construye portafolios de
Markowitz, con rendimientos históricos o de Black-Litterman, y reparte el capital
entre ellos y renta fija según el perfil de riesgo del inversionista.

---

## Qué hace

### Análisis por activo

Sobre una serie de precios ajustados, para cada activo:

1. Calcula log-rendimientos y estadística descriptiva anualizada
2. Contrasta cuatro supuestos del modelo (normalidad, independencia, varianza
   constante y volatilidad condicional)
3. Ajusta uno de tres modelos y proyecta precios a un horizonte `H` con fórmula
   cerrada, sin simulación:
   - **A** — caminata aleatoria sin deriva
   - **B** — caminata aleatoria con deriva
   - **C** — ARIMA-GARCH: compara ARIMA(1,1,1), ARIMA(0,1,1) y ARIMA(1,1,0)
     sobre el log-precio, elige el de menor AIC y modela sus residuos con un
     GARCH(1,1) t-Student. La varianza acumulada a `H` se calcula exacta con la
     volatilidad condicional del GARCH y los pesos ψ del ARMA
4. Deriva stop-loss, take-profit, precio de equilibrio con costos y VaR
   paramétrico
5. Valida el modelo contra lo que realmente ocurrió (walk-forward); el modelo C
   se reestima completo en cada origen
6. Decide si hay señal y, si no la hay, dice cuál de las siete reglas falló

Y sobre el conjunto: descarta los activos dominados en el plano
media-volatilidad y señala el de mejor razón retorno-volatilidad.

### CAPM

Para cada activo estima el modelo de mercado en excesos,
`R_i - rf = α + β (R_m - rf) + ε`, sobre rendimientos simples semanales:

- **Mercado por país**: S&P 500 para tickers de EE. UU., COLCAP para `.CL`,
  Nikkei 225 para `.T`, etc. Con el fixture, un proxy equiponderado.
- **Todo en USD**: activos e índices extranjeros se convierten con el tipo de
  cambio de Yahoo (`COP=X`, `JPY=X`...).
- **Tasa libre de riesgo**: bonos del Tesoro de EE. UU. (`^IRX`, `^FVX`,
  `^TNX`, `^TYX`). Los excesos usan la letra a 13 semanas; el rendimiento
  requerido, el plazo más cercano al horizonte o el que elija el usuario.
- **Salidas**: β cruda y ajustada de Blume, α de Jensen con p-valor, R²,
  riesgo sistemático vs. idiosincrático, rendimiento requerido, Sharpe,
  Treynor y la Línea del Mercado de Valores.

### Portafolios

- **Markowitz sin ventas en corto**, sobre rendimientos simples: mínima
  varianza global, tangente (máximo Sharpe), máxima rentabilidad y frontera
  eficiente, con un tope opcional de peso por activo. Se resuelven con SLSQP y
  gradientes analíticos.
- **Black-Litterman**: rendimientos de equilibrio `Π = δ Σ w_mkt` con pesos de
  mercado por capitalización de Yahoo (en USD) o iguales, visiones absolutas y
  relativas con confianza tipo Idzorek, y prima de riesgo y `τ` ajustables. Sin
  visiones, el portafolio tangente es el de mercado.
- **Perfil de riesgo**: un cuestionario de cinco preguntas, obligatorio antes de
  usar el resto del bot, estima la aversión al riesgo `A` (de 10 a 0.1). Con la
  separación de Tobin, `y* = (E[R_t] - rf) / (A σ_t²)` decide qué parte del
  capital va al tangente y qué parte a renta fija; si `y* > 1`, sin
  apalancamiento, se elige el portafolio de máxima utilidad sobre la frontera.

---

## Estructura

```
src/                  motor financiero — no conoce la interfaz
  contracts.py        parámetros del análisis
  data.py             carga, validación, remuestreo, cobertura, alineación;
                      descargas de Yahoo: precios, índices, tasas, tipos de
                      cambio y capitalización
  analytics.py        descriptivos y diagnósticos
  forecasting.py      modelos A, B y C (ARIMA-GARCH), trayectoria, walk-forward
  risk_rules.py       VaR, niveles con costos, probabilidades, puerta de señal
  compare.py          coordenadas históricas y dominancia
  capm.py             beta, alfa de Jensen, rendimiento requerido, desempeño
  portfolio.py        Markowitz sin cortos, frontera eficiente y asignación
                      según el perfil de riesgo
  black_litterman.py  equilibrio, visiones P/Q/Ω y posterior
  pipeline.py         orquestador: encadena el motor sobre un activo
app.py                punto de entrada: panel lateral común, carga de datos y
                      bloqueo hasta guardar el perfil de riesgo
app/                  interfaz — solo dibuja, no calcula
  contexto.py         estado común de las vistas y descargas cacheadas
  charts.py           figuras de Plotly
  vistas/             una ventana por archivo
    __init__.py       registro y orden del menú
    comun.py          selector de activo y análisis cacheado
    metodologia.py, perfil.py, datos.py, resumen.py, precio.py,
    rendimientos.py, proyeccion.py, comparacion.py, capm.py,
    portafolio.py, glosario.py
notebooks/            cuadernos de verificación
```

La separación es estricta: **`src/` nunca importa Streamlit ni Plotly**. El
motor se puede ejecutar entero desde un notebook, sin necesidad de la interfaz.

Para agregar una ventana se crea `app/vistas/<nombre>.py` con una función
`render(ctx)` —y `controles(fuente)` si necesita parámetros propios en el
panel— y se registra en `app/vistas/__init__.py`.

---

## Instalación

Requiere **Python 3.12**.

> **Clona el repositorio en una ruta sin tildes ni símbolos.** En Windows,
> yfinance descarga a través de libcurl, que no sabe abrir rutas con caracteres
> no ASCII. Si el proyecto vive en una carpeta como `D:\7°semestre\Teoría...`,
> **todas** las descargas fallan con un error de certificados difícil de
> diagnosticar, porque el error que se reporta es «sin datos en el rango
> pedido».

```bash
git clone https://github.com/DavidOssa77/analytic-trading-bot.git
```

```bash
cd analytic-trading-bot
```

```bash
python -m venv .venv
```

```bash
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

## Uso

### La aplicación

Para abrir la app ejecutar lo siguiente en consola:
```bash
.venv\Scripts\python.exe -m streamlit run app.py
```

- **Once vistas**: metodología, perfil de riesgo, datos, resumen, precio,
  rendimientos, proyección, comparación, CAPM, portafolio y glosario.
- **El perfil de riesgo es el punto de entrada**: hasta guardarlo solo se
  pueden abrir Metodología, Perfil de riesgo y Glosario. Dura mientras la
  pestaña del navegador siga abierta.
- Los parámetros —fuente, frecuencia, horizonte, modelo, comisiones, razón
  beneficio-riesgo mínima, confianza y capital— se ajustan desde el panel
  lateral y recalculan todo al instante. Lo elegido en Portafolio se conserva
  al cambiar de vista.
- El modelo C tarda unos segundos la primera vez por activo (reestima el
  ARIMA-GARCH en cada origen del walk-forward); después sale de la caché.
- El **Glosario** lista los tickers verificados en Yahoo y los sufijos de
  mercado reconocidos.

### Los cuadernos

```bash
.venv\Scripts\python.exe -m jupyter notebook
```

- **`notebooks/exploracion.ipynb`** recorre el motor completo sobre el fixture
  del curso y reproduce los casos de prueba.
- **`notebooks/datos_reales.ipynb`** ejercita lo que el fixture no puede probar:
  descarga desde Yahoo Finance, cobertura con huecos, alineación de calendarios
  distintos y remuestreo.

---

## Fuentes de datos

**Fixture del curso** (`notebooks/Fixture_20_activos_sintetico_TBS_EIA.csv`) —
veinte activos sintéticos y deterministas. Usa nombres de tickers reales como
etiquetas, pero **los precios no son datos de mercado**: AAPL cotiza a 73.63 en
el fixture y a 344.63 en la realidad (octubre de 2026). Su virtud es ser
reproducible al decimal, lo que permite verificar el motor contra resultados
esperados. Para el modelo C conviene usar Yahoo: los rendimientos sintéticos
tienen una autocorrelación muy alta y el GARCH queda en la frontera.

**Yahoo Finance** — precios ajustados por dividendos y splits (`auto_adjust=True`).
Los tickers que fallan se aíslan: los válidos conservan su análisis y la
aplicación informa cuáles no respondieron. Los fallos no se guardan en caché,
así que un error pasajero se reintenta en la siguiente interacción.

---

## Verificación

El motor se contrasta contra los resultados esperados publicados por el curso.
Las quince columnas de los veinte activos coinciden con un error máximo de
`7.77e-16`, muy por debajo de la tolerancia exigida (`atol=1e-8`). La sección
final de `exploracion.ipynb` ejecuta esa comparación.

Las ampliaciones se verifican contra propiedades teóricas con el fixture:

- **Markowitz**: el portafolio de mínima varianza cumple las condiciones KKT.
- **Black-Litterman**: sin visiones, el tangente es el portafolio de mercado y
  `Π = β · prima`; con una visión, el rendimiento recorre exactamente la
  fracción de confianza; la forma usada coincide con la fórmula maestra.
- **Perfil de riesgo**: con `y* ≤ 1` el portafolio cae sobre la línea del
  mercado de capitales; con `y* = 1` los dos casos coinciden en el tangente.
- **Modelo C**: la media y la varianza acumuladas coinciden con una simulación
  Monte Carlo del ARMA-GARCH ajustado (error menor al 1 %), y A y B dan
  exactamente los mismos resultados que antes de agregarlo.

La interfaz se recorre con `streamlit.testing.v1.AppTest`, comprobando que las
vistas que no se tocan muestran exactamente lo mismo que antes.

---

## Limitaciones conocidas

Medidas sobre los datos del proyecto y declaradas en la vista *Metodología* de la app:

- **La homocedasticidad no se cumple.** ARCH-LM rechaza en los veinte activos:
  la volatilidad viene por rachas. Los modelos A y B usan una volatilidad única,
  así que sus intervalos son correctos en promedio pero mal calibrados en cada
  momento; el modelo C lo corrige con un GARCH.
- **El modelo C aproxima la distribución acumulada** con una normal de media y
  varianza exactas: a horizontes cortos el VaR puede quedarse corto frente a las
  colas t-Student.
- **El VaR paramétrico supone normalidad** y subestima la pérdida extrema:
  2.84 % frente a 5.25 % del cuantil histórico sobre el mismo activo.
- **El walk-forward usa diez orígenes solapados**, que comparten casi todas sus
  observaciones; la cobertura tiende a salir cerca de 0 o de 1.
- **El CAPM se estima con historia.** La β cambia en el tiempo y el α
  realizado no predice el futuro. Ningún índice es el portafolio de mercado
  teórico (crítica de Roll).
- **El mercado de cada activo sale del sufijo del ticker.** Un sufijo que no
  está en la tabla de mercados se trata como de EE. UU., en dólares y contra el
  S&P 500.
- **Markowitz es sensible a los rendimientos esperados**: con medias históricas,
  cambios pequeños en la muestra mueven mucho los pesos.
- **Black-Litterman usa la capitalización de hoy** y solo de los activos
  elegidos, no la del periodo analizado.
- **El paso del cuestionario a `A` es una convención**: el rango de 0.1 a 10 y
  la escala lineal son supuestos razonables, no una medición.

---

## Equipo

### **David Ossa, Maria Camila Gaviria, María Arango**

### Universidad EIA · 2026
