"""Vista Perfil de riesgo: cuestionario que fija la aversion al riesgo A.

El perfil se guarda en st.session_state["perfil"] como un diccionario
con A, nombre y respuestas. Mientras no exista, app.py bloquea las
vistas que no estan en SIN_PERFIL; la vista Portafolio lo usa para
repartir el capital entre el tangente y renta fija.
"""

import streamlit as st

# Cada respuesta vale 1, 2 o 3 puntos, de menor a mayor tolerancia al
# riesgo. El puntaje se lleva linealmente a A: el minimo da A_MAX (muy
# conservador) y el maximo A_MIN (agresivo).
PREGUNTAS = (
    ("1. ¿En cuanto tiempo necesitara este dinero?",
     ("Menos de 2 años", "Entre 2 y 7 años", "Mas de 7 años")),
    ("2. Si perdiera todo lo invertido, ¿como afectaria su situacion "
     "financiera?",
     ("Gravemente: es buena parte de mi patrimonio o no tengo fondo de "
      "emergencia",
      "Moderadamente: me doleria, pero cubriria mis gastos sin problema",
      "Poco: es una fraccion pequeña de mi patrimonio")),
    ("3. Su portafolio cae 20 % en un mes. ¿Que hace?",
     ("Vendo para evitar mas perdidas", "Mantengo y espero la recuperacion",
      "Aprovecho para comprar mas")),
    ("4. ¿Cual de estas alternativas de inversion a un año prefiere?",
     ("Ganancia segura de 6 %",
      "Ganancia probable de 12 %, con posibilidad de perder hasta 5 %",
      "Ganancia probable de 25 %, con posibilidad de perder hasta 20 %")),
    ("5. ¿Cual describe mejor su experiencia invirtiendo?",
     ("Solo he usado productos de bajo riesgo (ahorros, CDT, fondos "
      "conservadores)",
      "He invertido en fondos o acciones, pero nunca he vivido una caida "
      "fuerte",
      "He mantenido inversiones durante caidas importantes del mercado")),
)
A_MIN, A_MAX = 0.1, 10.0


def _aversion(puntaje):
    """Lleva el puntaje del cuestionario a (A, nombre del perfil).

    puntaje --> suma de los puntos, entre len(PREGUNTAS) y 3 veces eso
    """
    n = len(PREGUNTAS)
    fraccion = (puntaje - n) / (2 * n)
    A = A_MAX - (A_MAX - A_MIN) * fraccion
    if fraccion < 1 / 3:
        return A, "conservador"
    if fraccion < 2 / 3:
        return A, "moderado"
    return A, "agresivo"


def _guardar(A, nombre, respuestas):
    """Guarda el perfil en la sesion; corre antes de redibujar."""
    st.session_state["perfil"] = {"A": A, "nombre": nombre,
                                  "respuestas": respuestas}


def render(ctx):
    """Dibuja la vista; no depende de los datos cargados."""
    st.subheader("Perfil de riesgo del inversionista")
    st.markdown(
        "Antes de usar el robot, responde estas preguntas. Con ellas se "
        "estima tu **aversion al riesgo A**: cuanto rendimiento exiges por "
        "cada unidad de varianza. En la vista Portafolio, A decide que "
        "parte de tu capital va al portafolio tangente y que parte a renta "
        "fija (separacion de Tobin), sin endeudarse.")

    guardado = st.session_state.get("perfil")
    respuestas = []
    for i, (pregunta, opciones) in enumerate(PREGUNTAS):
        previa = guardado["respuestas"][i] if guardado else 1
        eleccion = st.radio(pregunta, opciones, index=previa)
        respuestas.append(opciones.index(eleccion))
    puntaje = sum(respuestas) + len(PREGUNTAS)
    A, nombre = _aversion(puntaje)

    if st.checkbox("Ajustar A a mano",
                   help="Parte del valor que da el cuestionario."):
        A = st.number_input("Aversion al riesgo A", 0.1, 200.0, A, 0.5,
                            "%.1f")

    k1, k2, k3 = st.columns(3)
    k1.metric("Perfil", nombre)
    k2.metric("Puntaje", f"{puntaje} de {3 * len(PREGUNTAS)}")
    k3.metric("Aversion al riesgo A", f"{A:.1f}")
    st.caption(
        f"El puntaje va de {len(PREGUNTAS)} a {3 * len(PREGUNTAS)} y se "
        f"lleva linealmente a A, de {A_MAX:g} (muy conservador) a "
        f"{A_MIN:g} (agresivo).")

    # El boton y el estado del perfil van en la misma fila: el boton toma
    # su ancho natural y el aviso ocupa el resto
    fila = st.container(horizontal=True, vertical_alignment="center")
    fila.button("Guardar perfil", type="primary", on_click=_guardar,
                args=(A, nombre, respuestas), width=140)
    aviso = fila.container()
    if guardado is None:
        aviso.info("Responde y pulsa **Guardar perfil** para habilitar el "
                   "resto de las vistas.")
    else:
        aviso.success(f"Perfil guardado: **{guardado['nombre']}**, "
                      f"A = {guardado['A']:.1f}. Puedes cambiarlo cuando "
                      "quieras.")
    sin_guardar = (guardado is not None
                   and (guardado["A"], guardado["respuestas"])
                   != (A, respuestas))
    if sin_guardar:
        st.caption("Hay cambios sin guardar.")
