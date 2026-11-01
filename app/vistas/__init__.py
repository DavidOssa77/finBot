"""Registro de vistas: el orden de VISTAS es el orden del menu lateral.

Para agregar una ventana: crear app/vistas/<nombre>.py con render(ctx)
(y, si necesita controles propios en el panel, controles(fuente)) y
registrarla aqui.
"""

from . import (datos, resumen, precio, rendimientos, proyeccion, comparacion,
               capm, portafolio, metodologia, glosario, perfil)

VISTAS = {
    "Metodologia": metodologia,
    "Perfil de riesgo": perfil,
    "Datos": datos,
    "Resumen": resumen,
    "Precio": precio,
    "Rendimientos": rendimientos,
    "Proyeccion": proyeccion,
    "Comparacion": comparacion,
    "CAPM": capm,
    "Portafolio": portafolio,
    "Glosario": glosario,
}
INICIAL = "Metodologia"
PERFIL = "Perfil de riesgo"
# Vistas que se pueden abrir antes de guardar el perfil de riesgo
SIN_PERFIL = (INICIAL, PERFIL, "Glosario")
