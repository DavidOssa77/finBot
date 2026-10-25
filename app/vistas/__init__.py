"""Registro de vistas: el orden de VISTAS es el orden del menu lateral.

Para agregar una ventana: crear app/vistas/<nombre>.py con render(ctx)
(y, si necesita controles propios en el panel, controles(fuente)) y
registrarla aqui.
"""

from . import (datos, resumen, precio, rendimientos, proyeccion, comparacion,
               capm, portafolio, metodologia)

VISTAS = {
    "Datos": datos,
    "Resumen": resumen,
    "Precio": precio,
    "Rendimientos": rendimientos,
    "Proyeccion": proyeccion,
    "Comparacion": comparacion,
    "CAPM": capm,
    "Portafolio": portafolio,
    "Metodologia": metodologia,
}
INICIAL = "Metodologia"
