# %% [markdown]
# # Sesión 5 · Actividad corta 2 — Galería de errores de un modelo de segmentación
#
# **Duración:** 10 min · **Cómputo:** CPU · **Datos:** CT del HCUCH (anonimizadas) segmentadas con **nnU-Net**
#
# Un modelo **nnU-Net** segmenta **lesiones hepáticas** (casos 1–5) y **nódulos pulmonares** (casos 6–10).
# Una métrica global como el Dice resume todo en un número; mirar los errores **caso a caso** muestra
# **qué tipo** de error comete el modelo y **si importa** clínicamente.
#
# En cada caso verás:
# - un **corte axial** con la referencia del experto (contorno <span style="color:#e53935">**rojo**</span>)
#   y la predicción del modelo (relleno <span style="color:#fbc02d">**amarillo**</span>);
# - una **vista 3D** de la referencia y de la predicción.
#
# | Paso | Qué haces | Tiempo |
# |---|---|---|
# | 1 | Recorres los 10 casos y clasificas el error principal | 6 min |
# | 2 | Miras el resumen de tus respuestas | 1 min |
# | 3 | Discutes qué errores captura el Dice y cuáles importan | 3 min |

# %%
# INCLUDE: _setup.py

# %%
#@title 📦 Cargar la galería
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as w
from IPython.display import display
from PIL import Image

RAIZ = obtener_datos("galeria")
casos = pd.read_csv(RAIZ / "casos.csv")
ERRORES = ["— sin revisar —", "Sin error relevante", "Lesión omitida (falso negativo)", "Falso positivo",
           "Borde impreciso (sub/sobresegmentación)", "Lesiones fusionadas o divididas", "¿Error de la referencia?"]
IMPACTO = ["— sin revisar —", "Ninguno", "Cambiaría la medición RECIST", "Cambiaría el diagnóstico o la etapa"]
respuestas = {c: {"error": ERRORES[0], "impacto": IMPACTO[0]} for c in casos.caso}


def mostrar(caso):
    fila = casos.set_index("caso").loc[caso]
    fig = plt.figure(figsize=(15, 5))
    a = fig.add_axes([0.0, 0.0, 0.46, 0.92])
    corte = np.asarray(Image.open(RAIZ / f"{caso}_corte.png"))
    a.imshow(corte); a.axis("off")
    x0, y0 = 30, corte.shape[0] - 30
    a.plot([x0, x0 + 5 * fila.px_por_cm], [y0, y0], color="white", lw=3)
    a.text(x0 + 2.5 * fila.px_por_cm, y0 - 12, "5 cm", color="white", ha="center", fontsize=11)
    a.set_title(f"{caso.replace('_', ' ')} · {fila.organo} · rojo: referencia, amarillo: modelo", fontsize=11)
    for i, (archivo, titulo) in enumerate((("ref3d", "Referencia (3D)"), ("pred3d", "nnU-Net (3D)"))):
        b = fig.add_axes([0.48 + 0.26 * i, 0.0, 0.25, 0.92])
        b.imshow(Image.open(RAIZ / f"{caso}_{archivo}.jpg")); b.axis("off"); b.set_title(titulo, fontsize=11)
    plt.show()


salida = w.Output()
selector = w.Dropdown(options=list(casos.caso), description="Caso")
error = w.Dropdown(options=ERRORES, description="Error principal", style={"description_width": "110px"},
                   layout=w.Layout(width="440px"))
impacto = w.Dropdown(options=IMPACTO, description="Impacto clínico", style={"description_width": "110px"},
                     layout=w.Layout(width="440px"))


def al_cambiar_caso(_=None):
    c = selector.value
    error.value, impacto.value = respuestas[c]["error"], respuestas[c]["impacto"]
    with salida:
        salida.clear_output(wait=True)
        mostrar(c)


def guardar(_=None):
    respuestas[selector.value] = {"error": error.value, "impacto": impacto.value}


selector.observe(al_cambiar_caso, names="value")
error.observe(guardar, names="value")
impacto.observe(guardar, names="value")

# %% [markdown]
# ## Paso 1 · Recorre los casos
# Elige un caso, mira el corte y la vista 3D, y registra el **error principal** y su **impacto clínico**.
# Tus respuestas se guardan al cambiar los menús.

# %%
#@title 🖼️ Galería
display(w.VBox([w.HBox([selector]), w.HBox([error, impacto]), salida]))
al_cambiar_caso()

# %% [markdown]
# ## Paso 2 · Resumen de tus respuestas

# %%
#@title 📋 Resumen
tabla = pd.DataFrame(respuestas).T.rename_axis("caso").reset_index()
tabla.insert(1, "órgano", casos.organo.values)
display(tabla.style.hide(axis="index"))
revisados = tabla[tabla.error != ERRORES[0]]
if len(revisados):
    revisados.error.value_counts().sort_values().plot.barh(figsize=(7, 3), color="#4c72b0", title="Errores principales")
    plt.tight_layout(); plt.show()

# %% [markdown]
# ## Paso 3 · ✍️ Discusión
#
# 1. ¿Qué tipos de error **casi no cambian el Dice** pero sí importarían en la clínica?
#    (Piensa en una lesión pequeña omitida dentro de un hígado con lesiones grandes.)
# 2. En RECIST se miden **hasta 5 lesiones diana** por su **diámetro mayor**. ¿Qué errores cambiarían esa medición?
# 3. ¿Encontraste casos donde quizás **la referencia** está incompleta? ¿Qué harías con ellos al evaluar el modelo?
#
# _Respuesta:_
#
# ### Para llevarse
# - Una métrica global esconde **tipos** de error distintos: hay que mirar los casos.
# - El impacto clínico depende de la **tarea** (medir, detectar, etapificar), no solo del solapamiento.
# - La referencia también se equivoca: revisar los desacuerdos modelo–experto es parte de la validación.
