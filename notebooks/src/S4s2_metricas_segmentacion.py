# %% [markdown]
# # Sesión 4 · Actividad corta 2 — Dice, IoU y Hausdorff: ¿cuánto concuerdan dos expertos?
#
# **Duración:** 10 min · **Cómputo:** CPU · **Datos:** CT del HCUCH (anonimizadas), segmentadas por **dos radiólogos** de forma independiente
#
# Antes de evaluar un modelo conviene saber cuánto concuerdan **los expertos entre sí**: ese es el techo
# realista de cualquier algoritmo.
#
# | Paso | Qué haces | Tiempo |
# |---|---|---|
# | 1 | Comparas las segmentaciones de dos expertos en cuatro estructuras | 4 min |
# | 2 | Deformas una máscara y ves qué métrica reacciona | 5 min |
# | 3 | Eliges qué métricas reportarías | 1 min |
#
# **Métricas**
# - **Dice** = 2·|A∩B| / (|A|+|B|) e **IoU (Jaccard)** = |A∩B| / |A∪B|: superposición, de 0 a 1.
# - **Distancia de Hausdorff (HD)**: la **peor** distancia entre los dos bordes, en mm.
# - **HD95**: percentil 95 de esas distancias (ignora el 5 % peor).
# - **MSD** (*mean surface distance*): distancia **promedio** entre los bordes, en mm.

# %%
# INCLUDE: _setup.py

# %%
#@title 📦 Cargar las CT y definir las métricas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as w
from scipy import ndimage

RAIZ = obtener_datos("ct")
NOMBRES = {"masa_pulmonar": "Masa pulmonar", "lesiones_hepaticas_grandes": "Lesiones hepáticas grandes",
           "lesiones_hepaticas_pequenas": "Lesiones hepáticas pequeñas", "adenopatias": "Adenopatías"}
casos = {k: dict(np.load(RAIZ / f"{k}.npz")) for k in NOMBRES}


def dice(a, b):
    s = a.sum() + b.sum()
    return 2 * (a & b).sum() / s if s else np.nan


def iou(a, b):
    u = (a | b).sum()
    return (a & b).sum() / u if u else np.nan


def distancias_borde(a, b, espaciado):
    """Distancias (mm) desde cada punto del borde de a al borde de b, y viceversa."""
    if not a.any() or not b.any():
        return np.array([np.inf])
    borde = lambda m: m & ~ndimage.binary_erosion(m)
    ba, bb = borde(a), borde(b)
    da = ndimage.distance_transform_edt(~bb, sampling=espaciado)[ba]
    db = ndimage.distance_transform_edt(~ba, sampling=espaciado)[bb]
    return np.concatenate([da, db])


def todas(a, b, espaciado):
    d = distancias_borde(a, b, espaciado)
    return {"Dice": dice(a, b), "IoU": iou(a, b), "HD (mm)": d.max(),
            "HD95 (mm)": np.percentile(d, 95), "MSD (mm)": d.mean()}


def ventana(hu, nivel=40, ancho=400):
    return np.clip((hu - (nivel - ancho / 2)) / ancho, 0, 1)


FORMATO = {"Dice": "{:.3f}", "IoU": "{:.3f}", "HD (mm)": "{:.1f}", "HD95 (mm)": "{:.1f}", "MSD (mm)": "{:.2f}"}

# %% [markdown]
# ## Paso 1 · Dos expertos, cuatro estructuras
#
# <span style="color:#d62728">**Rojo**</span>: Experto 1 · <span style="color:#17becf">**Celeste**</span>: Experto 2.
# Recorre los cortes de cada estructura.

# %%
#@title 👩‍⚕️👨‍⚕️ Comparar expertos


def comparar(estructura, corte):
    c = casos[estructura]
    i = min(corte, len(c["z"]) - 1)
    e1, e2 = c["exp1"][i].astype(bool), c["exp2"][i].astype(bool)
    esp = tuple(c["spacing"][:2])
    fig, ejes = plt.subplots(1, 2, figsize=(11, 5))
    for eje in ejes:
        eje.imshow(ventana(c["ct"][i]), cmap="gray", aspect=esp[0] / esp[1])
        eje.axis("off")
    ejes[0].contour(e1, [0.5], colors="#d62728", linewidths=1.2)
    ejes[0].contour(e2, [0.5], colors="#17becf", linewidths=1.2)
    ejes[0].set_title("Contornos")
    desacuerdo = np.zeros(e1.shape + (4,))
    desacuerdo[e1 & ~e2] = (0.84, 0.15, 0.16, 0.8)
    desacuerdo[e2 & ~e1] = (0.09, 0.75, 0.81, 0.8)
    desacuerdo[e1 & e2] = (1, 1, 1, 0.35)
    ejes[1].imshow(desacuerdo, aspect=esp[0] / esp[1])
    ejes[1].set_title("Blanco: ambos · color: solo un experto")
    plt.suptitle(f"{NOMBRES[estructura]} · corte {i + 1}/{len(c['z'])}")
    plt.tight_layout(); plt.show()
    m = todas(e1, e2, esp)
    print("Este corte:  " + " · ".join(f"{k} = {FORMATO[k].format(v)}" for k, v in m.items()))
    print(f"Volumen 3D:  Dice = {float(c['vol_dice']):.3f} · Experto 1 = {float(c['vol_ml_exp1']):.0f} ml · "
          f"Experto 2 = {float(c['vol_ml_exp2']):.0f} ml")


w.interact(comparar,
           estructura=w.Dropdown(options=[(v, k) for k, v in NOMBRES.items()], description="Estructura"),
           corte=w.IntSlider(value=0, min=0, max=7, description="Corte"));

# %%
#@title 📋 Resumen en volumen (3D)
resumen = pd.DataFrame({NOMBRES[k]: {"Dice 3D": float(c["vol_dice"]), "IoU 3D": float(c["vol_iou"]),
                                      "Volumen experto 1 (ml)": float(c["vol_ml_exp1"]),
                                      "Volumen experto 2 (ml)": float(c["vol_ml_exp2"])}
                        for k, c in casos.items()}).T
display(resumen.style.format({"Dice 3D": "{:.3f}", "IoU 3D": "{:.3f}",
                              "Volumen experto 1 (ml)": "{:.0f}", "Volumen experto 2 (ml)": "{:.0f}"}))

# %% [markdown]
# ### ✍️ Pregunta 1
# Los mismos dos expertos obtienen Dice desde 0,93 hasta 0,70. ¿Qué explica la diferencia?
# Pista: compara el **tamaño** de las estructuras y mira **qué** incluyó cada experto en las adenopatías.
#
# _Respuesta:_

# %% [markdown]
# ## Paso 2 · ¿Qué métrica reacciona a qué error?
#
# Tomamos al **Experto 1 como referencia** y fabricamos una "predicción" deformándola:
# - **Dilatar / erosionar**: el borde se mueve hacia afuera (+) o hacia adentro (−), en píxeles (1 px ≈ 2 mm).
# - **Desplazar**: toda la máscara se corre.
# - **Falso positivo lejano**: se agrega una manchita fuera de la estructura.
#
# 🎯 Prueba: (a) dilata 1 px una estructura grande y una pequeña; (b) agrega solo el falso positivo.

# %%
#@title 🧪 Laboratorio de métricas


def deformar(ref, borde_px, dx, dy, falso_positivo):
    m = ref.copy()
    if borde_px > 0:
        m = ndimage.binary_dilation(m, iterations=borde_px)
    elif borde_px < 0:
        m = ndimage.binary_erosion(m, iterations=-borde_px)
    m = ndimage.shift(m.astype(float), (dy, dx), order=0) > 0.5
    if falso_positivo:
        filas, cols = np.ogrid[:m.shape[0], :m.shape[1]]
        dist = ndimage.distance_transform_edt(~ref)
        f, c = np.unravel_index(np.argmax(dist), m.shape)   # el punto más lejano de la estructura
        m |= (filas - f) ** 2 + (cols - c) ** 2 <= 9
    return m


def laboratorio(estructura, corte, borde_px, dx, dy, falso_positivo):
    c = casos[estructura]
    i = min(corte, len(c["z"]) - 1)
    ref = c["exp1"][i].astype(bool)
    pred = deformar(ref, borde_px, dx, dy, falso_positivo)
    esp = tuple(c["spacing"][:2])
    m = todas(pred, ref, esp)

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [1.2, 1]})
    a.imshow(ventana(c["ct"][i]), cmap="gray", aspect=esp[0] / esp[1])
    a.contour(ref, [0.5], colors="#d62728", linewidths=1.2)
    a.contour(pred, [0.5], colors="#ffbf00", linewidths=1.2)
    a.set_title("Rojo: referencia · amarillo: predicción"); a.axis("off")
    d = np.linspace(0.01, 1, 100)
    b.plot(d, d / (2 - d), color="#555", lw=1, label="IoU = Dice / (2 − Dice)")
    b.scatter([m["Dice"]], [m["IoU"]], color="#ffbf00", edgecolor="k", s=80, zorder=3, label="esta predicción")
    b.set_xlabel("Dice"); b.set_ylabel("IoU"); b.set_xlim(0, 1); b.set_ylim(0, 1); b.grid(alpha=0.3)
    b.legend(loc="upper left")
    plt.tight_layout(); plt.show()
    display(pd.DataFrame([m]).style.format(FORMATO).hide(axis="index"))


w.interact(laboratorio,
           estructura=w.Dropdown(options=[(v, k) for k, v in NOMBRES.items()], description="Estructura"),
           corte=w.IntSlider(value=3, min=0, max=7, description="Corte"),
           borde_px=w.IntSlider(value=0, min=-4, max=4, description="Borde (px)"),
           dx=w.IntSlider(value=0, min=-10, max=10, description="Mover x (px)"),
           dy=w.IntSlider(value=0, min=-10, max=10, description="Mover y (px)"),
           falso_positivo=w.Checkbox(False, description="Falso positivo lejano"));

# %% [markdown]
# ### ✍️ Pregunta 2
# 1. Al dilatar 1 px, ¿cuánto baja el Dice en la masa pulmonar y cuánto en las lesiones pequeñas? ¿Por qué?
# 2. Con el falso positivo lejano, ¿qué métrica casi no cambia y cuál se dispara?
# 3. Dice e IoU, ¿entregan información distinta o son la misma información en otra escala?
#
# _Respuesta:_
#
# ## Paso 3 · ✍️ Decisión
# Vas a evaluar un modelo que segmenta **adenopatías** para planificar radioterapia. ¿Qué **dos** métricas
# reportarías y qué valor de referencia usarías para decir que el modelo es "suficientemente bueno"?
#
# _Respuesta:_
#
# ### Para llevarse
# - Dice e IoU miden **superposición** y castigan más los errores en estructuras **pequeñas**.
# - Hausdorff mide el **peor** error de borde: un solo falso positivo lejano lo dispara. HD95 y MSD son más robustos.
# - La concordancia entre expertos es el **techo** razonable para un modelo, y la **definición** de la estructura importa tanto como el borde.
