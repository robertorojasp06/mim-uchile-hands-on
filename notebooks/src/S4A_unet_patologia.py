# %% [markdown]
# # Sesión 4 · Actividad A — Segmentar láminas H&E con una U-Net preentrenada
#
# **Duración:** 25 min · **Cómputo:** CPU · **Datos:** parches de láminas de colon y gástrico (proyecto DeepPath, anonimizados)
#
# Antes de buscar un tumor en una lámina completa (*whole slide image*, WSI) hay que saber **dónde hay tejido**
# y **dónde hay artefactos** (burbujas, tinta, pliegues, desenfoque, marcas de lápiz…). Una **U-Net** pequeña
# hace esa segmentación en tres clases: **fondo**, **tejido** y **artefacto**.
#
# ![Lámina completa de colon con sus artefactos](https://raw.githubusercontent.com/robertorojasp06/mim-uchile-hands-on/main/notebooks/img/S4A_lamina_colon.jpg)
#
# Así se ve una lámina completa de colon teñida con H&E. Casi todo el vidrio es **fondo**, y el **tejido** son
# varios cortes del mismo bloque. También hay **artefactos** del montaje y del escaneo: aire atrapado bajo el
# cubreobjetos, el borde del portaobjetos y una sombra en el extremo del vidrio. La parte inferior de la lámina se
# recortó porque tiene anotaciones escritas a mano (otro artefacto: **marcas de lápiz**) que podrían identificar el caso.
# Una lámina así pesa unos 22 gigapíxeles a 40x, mientras que la U-Net solo ve parches de 300 × 300 píxeles (panel c).
# Por eso la lámina se recorre por parches, y por eso importa a qué aumento se trabaja.
#
# Tenemos tres versiones del modelo, entrenadas **solo con colon** a tres aumentos (10x, 20x y 40x),
# con parches de 300 × 300 píxeles. Cada modelo se usa con parches de su mismo aumento.
# Los parches de este notebook **no** se usaron para entrenar.
#
# | Paso | Qué haces | Tiempo |
# |---|---|---|
# | 1 | Exploras los parches y sus anotaciones | 3 min |
# | 2 | Corres la U-Net en un parche y lees sus probabilidades | 5 min |
# | 3 | Decides a qué aumento trabajar | 10 min |
# | 4 | Pruebas el modelo en otro órgano (validación externa) | 6 min |

# %%
# INCLUDE: _setup.py

# %%
#@title 📦 Cargar la U-Net y los parches
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as w
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import patheffects
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from PIL import Image

RAIZ = obtener_datos("wsi")
indice = pd.read_csv(RAIZ / "index.csv")
AUMENTOS = ["10x", "20x", "40x"]
UM_POR_PIXEL = {"10x": 0.906, "20x": 0.453, "40x": 0.227}   # escáner Hamamatsu NanoZoomer de las láminas
BARRA_UM = {"10x": 100, "20x": 50, "40x": 20}                # largo de la barra de escala en cada aumento
CLASES = ["fondo", "tejido", "artefacto"]
CMAP = ListedColormap(["#eeeeee", "#c2185b", "#1976d2"])
LEYENDA = [Patch(color=CMAP(i), label=c) for i, c in enumerate(CLASES)]


class UNet300(nn.Module):
    """U-Net para parches RGB de 300 x 300 (DeepPath, SCIAN-Lab)."""

    def __init__(self):
        super().__init__()
        self.conv1, self.conv2 = nn.Conv2d(3, 16, 3, padding=1), nn.Conv2d(16, 32, 3, padding=1)
        self.conv3, self.conv4 = nn.Conv2d(32, 64, 3, padding=1), nn.Conv2d(64, 64, 3, padding=1)
        self.conv5, self.conv6 = nn.Conv2d(128, 32, 3, padding=1), nn.Conv2d(64, 16, 3, padding=1)
        self.conv7, self.conv8 = nn.Conv2d(32, 16, 3, padding=1), nn.Conv2d(16, 3, 1)

    def forward(self, x):
        sube = lambda t, **k: F.interpolate(t, mode="bilinear", align_corners=False, **k)
        x1 = F.relu(self.conv1(x))                                         # codificador
        x2 = F.relu(self.conv2(F.max_pool2d(x1, 2)))
        x3 = F.relu(self.conv3(F.max_pool2d(x2, 2)))
        x = sube(F.relu(self.conv4(F.max_pool2d(x3, 2))), size=x3.shape[-2:])
        x = sube(F.relu(self.conv5(torch.cat((x, x3), 1))), scale_factor=2)  # decodificador + conexiones
        x = sube(F.relu(self.conv6(torch.cat((x, x2), 1))), scale_factor=2)
        x = F.relu(self.conv7(torch.cat((x, x1), 1)))
        return self.conv8(x)                                               # 3 mapas de logits: fondo, tejido, artefacto


modelos = {}
for aum in AUMENTOS:
    modelos[aum] = UNet300()
    modelos[aum].load_state_dict(torch.load(RAIZ / "models" / f"unet_colon_{aum}.pt", map_location="cpu"))
    modelos[aum].eval()
print(f"3 U-Nets cargadas (una por aumento), cada una con "
      f"{sum(p.numel() for p in modelos['10x'].parameters()):,} parámetros.".replace(",", "."))

imagen = {r: np.asarray(Image.open(RAIZ / r)) for r in indice.image}
mascara = {r: np.asarray(Image.open(RAIZ / r)) for r in indice["mask"]}
_cache = {}


def probabilidades(organo, aumento):
    """Probabilidades (n, 3, 300, 300) del modelo de `aumento` sobre los parches de `organo` a ese aumento."""
    clave = (organo, aumento)
    if clave not in _cache:
        sub = indice[(indice.organo == organo) & (indice.aumento == aumento)]
        x = torch.tensor(np.stack([imagen[r] for r in sub.image]), dtype=torch.float32).permute(0, 3, 1, 2) / 255
        with torch.inference_mode():
            p = torch.cat([F.softmax(modelos[aumento](x[i:i + 16]), 1) for i in range(0, len(x), 16)])
        _cache[clave] = p.numpy().astype(np.float16)
    return _cache[clave]


def referencia(organo, aumento):
    sub = indice[(indice.organo == organo) & (indice.aumento == aumento)]
    return np.stack([mascara[r] for r in sub["mask"]])


def dice(pred, ref, clase):
    a, b = pred == clase, ref == clase
    s = a.sum() + b.sum()
    return 2 * (a & b).sum() / s if s else np.nan


def barra_escala(eje, aumento, um_por_pixel=None):
    """Dibuja una barra de escala en la esquina inferior izquierda de un parche de 300 x 300."""
    um = BARRA_UM[aumento]
    largo = um / (um_por_pixel or UM_POR_PIXEL[aumento])
    eje.plot([15, 15 + largo], [282, 282], color="black", lw=4, solid_capstyle="butt",
             path_effects=[patheffects.withStroke(linewidth=7, foreground="white")])
    eje.text(15 + largo / 2, 268, f"{um} µm", color="black", ha="center", va="bottom", fontsize=9,
             fontweight="bold", path_effects=[patheffects.withStroke(linewidth=3, foreground="white")])


print("\nNúmero de parches por órgano (filas) y aumento (columnas):")
display(indice.groupby(["organo", "aumento"]).size().unstack().rename_axis(index="órgano", columns="aumento"))
n_reg = indice.groupby("organo").region.nunique()
print(f"Colon: {n_reg['colon']} regiones × 3 parches = {3 * n_reg['colon']} parches por aumento. "
      f"Gástrico: {n_reg['gastrico']} regiones × 3 = {3 * n_reg['gastrico']} (no tiene marcas de lápiz).")
print("Cada parche mide 300 × 300 píxeles y lo que abarca depende del aumento: "
      + ", ".join(f"{a} = {300 * u:.0f} µm" for a, u in UM_POR_PIXEL.items()) + ".")

# %% [markdown]
# ## Paso 1 · Explora los parches
# Cada parche tiene una **anotación** hecha por un experto: <span style="color:#c2185b">**tejido**</span>,
# <span style="color:#1976d2">**artefacto**</span> o fondo. Fíjate en la barra de escala: un parche a 10x
# abarca 4 veces más ancho (16 veces más área) que uno a 40x.
#
# > 💡 **Recuerda:** los parches de cada aumento son **distintos**. Se cortaron en lugares distintos de las láminas;
# > no son el mismo parche de 40x reducido a 20x y 10x. Por eso "burbujas_aire #0" a 10x y a 40x muestran
# > burbujas diferentes.

# %%
#@title 🔬 Galería


def galeria(organo, aumento, region):
    sub = indice[(indice.organo == organo) & (indice.aumento == aumento) & (indice.region == region)]
    if sub.empty:
        print("No hay parches de esta región para este órgano.")
        return
    fig, ejes = plt.subplots(2, len(sub), figsize=(3.3 * len(sub), 6.4))
    for j, (_, f) in enumerate(sub.iterrows()):
        ejes[0, j].imshow(imagen[f.image]); ejes[0, j].set_title(f"{region} #{f.k}")
        barra_escala(ejes[0, j], aumento)
        ejes[1, j].imshow(mascara[f["mask"]], cmap=CMAP, vmin=0, vmax=2, interpolation="nearest")
    for e in ejes.flat:
        e.axis("off")
    ejes[1, -1].legend(handles=LEYENDA, loc="lower right", fontsize=8)
    plt.tight_layout(); plt.show()


w.interact(galeria,
           organo=w.Dropdown(options=["colon", "gastrico"], description="Órgano"),
           aumento=w.Dropdown(options=AUMENTOS, value="20x", description="Aumento"),
           region=w.Dropdown(options=sorted(indice.region.unique()), value="pliegues_tejido", description="Región"));

# %% [markdown]
# ## Paso 2 · La U-Net en un parche
# La U-Net tiene forma de **U**. El **codificador** (bajada) aplica convoluciones y reduce la imagen a la mitad
# en cada nivel: pierde detalle espacial, pero cada píxel "ve" una zona más grande y reconoce **qué** hay. El
# **decodificador** (subida) vuelve a agrandar la imagen hasta 300 × 300. Las **conexiones de salto** copian los
# mapas del codificador al decodificador para recuperar **dónde** está cada cosa, con bordes nítidos.
#
# ![Arquitectura de la U-Net de este notebook](https://raw.githubusercontent.com/robertorojasp06/mim-uchile-hands-on/main/notebooks/img/S4A_unet300.png)
#
# Esta es una U-Net **reducida**, pensada para correr rápido. La
# [U-Net original](https://arxiv.org/abs/1505.04597) (Ronneberger et al., 2015) tiene 4 niveles de bajada,
# dos convoluciones por nivel, de 64 a 1.024 canales y ~31 millones de parámetros: unas 280 veces más.
#
# La última capa de la red entrega, para cada píxel, **3 números** (*logits*), uno por clase. La función
# **softmax** los convierte en probabilidades:
#
# $$P(c) = \frac{e^{z_c}}{e^{z_\text{fondo}} + e^{z_\text{tejido}} + e^{z_\text{artefacto}}}$$
#
# Por eso, en cada píxel, cada probabilidad está entre 0 y 1 y **P(fondo) + P(tejido) + P(artefacto) = 1**.
# La predicción final es la clase más probable, y la **confianza** es la probabilidad de esa clase ganadora.
# Mueve los deslizadores *Fila* y *Columna* para leer las tres probabilidades de un píxel (marcado con **+**).

# %%
#@title 🤖 Inferencia


def inferencia(aumento, region, k, fila, columna):
    sub = indice[(indice.organo == "colon") & (indice.aumento == aumento)].reset_index(drop=True)
    i = sub[(sub.region == region) & (sub.k == k)].index[0]
    p = probabilidades("colon", aumento)[i].astype(np.float32)
    pred, ref = p.argmax(0), mascara[sub.loc[i, "mask"]]
    fig, ejes = plt.subplots(2, 4, figsize=(16, 8), layout="constrained")
    ejes[0, 0].imshow(imagen[sub.loc[i, "image"]]); ejes[0, 0].set_title(f"Parche ({aumento})")
    barra_escala(ejes[0, 0], aumento)
    ejes[0, 1].imshow(ref, cmap=CMAP, vmin=0, vmax=2, interpolation="nearest"); ejes[0, 1].set_title("Anotación del experto")
    ejes[0, 2].imshow(pred, cmap=CMAP, vmin=0, vmax=2, interpolation="nearest"); ejes[0, 2].set_title("Predicción (clase más probable)")
    ejes[0, 2].legend(handles=LEYENDA, loc="lower right", fontsize=8)
    im = ejes[0, 3].imshow(p.max(0), cmap="viridis", vmin=1 / 3, vmax=1)
    ejes[0, 3].set_title("Confianza = P(clase ganadora)")
    fig.colorbar(im, ax=ejes[0, 3], shrink=0.8)
    for c in range(3):
        im = ejes[1, c].imshow(p[c], cmap="magma", vmin=0, vmax=1); ejes[1, c].set_title(f"P({CLASES[c]})")
    fig.colorbar(im, ax=ejes[1, :3], shrink=0.8, label="probabilidad")
    for e in ejes.flat[:7]:
        e.plot(columna, fila, "+", color="cyan", ms=16, mew=2)
    for e in ejes.flat:
        e.axis("off")
    texto = "\n".join(f"P({CLASES[c]}) = {p[c, fila, columna]:.3f}" for c in range(3))
    ejes[1, 3].text(0.05, 0.6, f"Píxel (fila {fila}, columna {columna})\n\n{texto}\n{'─' * 22}\n"
                    f"suma           = {p[:, fila, columna].sum():.3f}\n\npredicción: {CLASES[pred[fila, columna]]}\n"
                    f"anotación:  {CLASES[ref[fila, columna]]}", family="monospace", fontsize=12, va="center")
    plt.show()
    print(" · ".join(f"Dice {CLASES[c]} = {dice(pred, ref, c):.2f}" for c in (1, 2))
          + "   (nan = la clase no está en el parche ni en la predicción)")


w.interact(inferencia,
           aumento=w.Dropdown(options=AUMENTOS, value="20x", description="Aumento"),
           region=w.Dropdown(options=sorted(indice.region.unique()), value="burbujas_aire", description="Región"),
           k=w.IntSlider(value=0, min=0, max=2, description="Parche"),
           fila=w.IntSlider(value=150, min=0, max=299, description="Fila", continuous_update=False),
           columna=w.IntSlider(value=150, min=0, max=299, description="Columna", continuous_update=False));

# %% [markdown]
# ### ✍️ Pregunta 1 · ¿Dónde duda la red?
# La red está **segura** en un píxel cuando una clase se lleva casi toda la probabilidad, por ejemplo
# 0,98 / 0,01 / 0,01: la confianza es 0,98. Está **insegura** cuando la probabilidad se reparte, por ejemplo
# 0,45 / 0,40 / 0,15: la clase ganadora gana por poco (confianza 0,45), y un cambio pequeño en la imagen podría
# cambiar la predicción. Con 3 clases, la confianza va de 1/3 (indecisión total) a 1.
#
# 1. En el mapa de **confianza**, ¿dónde están las zonas oscuras? ¿En el interior de las regiones o en los bordes
#    entre clases? Lleva el **+** a una de ellas y anota las tres probabilidades.
# 2. ¿Las zonas de baja confianza coinciden con los errores (compara predicción con anotación)?
# 3. Recorre varias regiones de artefacto: ¿cuál le cuesta más a la red?
#
# _Respuesta:_
#
# ## Paso 3 · ¿A qué aumento trabajar?
# Más aumento muestra más detalle, pero cada paso (10x → 20x → 40x) multiplica por 4 el número de píxeles y
# de parches. El aumento correcto depende de **qué tamaño tienen las estructuras que necesitas ver** y de
# **cuánto tiempo puedes gastar** por lámina.
#
# **3a.** Simulamos la misma zona a los tres aumentos reduciendo la resolución de un parche de 40x.

# %%
#@title 🔍 La misma zona a 40x, 20x y 10x
ESTRUCTURAS_UM = {  # tamaños típicos aproximados
    "fragmento de tejido": 5000, "burbuja de aire": 500, "pliegue de tejido": 200,
    "cripta de colon (ancho)": 60, "núcleo celular": 7, "nucléolo / cromosomas de una mitosis": 1.5,
}


def misma_zona(region, k):
    f = indice[(indice.organo == "colon") & (indice.aumento == "40x") & (indice.region == region) & (indice.k == k)].iloc[0]
    original = Image.fromarray(imagen[f.image])
    fig, ejes = plt.subplots(1, 3, figsize=(13, 4.8), layout="constrained")
    for e, aum in zip(ejes, ("40x", "20x", "10x")):
        lado = int(300 * UM_POR_PIXEL["40x"] / UM_POR_PIXEL[aum])
        reducida = original.resize((lado, lado), Image.BOX).resize((300, 300), Image.NEAREST)
        e.imshow(reducida); e.axis("off"); e.set_title(f"{aum}: la zona ocupa {lado} × {lado} píxeles")
        barra_escala(e, "40x")
    fig.suptitle(f"{region} #{k}: la misma zona de {300 * UM_POR_PIXEL['40x']:.0f} µm a tres aumentos")
    plt.show()
    tabla = pd.DataFrame({"tamaño (µm)": ESTRUCTURAS_UM})
    for aum in AUMENTOS:
        tabla[f"píxeles a {aum}"] = (tabla["tamaño (µm)"] / UM_POR_PIXEL[aum]).round(0).astype(int)
    print("Cuántos píxeles mide cada estructura (regla práctica: para segmentarla bien debería medir ≥ 10 píxeles):")
    display(tabla.style.map(lambda v: f"background-color: {'#d4edda' if v >= 10 else '#f8d7da'}",
                            subset=[f"píxeles a {a}" for a in AUMENTOS]).format("{:g}"))


w.interact(misma_zona,
           region=w.Dropdown(options=sorted(indice.region.unique()), value="mucosa", description="Región"),
           k=w.IntSlider(value=0, min=0, max=2, description="Parche"));

# %% [markdown]
# **3b.** ¿Cambia la calidad de la segmentación de tejido y artefacto? Corremos cada modelo sobre los parches de
# colon de su aumento (~30 s en CPU).
#
# > 💡 **Recuerda:** a diferencia de la simulación de 3a, aquí los parches de cada aumento son **distintos** (no son
# > versiones reducidas de los mismos parches de 40x). Al comparar aumentos comparamos también zonas distintas.
#
# **Cómo se calcula el Dice con 3 clases.** Se calcula **una clase a la vez**, como si fuera un problema binario
# (esa clase contra el resto). Para la clase *c*, sea **A** el conjunto de píxeles que la red predijo como *c* y
# **B** el de píxeles que el experto anotó como *c*:
#
# $$\text{Dice}_c = \frac{2\,|A \cap B|}{|A| + |B|}$$
#
# Vale 1 si coinciden exactamente y 0 si no se tocan. Reportamos el Dice de **tejido** y el de **artefacto**; el
# de fondo casi siempre es ~1 y no aporta.
#
# Con muchos parches hay dos formas de juntar el resultado:
# - **Agrupado**: se juntan los píxeles de todos los parches, como si fueran una sola imagen grande, y se calcula
#   un solo Dice. Cada **píxel** pesa lo mismo.
# - **Promedio por parche**: se calcula un Dice en cada parche y se promedian. Cada **parche** pesa lo mismo, tenga
#   mucho o poco de esa clase. Los parches donde la clase no aparece ni en la anotación ni en la predicción
#   (Dice indefinido) no entran al promedio.
#
# Ejemplo con dos parches y la clase artefacto:
#
# | | artefacto anotado | artefacto predicho | aciertos | Dice del parche |
# |---|---|---|---|---|
# | Parche 1 (una burbuja grande) | 40.000 px | 40.000 px | 36.000 px | 0,90 |
# | Parche 2 (un borde de 200 px) | 200 px | 200 px | 20 px | 0,10 |
#
# **Promedio por parche** = (0,90 + 0,10) / 2 = **0,50**.
# **Agrupado** = 2 × (36.000 + 20) / (40.200 + 40.200) = **0,90**.
# Un error en 200 píxeles hunde el promedio por parche, pero casi no mueve el agrupado.

# %%
#@title 🧮 Dice a cada aumento
filas, por_region = [], []
for aum in AUMENTOS:
    sub = indice[(indice.organo == "colon") & (indice.aumento == aum)].reset_index(drop=True)
    pred, ref = probabilidades("colon", aum).argmax(1), referencia("colon", aum)
    for c in (1, 2):
        filas.append({"aumento": aum, "clase": CLASES[c], "Dice agrupado": dice(pred, ref, c),
                      "Dice promedio por parche": np.nanmean([dice(pred[i], ref[i], c) for i in range(len(ref))])})
    for region, g in sub[sub.categoria != "fondo"].groupby("region"):
        c = {"tejido": 1, "artefacto": 2}[g.categoria.iloc[0]]
        por_region.append({"categoría": g.categoria.iloc[0], "región": region, "aumento": aum,
                           "Dice": dice(pred[g.index], ref[g.index], c)})

print("Todos los parches de colon de cada aumento (39):")
display(pd.DataFrame(filas).pivot(index="clase", columns="aumento").round(2))
print("Dice agrupado de cada región, sobre sus 3 parches y solo en la clase de esa región\n"
      "(regiones de tejido → Dice de tejido; regiones de artefacto → Dice de artefacto).\n"
      "Ojo: los parches de cada aumento son distintos (se sortearon por separado) y son solo 3 por región,\n"
      "así que una diferencia entre aumentos puede deberse a qué parches tocaron y no al aumento.")
display(pd.DataFrame(por_region).pivot_table(index=["categoría", "región"], columns="aumento", values="Dice")
        .style.background_gradient(cmap="Greens", vmin=0.4, vmax=1).format("{:.2f}"))

# %% [markdown]
# **3c.** ¿Cuánto cuesta procesar una lámina completa a cada aumento?

# %%
#@title ⏱️ ¿Cuánto cuesta procesar una lámina completa?
#@markdown Tamaño del tejido en la lámina (mm):
alto_mm = 15  #@param {type:"number"}
ancho_mm = 20  #@param {type:"number"}

x = torch.rand(16, 3, 300, 300)
t0 = time.time()
with torch.inference_mode():
    modelos["20x"](x)
seg_por_parche = (time.time() - t0) / 16
filas = []
for aum, um in UM_POR_PIXEL.items():
    n = int(np.ceil(alto_mm * 1000 / um / 300) * np.ceil(ancho_mm * 1000 / um / 300))
    filas.append({"aumento": aum, "parches de 300 px": n, "minutos en esta CPU": n * seg_por_parche / 60})
display(pd.DataFrame(filas).style.format({"parches de 300 px": "{:,}", "minutos en esta CPU": "{:.1f}"}).hide(axis="index"))

# %% [markdown]
# **3d.** Así se ve una lámina completa segmentada: la misma lámina de colon de la introducción, recorrida
# parche a parche por la U-Net de 10x y vuelta a armar.
#
# ![Lámina completa de colon segmentada en fondo, tejido y artefacto](https://raw.githubusercontent.com/robertorojasp06/mim-uchile-hands-on/main/notebooks/img/S4A_lamina_segmentada.jpg)
#
# Fíjate en tres cosas:
# - Los cortes de tejido salen limpios, y el borde del portaobjetos y la sombra del extremo derecho salen como artefacto.
# - En las burbujas de aire bajo el cubreobjetos (zoom c–d) la red **duda**: mezcla píxeles de tejido y de
#   artefacto en un patrón granulado. Es la misma baja confianza que viste en el Paso 2, ahora a escala de lámina.
# - Para esta lámina se usaron 15.225 parches y ~20 min de CPU a 10x (en el computador del laboratorio).
#   A 40x serían 16 veces más.
#
# ### ✍️ Decisión 1
# 1. ¿Mejora la segmentación de tejido y artefacto al subir el aumento? ¿Compensa multiplicar el costo por 16?
# 2. ¿Por qué el Dice **promedio por parche** es más bajo que el **agrupado**? ¿Cuál reportarías?
# 3. Elige un aumento para cada escenario y justifícalo con la tabla de tamaños y la de costo:
#    - **Control de calidad**: revisar **1.000 láminas** por semana para detectar las que tienen muchas
#      burbujas, pliegues o desenfoque y deben volver a escanearse.
#    - **Conteo de mitosis**: el paso siguiente cuenta células en división dentro del tejido conservado.
# 4. ¿Podrías combinar aumentos en un mismo flujo (por ejemplo, la máscara de tejido a uno y el análisis a otro)?
#
# _Respuesta:_
#
# ## Paso 4 · Validación externa: ¿funciona en estómago?
# Los modelos se entrenaron solo con **colon**. Ahora los probamos en parches de **mucosa gástrica**,
# usando el modelo del mismo aumento que los datos.

# %%
#@title 🌍 Colon vs. gástrico
filas, global_ = [], []
for organo in ("colon", "gastrico"):
    for aum in AUMENTOS:
        sub = indice[(indice.organo == organo) & (indice.aumento == aum)].reset_index(drop=True)
        pred, ref = probabilidades(organo, aum).argmax(1), referencia(organo, aum)
        global_.append({"órgano": organo, "aumento": aum, **{f"Dice {CLASES[c]}": dice(pred, ref, c) for c in (1, 2)}})
        for region, g in sub.groupby("region"):
            c = {"tejido": 1, "artefacto": 2, "fondo": 0}[g.categoria.iloc[0]]
            filas.append({"órgano": organo, "aumento": aum, "región": region, "categoría": g.categoria.iloc[0],
                          "Dice de su clase": dice(pred[g.index], ref[g.index], c)})
externa = pd.DataFrame(filas)
display(pd.DataFrame(global_).pivot(index="aumento", columns="órgano").round(2))

tabla = externa.pivot_table(index=["categoría", "región"], columns="órgano", values="Dice de su clase").round(2)
tabla["diferencia"] = (tabla["gastrico"] - tabla["colon"]).round(2)
display(tabla.style.background_gradient(subset=["diferencia"], cmap="RdYlGn", vmin=-0.4, vmax=0.4).format("{:.2f}", na_rep="—"))

# %% [markdown]
# ### ✍️ Decisión 2
# 1. ¿En qué regiones el modelo **empeora** al pasar a estómago? ¿Por qué podría ocurrir?
# 2. ¿Lo usarías en estómago sin reentrenar? ¿Qué necesitarías para estar seguro?
#
# _Respuesta:_
#
# ### Para llevarse
# - El aumento se elige según el **tamaño de lo que hay que ver** y el **costo**: cada paso multiplica por 4 los píxeles.
# - **Cómo se agrega** una métrica (por parche o agrupada) cambia el número: hay que decirlo al reportar.
# - La validación **externa** (otro órgano, otro hospital, otro escáner) muestra dónde falla el modelo.
