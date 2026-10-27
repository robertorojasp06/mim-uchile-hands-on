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
# con parches de 300 × 300 píxeles. Los parches de este notebook **no** se usaron para entrenar.
#
# | Paso | Qué haces | Tiempo |
# |---|---|---|
# | 1 | Exploras los parches y sus anotaciones | 3 min |
# | 2 | Corres la U-Net en un parche y lees sus probabilidades | 5 min |
# | 3 | Decides a qué aumento trabajar | 6 min |
# | 4 | Pruebas el modelo en otro órgano (validación externa) | 5 min |
# | 5 | Eliges el umbral de artefacto | 5 min |

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
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from PIL import Image

RAIZ = obtener_datos("wsi")
indice = pd.read_csv(RAIZ / "index.csv")
AUMENTOS = ["10x", "20x", "40x"]
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
        return self.conv8(x)


modelos = {}
for aum in AUMENTOS:
    modelos[aum] = UNet300()
    modelos[aum].load_state_dict(torch.load(RAIZ / "models" / f"unet_colon_{aum}.pt", map_location="cpu"))
    modelos[aum].eval()
print(f"Parámetros de la U-Net: {sum(p.numel() for p in modelos['10x'].parameters()):,}".replace(",", "."))

imagen = {r: np.asarray(Image.open(RAIZ / r)) for r in indice.image}
mascara = {r: np.asarray(Image.open(RAIZ / r)) for r in indice["mask"]}
_cache = {}


def probabilidades(modelo, organo, aumento):
    """Probabilidades (n, 3, 300, 300) del modelo `modelo` sobre los parches de `organo` a `aumento`."""
    clave = (modelo, organo, aumento)
    if clave not in _cache:
        sub = indice[(indice.organo == organo) & (indice.aumento == aumento)]
        x = torch.tensor(np.stack([imagen[r] for r in sub.image]), dtype=torch.float32).permute(0, 3, 1, 2) / 255
        with torch.inference_mode():
            p = torch.cat([F.softmax(modelos[modelo](x[i:i + 16]), 1) for i in range(0, len(x), 16)])
        _cache[clave] = p.numpy().astype(np.float16)
    return _cache[clave]


def referencia(organo, aumento):
    sub = indice[(indice.organo == organo) & (indice.aumento == aumento)]
    return np.stack([mascara[r] for r in sub["mask"]])


def dice(pred, ref, clase):
    a, b = pred == clase, ref == clase
    s = a.sum() + b.sum()
    return 2 * (a & b).sum() / s if s else np.nan


print(indice.groupby(["organo", "aumento"]).size().unstack())

# %% [markdown]
# ## Paso 1 · Explora los parches
# Cada parche tiene una **anotación** hecha por un experto: <span style="color:#c2185b">**tejido**</span>,
# <span style="color:#1976d2">**artefacto**</span> o fondo.

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
# La red entrega, para cada píxel, una **probabilidad** por clase. La predicción final es la clase más probable.

# %%
#@title 🤖 Inferencia


def inferencia(modelo, aumento, region, k):
    sub = indice[(indice.organo == "colon") & (indice.aumento == aumento)].reset_index(drop=True)
    fila = sub[(sub.region == region) & (sub.k == k)].index[0]
    p = probabilidades(modelo, "colon", aumento)[fila].astype(np.float32)
    pred, ref = p.argmax(0), mascara[sub.loc[fila, "mask"]]
    fig, ejes = plt.subplots(1, 5, figsize=(17, 3.8))
    ejes[0].imshow(imagen[sub.loc[fila, "image"]]); ejes[0].set_title(f"Parche ({aumento})")
    ejes[1].imshow(ref, cmap=CMAP, vmin=0, vmax=2, interpolation="nearest"); ejes[1].set_title("Anotación del experto")
    ejes[2].imshow(pred, cmap=CMAP, vmin=0, vmax=2, interpolation="nearest"); ejes[2].set_title(f"Predicción (modelo {modelo})")
    for e, c in zip(ejes[3:], (1, 2)):
        im = e.imshow(p[c], cmap="magma", vmin=0, vmax=1); e.set_title(f"P({CLASES[c]})")
    fig.colorbar(im, ax=ejes[3:], fraction=0.025)
    for e in ejes:
        e.axis("off")
    plt.show()
    print(" · ".join(f"Dice {CLASES[c]} = {dice(pred, ref, c):.2f}" for c in (1, 2))
          + "   (nan = la clase no está en el parche ni en la predicción)")


w.interact(inferencia,
           modelo=w.Dropdown(options=AUMENTOS, value="20x", description="Modelo"),
           aumento=w.Dropdown(options=AUMENTOS, value="20x", description="Datos"),
           region=w.Dropdown(options=sorted(indice.region.unique()), value="burbujas_aire", description="Región"),
           k=w.IntSlider(value=0, min=0, max=2, description="Parche"));

# %% [markdown]
# ### ✍️ Pregunta 1
# ¿Dónde está **menos segura** la red (probabilidades cercanas a 0,5)? ¿Qué artefacto le cuesta más?
#
# _Respuesta:_
#
# ## Paso 3 · ¿A qué aumento trabajar?
# Corremos los **3 modelos** sobre los parches de colon a los **3 aumentos** (9 combinaciones; ~1 min en CPU).
# El Dice de cada clase se calcula de dos formas:
# - **agrupado**: se suman los píxeles de todos los parches y se calcula un solo Dice;
# - **promedio por parche**: se calcula un Dice por parche y se promedia.

# %%
#@title 🧮 Matriz modelo × aumento de los datos
filas = []
for mod in AUMENTOS:
    for aum in AUMENTOS:
        pred, ref = probabilidades(mod, "colon", aum).argmax(1), referencia("colon", aum)
        for c in (1, 2):
            filas.append({"modelo": mod, "datos": aum, "clase": CLASES[c], "Dice agrupado": dice(pred, ref, c),
                          "Dice promedio por parche": np.nanmean([dice(pred[i], ref[i], c) for i in range(len(ref))])})
matriz = pd.DataFrame(filas)

fig, ejes = plt.subplots(1, 2, figsize=(11, 4))
for e, clase in zip(ejes, ("tejido", "artefacto")):
    m = matriz[matriz.clase == clase].pivot(index="modelo", columns="datos", values="Dice agrupado")
    e.imshow(m, cmap="Greens", vmin=0.6, vmax=1)
    for (i, j), v in np.ndenumerate(m.to_numpy()):
        e.text(j, i, f"{v:.2f}", ha="center", va="center", color="white" if v > 0.9 else "black")
    e.set_xticks(range(3), m.columns); e.set_yticks(range(3), m.index)
    e.set_xlabel("aumento de los datos"); e.set_ylabel("modelo entrenado a"); e.set_title(f"Dice agrupado · {clase}")
plt.tight_layout(); plt.show()
display(matriz.pivot_table(index=["modelo", "datos"], columns="clase").round(2))

# %%
#@title ⏱️ ¿Cuánto cuesta procesar una lámina completa?
#@markdown Tamaño del tejido en la lámina (mm):
alto_mm = 15  #@param {type:"number"}
ancho_mm = 20  #@param {type:"number"}
UM_POR_PIXEL = {"10x": 0.92, "20x": 0.46, "40x": 0.23}   # escáner Hamamatsu NanoZoomer (aprox.)

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
# ### ✍️ Decisión 1
# 1. ¿El modelo funciona mejor cuando el aumento de los datos coincide con el del entrenamiento? ¿Cuánto importa?
# 2. ¿Por qué el Dice **promedio por parche** es más bajo que el **agrupado**? ¿Cuál reportarías?
# 3. Hay que procesar **1.000 láminas** por semana para un control de calidad. ¿Qué aumento eliges y por qué?
#
# _Respuesta:_
#
# ## Paso 4 · Validación externa: ¿funciona en estómago?
# Los modelos se entrenaron solo con **colon**. Ahora los probamos en parches de **mucosa gástrica**,
# usando el modelo del mismo aumento que los datos.

# %%
#@title 🌍 Colon vs. gástrico
filas = []
for organo in ("colon", "gastrico"):
    for aum in AUMENTOS:
        sub = indice[(indice.organo == organo) & (indice.aumento == aum)].reset_index(drop=True)
        pred, ref = probabilidades(aum, organo, aum).argmax(1), referencia(organo, aum)
        for region, g in sub.groupby("region"):
            c = {"tejido": 1, "artefacto": 2, "fondo": 0}[g.categoria.iloc[0]]
            filas.append({"órgano": organo, "aumento": aum, "región": region, "categoría": g.categoria.iloc[0],
                          "Dice de su clase": dice(pred[g.index], ref[g.index], c)})
externa = pd.DataFrame(filas)

global_ = []
for organo in ("colon", "gastrico"):
    for aum in AUMENTOS:
        pred, ref = probabilidades(aum, organo, aum).argmax(1), referencia(organo, aum)
        global_.append({"órgano": organo, "aumento": aum, **{f"Dice {CLASES[c]}": dice(pred, ref, c) for c in (1, 2)}})
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
# ## Paso 5 · El umbral de artefacto
# En vez de elegir la clase más probable, marcamos un píxel como **artefacto** si P(artefacto) ≥ umbral.
# - Umbral **bajo** → se descarta más: menos artefactos colados, pero se pierde tejido útil.
# - Umbral **alto** → se descarta menos: se conserva tejido, pero se cuelan artefactos al análisis.

# %%
#@title 🎚️ Umbral de artefacto


def con_umbral(organo, aumento, umbral):
    p = probabilidades(aumento, organo, aumento).astype(np.float32)
    ref = referencia(organo, aumento)
    umbrales = np.linspace(0.05, 0.95, 19)
    perdido, colado = [], []
    for u in umbrales:
        art = p[:, 2] >= u
        perdido.append((art & (ref == 1)).sum() / (ref == 1).sum())
        colado.append((~art & (ref == 2)).sum() / (ref == 2).sum())
    art = p[:, 2] >= umbral
    pred = np.where(art, 2, p[:, :2].argmax(1))
    plt.figure(figsize=(8, 3.8))
    plt.plot(umbrales, 100 * np.array(perdido), "o-", color="#c2185b", label="tejido real descartado como artefacto")
    plt.plot(umbrales, 100 * np.array(colado), "o-", color="#1976d2", label="artefacto real que se cuela como tejido/fondo")
    plt.axvline(umbral, color="k", ls=":")
    plt.xlabel("umbral de P(artefacto)"); plt.ylabel("% de píxeles"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.show()
    print(f"Umbral {umbral:.2f}: Dice tejido = {dice(pred, ref, 1):.2f} · Dice artefacto = {dice(pred, ref, 2):.2f} · "
          f"tejido descartado = {100 * (art & (ref == 1)).sum() / (ref == 1).sum():.1f} % · "
          f"artefacto colado = {100 * (~art & (ref == 2)).sum() / (ref == 2).sum():.1f} %")


w.interact(con_umbral,
           organo=w.Dropdown(options=["colon", "gastrico"], description="Órgano"),
           aumento=w.Dropdown(options=AUMENTOS, value="20x", description="Aumento"),
           umbral=w.FloatSlider(value=0.5, min=0.05, max=0.95, step=0.05, description="Umbral", continuous_update=False));

# %% [markdown]
# ### ✍️ Decisión 3
# El siguiente paso del flujo es **cuantificar células tumorales** en el tejido conservado.
# ¿Qué error es más grave: descartar tejido o dejar pasar artefactos? Elige un umbral y justifícalo.
#
# _Umbral elegido y por qué:_
#
# ### Para llevarse
# - Un modelo preentrenado se usa **en las mismas condiciones** en que se entrenó (aumento, tinción, órgano) o se **valida** de nuevo.
# - **Cómo se agrega** una métrica (por parche o agrupada) cambia el número: hay que decirlo al reportar.
# - La validación **externa** (otro órgano, otro hospital, otro escáner) muestra dónde falla el modelo.
# - El umbral traslada a números una **decisión del flujo clínico**.
