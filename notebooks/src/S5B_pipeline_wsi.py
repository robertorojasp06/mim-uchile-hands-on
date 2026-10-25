# %% [markdown]
# # Sesión 5 · Actividad B — De la lámina completa a los parches: un *pipeline* de patología digital
#
# **Duración:** 25 min · **Cómputo:** CPU · **Datos:** una lámina H&E de colon (proyecto DeepPath, anonimizada)
#
# Una lámina escaneada a 40x (*whole slide image*, WSI) tiene **miles de millones de píxeles**: ninguna red
# la procesa entera. El flujo típico es:
#
# **miniatura → detectar tejido → dividir en parches → aplicar el modelo a cada parche → reconstruir**
#
# Cada paso tiene decisiones que cambian el resultado y el **costo**. Aquí las tomas tú.
#
# | Paso | Qué haces | Tiempo |
# |---|---|---|
# | 1 | Exploras la pirámide de resoluciones | 3 min |
# | 2 | Detectas el tejido en la miniatura | 6 min |
# | 3 | Decides el aumento y el tamaño de los parches | 6 min |
# | 4 | Segmentas una región con la U-Net y reconstruyes la máscara | 6 min |
# | 5 | Configuras el *pipeline* para un hospital | 4 min |

# %%
# INCLUDE: _setup.py

# %%
#@title 📦 Cargar la lámina y la U-Net
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as w
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle, Patch
from PIL import Image
from scipy import ndimage
from skimage import color, filters, morphology

Image.MAX_IMAGE_PIXELS = None
RAIZ = obtener_datos("lamina")
info = json.loads((RAIZ / "info.json").read_text())
miniatura = np.asarray(Image.open(RAIZ / "miniatura.jpg"))
regiones = {k: np.asarray(Image.open(RAIZ / f"region_{k}.jpg")) for k in info["regiones"]}
RED_MIN = info["miniatura"]["reduccion"]            # nivel 0 -> miniatura
CMAP = ListedColormap(["#eeeeee", "#c2185b", "#1976d2"])


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
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(F.max_pool2d(x1, 2)))
        x3 = F.relu(self.conv3(F.max_pool2d(x2, 2)))
        x = sube(F.relu(self.conv4(F.max_pool2d(x3, 2))), size=x3.shape[-2:])
        x = sube(F.relu(self.conv5(torch.cat((x, x3), 1))), scale_factor=2)
        x = sube(F.relu(self.conv6(torch.cat((x, x2), 1))), scale_factor=2)
        x = F.relu(self.conv7(torch.cat((x, x1), 1)))
        return self.conv8(x)


unet = UNet300()
unet.load_state_dict(torch.load(RAIZ / "unet_colon_10x.pt", map_location="cpu"))
unet.eval()
with torch.inference_mode():
    t0 = time.time(); unet(torch.rand(16, 3, 300, 300)); SEG_POR_PARCHE = (time.time() - t0) / 16
print(f"U-Net: {SEG_POR_PARCHE * 1000:.0f} ms por parche de 300 × 300 en este computador")

# %% [markdown]
# ## Paso 1 · La pirámide
# El archivo guarda la misma lámina a varias resoluciones. Los recuadros verdes son las dos regiones que
# segmentaremos en el Paso 4.

# %%
#@title 🔺 Niveles de la pirámide
niveles = pd.DataFrame(info["niveles"])
niveles["megapíxeles"] = niveles.ancho * niveles.alto / 1e6
niveles["GB sin comprimir (RGB)"] = niveles.ancho * niveles.alto * 3 / 1e9
display(niveles[["nivel", "aumento", "um_por_pixel", "ancho", "alto", "megapíxeles", "GB sin comprimir (RGB)"]]
        .head(8).style.format({"aumento": "{:g}x", "um_por_pixel": "{:.2f}", "megapíxeles": "{:,.0f}",
                               "GB sin comprimir (RGB)": "{:.2f}"}).hide(axis="index"))

plt.figure(figsize=(10, 8.2))
plt.imshow(miniatura)
for k, r in info["regiones"].items():
    lado = r["lado"] * r["reduccion"] / RED_MIN
    plt.gca().add_patch(Rectangle((r["x0"] / RED_MIN, r["y0"] / RED_MIN), lado, lado, fill=False, ec="#00c853", lw=2))
    plt.text(r["x0"] / RED_MIN, r["y0"] / RED_MIN - 8, f"Región {k}", color="#00c853", fontsize=12, weight="bold")
plt.title(f"Miniatura ({info['niveles'][info['miniatura']['nivel']]['aumento']:g}x)"); plt.axis("off"); plt.show()

# %% [markdown]
# ### ✍️ Pregunta 1
# ¿Cuántos GB ocuparía la lámina a 40x sin comprimir? ¿Por qué el escáner guarda una pirámide y no solo la imagen a 40x?
#
# _Respuesta:_
#
# ## Paso 2 · ¿Dónde hay tejido?
# Antes de usar cualquier red, se descarta el vidrio vacío con un método **clásico** en la miniatura:
# un **umbral** sobre la **saturación** (el tejido teñido es colorido; el vidrio es gris) o sobre el **brillo**.
# El método de **Otsu** elige el umbral automáticamente.

# %%
#@title 🧫 Detección de tejido
hsv = color.rgb2hsv(miniatura)
gris = color.rgb2gray(miniatura)
UMBRAL_OTSU = {"saturación": filters.threshold_otsu(hsv[..., 1]), "brillo": filters.threshold_otsu(gris)}


def detectar_tejido(canal, umbral, tamano_minimo, rellenar):
    m = hsv[..., 1] > umbral if canal == "saturación" else gris < umbral
    m = morphology.remove_small_objects(m, tamano_minimo)
    if rellenar:
        m = ndimage.binary_fill_holes(m)
    return m


def ver_tejido(canal, usar_otsu, umbral_manual, tamano_minimo, rellenar):
    global tejido
    umbral = UMBRAL_OTSU[canal] if usar_otsu else umbral_manual
    tejido = detectar_tejido(canal, umbral, tamano_minimo, rellenar)
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 5.8))
    a.imshow(miniatura); a.imshow(np.ma.masked_where(~tejido, tejido), cmap="spring", alpha=0.35); a.axis("off")
    a.set_title(f"Tejido detectado: {100 * tejido.mean():.1f} % de la lámina")
    valores = hsv[..., 1] if canal == "saturación" else gris
    b.hist(valores.ravel(), bins=100, color="#888"); b.axvline(umbral, color="#d81b60", lw=2)
    b.set_xlabel(canal); b.set_yscale("log"); b.set_title(f"Umbral = {umbral:.3f}" + (" (Otsu)" if usar_otsu else ""))
    plt.tight_layout(); plt.show()


tejido = None
w.interact(ver_tejido,
           canal=w.Dropdown(options=["saturación", "brillo"], description="Canal"),
           usar_otsu=w.Checkbox(value=True, description="Umbral de Otsu"),
           umbral_manual=w.FloatSlider(value=0.1, min=0.0, max=1.0, step=0.01, description="Umbral manual",
                                       continuous_update=False),
           tamano_minimo=w.IntSlider(value=200, min=0, max=3000, step=50, description="Objeto mín. (px)",
                                     continuous_update=False),
           rellenar=w.Checkbox(value=True, description="Rellenar huecos"));

# %% [markdown]
# ### ✍️ Pregunta 2
# Compara **saturación** y **brillo** con Otsu. ¿Qué detecta de más el brillo? (Mira los bordes de la lámina.)
# ¿Para qué sirve eliminar objetos pequeños y rellenar huecos?
#
# _Respuesta:_
#
# ## Paso 3 · ¿Cuántos parches?
# La lámina se divide en una grilla de parches. Solo se procesan los que tienen suficiente tejido.
# (Usa la máscara de tejido que dejaste en el Paso 2.)

# %%
#@title 🧩 Teselado


def teselar(aumento, lado_parche, tejido_minimo):
    red = {"10x": 4, "20x": 2, "40x": 1}[aumento]
    paso = lado_parche * red / RED_MIN                  # lado del parche en píxeles de la miniatura
    nx, ny = int(np.ceil(tejido.shape[1] / paso)), int(np.ceil(tejido.shape[0] / paso))
    frac = np.asarray(Image.fromarray(tejido.astype(np.float32)).resize((nx, ny), Image.BOX))
    usados = frac >= tejido_minimo / 100
    n = int(usados.sum())
    minutos = n * SEG_POR_PARCHE * (lado_parche / 300) ** 2 / 60
    plt.figure(figsize=(10, 8.2))
    plt.imshow(miniatura)
    plt.imshow(np.ma.masked_where(~usados, usados), cmap="winter", alpha=0.35, interpolation="nearest",
               extent=(0, nx * paso, ny * paso, 0))
    plt.xlim(0, tejido.shape[1]); plt.ylim(tejido.shape[0], 0); plt.axis("off")
    plt.title(f"{aumento}, parches de {lado_parche} px: {n:,} de {nx * ny:,} parches".replace(",", "."))
    plt.show()
    print(f"Tiempo estimado de la U-Net en esta CPU: {minutos:.1f} min · "
          f"almacenamiento de los parches (JPEG ~1/10): {n * lado_parche ** 2 * 3 / 10 / 1e9:.2f} GB")


w.interact(teselar,
           aumento=w.Dropdown(options=["10x", "20x", "40x"], value="10x", description="Aumento"),
           lado_parche=w.Dropdown(options=[256, 300, 512], value=300, description="Parche (px)"),
           tejido_minimo=w.IntSlider(value=10, min=0, max=100, step=5, description="% tejido mín.",
                                     continuous_update=False));

# %% [markdown]
# ### ✍️ Decisión 1
# Con un umbral de tejido de 0 % se procesa toda la grilla. ¿Cuánto tiempo ahorra la detección de tejido?
# ¿Qué se arriesga si el **% de tejido mínimo** es muy alto? (Piensa en los bordes del tejido.)
#
# _Respuesta:_
#
# ## Paso 4 · Segmentar una región y reconstruir
# Aplicamos la U-Net (entrenada a 10x) parche a parche sobre una región de 3.000 × 3.000 px a 10x, de dos formas:
# - **sin superposición**: parches contiguos de 300 px;
# - **con superposición del 50 %**: cada píxel se predice varias veces y se promedian las probabilidades.

# %%
#@title 🔬 Segmentar una región (≈20–60 s)


def segmentar(img, paso):
    x = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255
    H, W = img.shape[:2]
    suma, cuenta = torch.zeros(3, H, W), torch.zeros(1, H, W)
    esquinas = [(f, c) for f in range(0, H - 300 + 1, paso) for c in range(0, W - 300 + 1, paso)]
    with torch.inference_mode():
        for i in range(0, len(esquinas), 16):
            lote = esquinas[i:i + 16]
            p = F.softmax(unet(torch.stack([x[:, f:f + 300, c:c + 300] for f, c in lote])), 1)
            for (f, c), pi in zip(lote, p):
                suma[:, f:f + 300, c:c + 300] += pi
                cuenta[:, f:f + 300, c:c + 300] += 1
    return (suma / cuenta).argmax(0).numpy().astype(np.uint8), len(esquinas)


def ver_region(region):
    img = regiones[region]
    t0 = time.time(); sin, n1 = segmentar(img, 300); t1 = time.time() - t0
    t0 = time.time(); con, n2 = segmentar(img, 150); t2 = time.time() - t0
    fig, ejes = plt.subplots(1, 3, figsize=(16, 5.6))
    ejes[0].imshow(img); ejes[0].set_title(f"Región {region} (10x)")
    ejes[1].imshow(sin, cmap=CMAP, vmin=0, vmax=2, interpolation="nearest")
    ejes[1].set_title(f"Sin superposición · {n1} parches · {t1:.0f} s")
    ejes[2].imshow(con, cmap=CMAP, vmin=0, vmax=2, interpolation="nearest")
    ejes[2].set_title(f"Superposición 50 % · {n2} parches · {t2:.0f} s")
    for e in ejes:
        e.axis("off")
    ejes[2].legend(handles=[Patch(color=CMAP(i), label=t) for i, t in enumerate(["fondo", "tejido", "artefacto"])],
                   loc="lower right")
    plt.tight_layout(); plt.show()

    z = (slice(1200, 1800), slice(1200, 1800))       # zoom en una esquina de parches (múltiplo de 300)
    fig, ejes = plt.subplots(1, 3, figsize=(13, 4.4))
    ejes[0].imshow(img[z]); ejes[1].imshow(sin[z], cmap=CMAP, vmin=0, vmax=2, interpolation="nearest")
    ejes[2].imshow(con[z], cmap=CMAP, vmin=0, vmax=2, interpolation="nearest")
    for e, t in zip(ejes, ["Zoom", "Sin superposición", "Con superposición"]):
        e.set_title(t); e.axis("off")
        for v in (300, 600):                             # bordes de los parches sin superposición
            e.axhline(v - 0.5, color="k", lw=0.6, ls=":"); e.axvline(v - 0.5, color="k", lw=0.6, ls=":")
    plt.tight_layout(); plt.show()

    r = info["regiones"][region]
    escala = r["reduccion"] / RED_MIN
    f0, c0 = int(r["y0"] / RED_MIN), int(r["x0"] / RED_MIN)
    lado = int(round(r["lado"] * escala))
    otsu = np.asarray(Image.fromarray(tejido[f0:f0 + lado, c0:c0 + lado]).resize(img.shape[1::-1], Image.NEAREST))
    unet_tejido = con == 1
    d = 2 * (otsu & unet_tejido).sum() / (otsu.sum() + unet_tejido.sum())
    print(f"Píxeles que cambian con la superposición: {100 * (sin != con).mean():.1f} %")
    print(f"Artefacto según la U-Net: {100 * (con == 2).mean():.1f} % de la región · "
          f"Dice tejido Otsu (miniatura) vs. U-Net: {d:.2f}")


w.interact_manual(ver_region, region=w.Dropdown(options=list(regiones), description="Región"));

# %% [markdown]
# ### ✍️ Pregunta 3
# 1. ¿Se notan los **bordes de los parches** sin superposición? ¿Cuánto más cuesta evitarlos?
# 2. ¿Qué cuenta Otsu como tejido que la U-Net marca como **artefacto**?
#
# _Respuesta:_
#
# ## Paso 5 · ✍️ Decisión: configura el *pipeline*
# Un hospital escanea **200 láminas de colon por día** y quiere un control de calidad automático
# (¿hay suficiente tejido? ¿hay artefactos que obliguen a repetir el escaneo?) en un servidor solo con CPU.
#
# | Decisión | Tu elección | Por qué |
# |---|---|---|
# | Detección de tejido (canal, umbral) | | |
# | Aumento | | |
# | Tamaño de parche y % de tejido mínimo | | |
# | ¿Superposición? | | |
# | Tiempo estimado por lámina | | |
#
# ### Para llevarse
# - Una WSI se procesa **por parches**: el aumento y el tamaño del parche fijan el costo (×4 por cada duplicación del aumento).
# - Un paso **clásico** y barato (Otsu en la miniatura) ahorra la mayor parte del cómputo.
# - Reconstruir con **superposición** suaviza los bordes de los parches a un costo de cómputo mayor.
