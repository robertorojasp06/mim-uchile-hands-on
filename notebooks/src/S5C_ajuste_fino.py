# %% [markdown]
# # Sesión 5 · Actividad C — Adaptar un modelo a datos locales: ¿cuántos parches hay que anotar?
#
# **Duración:** 25 min · **Cómputo:** CPU (≈1–2 min por entrenamiento; con GPU, segundos)
# · **Datos:** parches H&E de mucosa **gástrica** (proyecto DeepPath, anonimizados)
#
# En la Sesión 4 vimos que la U-Net entrenada con **colon** empeora en **estómago**, sobre todo con los artefactos.
# Hay dos caminos para tener un modelo gástrico:
#
# - **Ajuste fino** (*fine-tuning*): partir del modelo de colon y seguir entrenándolo con algunos parches gástricos.
# - **Desde cero**: entrenar una U-Net nueva solo con parches gástricos.
#
# Cada parche anotado cuesta tiempo de un patólogo. La pregunta es **cuántos** hacen falta.
#
# | Paso | Qué haces | Tiempo |
# |---|---|---|
# | 1 | Mides el punto de partida: el modelo de colon en estómago | 3 min |
# | 2 | Entrenas con N parches gástricos (ajuste fino o desde cero) | 12 min |
# | 3 | Comparas con las curvas de referencia | 5 min |
# | 4 | Decides cuántos parches anotar | 5 min |

# %%
# INCLUDE: _setup.py

# %%
#@title 📦 Cargar los parches gástricos y el modelo de colon
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython.display import clear_output, display
from matplotlib.colors import ListedColormap

RAIZ = obtener_datos("ajuste")
datos = np.load(RAIZ / "datos.npz")
curvas = pd.read_csv(RAIZ / "curvas.csv")
DISPOSITIVO = "cuda" if torch.cuda.is_available() else "cpu"
CMAP = ListedColormap(["#eeeeee", "#c2185b", "#1976d2"])
x_ent = torch.tensor(datos["train_x"]).permute(0, 3, 1, 2).float() / 255
y_ent = torch.tensor(datos["train_y"]).long()
x_pru = torch.tensor(datos["test_x"]).permute(0, 3, 1, 2).float() / 255
y_pru = datos["test_y"]
region_pru = datos["test_region"]


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


def modelo_colon():
    red = UNet300()
    red.load_state_dict(torch.load(RAIZ / "unet_colon_20x.pt", map_location="cpu"))
    return red


def predecir(red):
    red.eval().to(DISPOSITIVO)
    with torch.inference_mode():
        return torch.cat([red(x_pru[i:i + 16].to(DISPOSITIVO)).argmax(1).cpu() for i in range(0, len(x_pru), 16)]).numpy()


def dice(pred, ref, c):
    a, b = pred == c, ref == c
    return 2 * (a & b).sum() / (a.sum() + b.sum())


print(f"Parches para entrenar: {len(x_ent)} (6 por región) · parches de prueba: {len(x_pru)} · dispositivo: {DISPOSITIVO}")

# %% [markdown]
# ## Paso 1 · El punto de partida

# %%
#@title 📏 Modelo de colon en estómago
pred_base = predecir(modelo_colon())
print(f"Dice tejido = {dice(pred_base, y_pru, 1):.3f} · Dice artefacto = {dice(pred_base, y_pru, 2):.3f}")
TEJIDOS = {"mucosa", "submucosa", "muscular", "adiposo", "bordes_tejido"}
CLASE_DE = lambda r: 1 if r in TEJIDOS else (0 if r == "fondo" else 2)
por_region = pd.DataFrame([{"región": r, "clase evaluada": ["fondo", "tejido", "artefacto"][CLASE_DE(r)],
                            "Dice": dice(pred_base[region_pru == r], y_pru[region_pru == r], CLASE_DE(r))}
                           for r in np.unique(region_pru)]).sort_values("Dice")
display(por_region.style.format({"Dice": "{:.2f}"}).hide(axis="index"))

# %% [markdown]
# ## Paso 2 · Entrenar con N parches gástricos
#
# Los parches se toman **en orden fijo**, balanceados por región: con N = 12 hay uno de cada región
# (mucosa, adiposo, burbujas, tinta, pliegues…). Así, todos los que elijan el mismo N usan los mismos parches.
#
# - **Estrategia**: *ajuste fino* (parte del modelo de colon) o *desde cero* (pesos aleatorios).
# - **Congelar el codificador**: en el ajuste fino, solo se entrenan las capas del decodificador.
#
# Durante el entrenamiento se aplican **giros y espejos** aleatorios (aumento de datos).
# Cada resultado se agrega a la bitácora.

# %%
#@title 🚀 Entrenar
n_parches = 16  #@param {type:"slider", min:0, max:72, step:4}
estrategia = "ajuste fino"  #@param ["ajuste fino", "desde cero"]
epocas = 8  #@param {type:"slider", min:1, max:20, step:1}
tasa_aprendizaje = 3e-4  #@param {type:"number"}
congelar_codificador = False  #@param {type:"boolean"}
semilla = 0  #@param {type:"integer"}

if "bitacora" not in globals():
    bitacora = []
torch.manual_seed(semilla)
rng = np.random.default_rng(semilla)
red = modelo_colon() if estrategia == "ajuste fino" else UNet300()
if congelar_codificador and estrategia == "ajuste fino":
    for capa in (red.conv1, red.conv2, red.conv3, red.conv4):
        capa.requires_grad_(False)
red.to(DISPOSITIVO)
entrenables = [p for p in red.parameters() if p.requires_grad]
historia, t0 = [], time.time()
if n_parches:
    optimizador = torch.optim.Adam(entrenables, lr=tasa_aprendizaje)
    for epoca in range(1, epocas + 1):
        red.train()
        orden, suma = rng.permutation(n_parches), 0.0
        for i in range(0, n_parches, 4):
            b = orden[i:i + 4]
            xb, yb = x_ent[b], y_ent[b]
            k = int(rng.integers(4))
            xb, yb = torch.rot90(xb, k, (2, 3)), torch.rot90(yb, k, (1, 2))
            if rng.random() < 0.5:
                xb, yb = xb.flip(3), yb.flip(2)
            optimizador.zero_grad()
            perdida = F.cross_entropy(red(xb.to(DISPOSITIVO)), yb.to(DISPOSITIVO))
            perdida.backward()
            optimizador.step()
            suma += perdida.item() * len(b)
        historia.append(suma / n_parches)
        clear_output(wait=True)
        plt.figure(figsize=(6, 3)); plt.plot(range(1, epoca + 1), historia, "o-")
        plt.xlabel("época"); plt.ylabel("costo de entrenamiento"); plt.grid(alpha=0.3)
        plt.title(f"Época {epoca}/{epocas} · {time.time() - t0:.0f} s"); plt.show()

pred = predecir(red)
resultado = {"N parches": n_parches, "estrategia": estrategia, "épocas": epocas, "tasa": tasa_aprendizaje,
             "codificador congelado": congelar_codificador and estrategia == "ajuste fino", "semilla": semilla,
             "Dice tejido": dice(pred, y_pru, 1), "Dice artefacto": dice(pred, y_pru, 2),
             "segundos": round(time.time() - t0)}
bitacora.append(resultado)
print(f"Dice tejido = {resultado['Dice tejido']:.3f} · Dice artefacto = {resultado['Dice artefacto']:.3f} "
      f"(partida: {dice(pred_base, y_pru, 1):.3f} / {dice(pred_base, y_pru, 2):.3f})")
display(pd.DataFrame(bitacora).style.format(precision=3).format({"tasa": "{:g}"}).hide(axis="index"))

# %%
#@title 🔍 Ver predicciones antes y después
ejemplos = [np.flatnonzero(region_pru == r)[0] for r in ("burbujas_aire", "desenfoque", "sombras", "mucosa")]
fig, ejes = plt.subplots(4, 4, figsize=(11, 11))
for fila, i in enumerate(ejemplos):
    for col, (img, titulo) in enumerate(((datos["test_x"][i], region_pru[i]), (y_pru[i], "anotación"),
                                         (pred_base[i], "modelo de colon"), (pred[i], "tu modelo"))):
        ejes[fila, col].imshow(img, **({} if col == 0 else dict(cmap=CMAP, vmin=0, vmax=2, interpolation="nearest")))
        ejes[fila, col].set_title(titulo, fontsize=10); ejes[fila, col].axis("off")
plt.tight_layout(); plt.show()

# %% [markdown]
# **Sugerencias** (cada una toma 1–2 min en CPU): N = 0, 4, 16 y 48 con *ajuste fino*; luego N = 16 y 48 *desde cero*;
# luego N = 16 con el codificador congelado. Si trabajan en grupo, repártanse los experimentos.
#
# ## Paso 3 · Comparar con las curvas de referencia
# Las líneas son el promedio de 3 semillas, entrenadas con este mismo procedimiento (8 épocas). Las estrellas son tus experimentos.

# %%
#@title 📈 Curvas de aprendizaje
fig, ejes = plt.subplots(1, 2, figsize=(12, 4))
colores = {"ajuste fino": "#2ca02c", "desde cero": "#9467bd"}
for eje, metrica, col in zip(ejes, ("Dice tejido", "Dice artefacto"), ("dice_tejido", "dice_artefacto")):
    for est, g in curvas.groupby("estrategia"):
        m = g.groupby("n_parches")[col].agg(["mean", "min", "max"])
        eje.plot(m.index, m["mean"], "o-", color=colores[est], label=est)
        eje.fill_between(m.index, m["min"], m["max"], color=colores[est], alpha=0.15)
    for r in bitacora:
        eje.scatter(r["N parches"], r[metrica], marker="*", s=180, color=colores[r["estrategia"]], edgecolor="k", zorder=3)
    eje.set_xlabel("parches gástricos anotados (N)"); eje.set_ylabel(metrica); eje.set_ylim(0, 1); eje.grid(alpha=0.3)
eje.legend(); plt.tight_layout(); plt.show()

# %% [markdown]
# ### ✍️ Pregunta
# 1. ¿Con cuántos parches el ajuste fino deja de mejorar claramente?
# 2. ¿Cuántos parches necesita el entrenamiento *desde cero* para alcanzar al ajuste fino? ¿Lo alcanza?
#    ¿Es justa la comparación con solo 8 épocas? Prueba *desde cero* con N = 64 y 20 épocas.
# 3. ¿Por qué el ajuste fino con **pocos** parches puede empeorar alguna clase?
#
# _Respuesta:_
#
# ## Paso 4 · ✍️ Decisión
# Anotar un parche gástrico toma ~**4 minutos** a un patólogo. Tienes **2 horas** de patólogo.
# - ¿Cuántos parches pides anotar y con qué estrategia entrenas?
# - ¿Qué regiones pedirías priorizar? (Mira el Dice por región del Paso 1.)
# - ¿Con qué datos validarías el modelo final antes de usarlo?
#
# _Respuesta:_
#
# ### Para llevarse
# - El **ajuste fino** reutiliza lo aprendido: con pocos ejemplos locales se recupera buena parte del desempeño.
# - **Desde cero**, el mismo número de ejemplos rinde mucho menos: el *deep learning* necesita muchos datos.
# - Elegir **qué** anotar (regiones donde el modelo falla) importa tanto como **cuánto** anotar.
