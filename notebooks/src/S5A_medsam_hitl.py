# %% [markdown]
# # Sesión 5 · Actividad A — MedSAM con un humano en el ciclo (*human-in-the-loop*)
#
# **Duración:** 25–30 min · **Cómputo:** CPU (≈0,3 s por corte) · **Datos:** CT del HCUCH (anonimizadas), segmentadas por **dos radiólogos**
#
# **MedSAM** es un **modelo fundacional** de segmentación: recibe una imagen y un *prompt* (una **caja**
# alrededor de la estructura) y devuelve su contorno, sin haber sido entrenado para esta tarea en particular.
#
# MedSAM tiene dos partes:
# - un **codificador de imagen** pesado (ViT, ~18 s por corte en CPU), que **ya ejecutamos** por ustedes;
# - un **decodificador** liviano que convierte la caja en una máscara (~0,3 s en CPU), que corre aquí.
#
# Compararemos MedSAM con dos radiólogos y simularemos un flujo de anotación donde el experto **solo corrige**.
#
# | Paso | Qué haces | Tiempo |
# |---|---|---|
# | 1 | Dibujas cajas en un corte y comparas con los expertos | 8 min |
# | 2 | Anotas un volumen completo: MedSAM propone, el experto corrige | 12 min |
# | 3 | Decides si usarías este flujo y cómo lo validarías | 5 min |

# %%
# INCLUDE: _setup.py

# %%
#@title 📦 Cargar MedSAM (decodificador) y las CT
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as w
import torch
import torch.nn.functional as F
from matplotlib.patches import Rectangle

RAIZ = obtener_datos("medsam")
decodificador = torch.jit.load(str(RAIZ / "decoder.pt"), map_location="cpu").eval()
NOMBRES = {"masa_pulmonar": "Masa pulmonar", "lesion_hepatica": "Lesión hepática"}
casos = {k: dict(np.load(RAIZ / f"{k}.npz")) for k in NOMBRES}
for k, c in casos.items():
    print(f"{NOMBRES[k]}: {len(c['z'])} cortes de {c['ct'].shape[1]} × {c['ct'].shape[2]} px "
          f"({c['spacing'][0]:.1f} mm/px, cortes cada {c['spacing'][2]:.0f} mm)")


def medsam(caso, i, caja):
    """caja = (x0, y0, x1, y1) en píxeles del recorte -> máscara booleana."""
    c = casos[caso]
    lado = c["ct"].shape[1]
    emb = torch.tensor(c["emb"][i:i + 1].astype(np.float32))
    box = torch.tensor([caja], dtype=torch.float32) * 1024 / lado
    with torch.no_grad():
        logits = decodificador(emb, box)
    return F.interpolate(logits[None], size=(lado, lado), mode="bilinear", align_corners=False)[0, 0].numpy() > 0


def dice(a, b):
    s = a.sum() + b.sum()
    return 2 * (a & b).sum() / s if s else np.nan


def caja_de(mascara, margen=2):
    f, c = np.nonzero(mascara)
    return (c.min() - margen, f.min() - margen, c.max() + margen, f.max() + margen)


def ventana(hu):
    return np.clip((hu + 160) / 400, 0, 1)

# %% [markdown]
# ## Paso 1 · Una caja, una máscara
# Mueve los límites de la caja (en píxeles). <span style="color:#fbc02d">**Amarillo**</span>: MedSAM ·
# <span style="color:#e53935">**rojo**</span>: Experto 1 · <span style="color:#17becf">**celeste**</span>: Experto 2.
#
# 🎯 Prueba: (a) una caja ajustada; (b) una caja holgada; (c) una caja que corta la lesión;
# (d) los primeros y últimos cortes, donde la lesión es pequeña.

# %%
#@title ✏️ Dibujar la caja


def un_corte(caso, corte, x, y):
    c = casos[caso]
    i = min(corte, len(c["z"]) - 1)
    caja = (x[0], y[0], x[1], y[1])
    t0 = time.time()
    m = medsam(caso, i, caja)
    dt = time.time() - t0
    e1, e2 = c["exp1"][i].astype(bool), c["exp2"][i].astype(bool)
    fig, ejes = plt.subplots(1, 2, figsize=(11, 5.2))
    for eje in ejes:
        eje.imshow(ventana(c["ct"][i]), cmap="gray"); eje.axis("off")
    ejes[0].add_patch(Rectangle((caja[0], caja[1]), caja[2] - caja[0], caja[3] - caja[1],
                                fill=False, ec="#00e676", lw=2))
    ejes[0].set_title("Tu caja (prompt)")
    ejes[1].imshow(np.ma.masked_where(~m, m), cmap="autumn_r", alpha=0.45, vmin=0, vmax=1)
    if e1.any():
        ejes[1].contour(e1, [0.5], colors="#e53935", linewidths=1.3)
    if e2.any():
        ejes[1].contour(e2, [0.5], colors="#17becf", linewidths=1.3)
    ejes[1].set_title("MedSAM vs. expertos")
    plt.suptitle(f"{NOMBRES[caso]} · corte {i + 1}/{len(c['z'])}"); plt.tight_layout(); plt.show()
    print(f"Dice MedSAM–Experto 1 = {dice(m, e1):.2f} · MedSAM–Experto 2 = {dice(m, e2):.2f} · "
          f"Experto 1–Experto 2 = {dice(e1, e2):.2f}   (decodificador: {dt:.2f} s)")


selector_caso = w.Dropdown(options=[(v, k) for k, v in NOMBRES.items()], description="Estructura")
corte = w.IntSlider(value=13, min=0, max=26, description="Corte", continuous_update=False)
caja_x = w.IntRangeSlider(description="Caja x", continuous_update=False)
caja_y = w.IntRangeSlider(description="Caja y", continuous_update=False)


def reiniciar_caja(_=None):
    c = casos[selector_caso.value]
    lado = c["ct"].shape[1]
    corte.max = len(c["z"]) - 1
    corte.value = min(corte.value, corte.max)
    x0, y0, x1, y1 = caja_de(c["exp1"][corte.value].astype(bool) | c["exp2"][corte.value].astype(bool), 3)
    for s, (a, b) in ((caja_x, (x0, x1)), (caja_y, (y0, y1))):
        s.min, s.max = 0, lado - 1
        s.value = (max(a, 0), min(b, lado - 1))


selector_caso.observe(reiniciar_caja, names="value")
reiniciar_caja()
w.interact(un_corte, caso=selector_caso, corte=corte, x=caja_x, y=caja_y);

# %% [markdown]
# ### ✍️ Pregunta 1
# 1. ¿Cuánto cambia el Dice entre una caja ajustada y una holgada?
# 2. En los cortes centrales, ¿MedSAM queda **dentro** de la variabilidad entre los dos expertos? ¿Y en los extremos?
#
# _Respuesta:_
#
# ## Paso 2 · Anotar un volumen: MedSAM propone, el experto corrige
#
# Simulamos el flujo *human-in-the-loop* corte a corte:
# 1. El experto dibuja **una caja** en el corte inicial.
# 2. En el corte vecino, la caja se **propaga**: es el rectángulo de la máscara anterior más un margen.
# 3. El experto revisa: si la máscara es buena (Dice con su propia anotación ≥ **umbral de aceptación**), **acepta**.
#    Si no, **dibuja una caja nueva**; si aún falla, **segmenta a mano** ese corte.
#
# El **Experto 1** hace de anotador. El **Experto 2** no participa: sirve para evaluar el resultado de forma independiente.

# %%
#@title 🔁 Simular la anotación asistida
estructura = "masa_pulmonar"  #@param ["masa_pulmonar", "lesion_hepatica"]
margen_px = 3  #@param {type:"slider", min:0, max:10, step:1}
umbral_aceptacion = 0.8  #@param {type:"slider", min:0.5, max:0.95, step:0.05}
#@markdown Segundos que tarda el experto en cada acción:
seg_aceptar = 3  #@param {type:"integer"}
seg_caja = 8  #@param {type:"integer"}
seg_manual = 60  #@param {type:"integer"}

c = casos[estructura]
n = len(c["z"])
e1, e2 = c["exp1"].astype(bool), c["exp2"].astype(bool)
inicio = int(np.argmax(e1.sum(axis=(1, 2))))          # el corte donde la lesión es más grande
final = np.zeros_like(e1)
accion = [""] * n
dice_propuesta = np.full(n, np.nan)

for orden in (range(inicio, n), range(inicio - 1, -1, -1)):
    previa = None
    for i in orden:
        if not e1[i].any():                     # el experto 1 no ve lesión en este corte
            accion[i], previa = "sin lesión", None
            continue
        if previa is None or not previa.any():
            m, accion[i] = medsam(estructura, i, caja_de(e1[i], margen_px)), "caja"
        else:
            m, accion[i] = medsam(estructura, i, caja_de(previa, margen_px)), "aceptar"
        dice_propuesta[i] = dice(m, e1[i])
        if dice_propuesta[i] < umbral_aceptacion and accion[i] == "aceptar":
            m, accion[i] = medsam(estructura, i, caja_de(e1[i], margen_px)), "caja"
        if dice(m, e1[i]) < umbral_aceptacion:
            m, accion[i] = e1[i], "manual"
        final[i] = m
        previa = m

costo = {"aceptar": seg_aceptar, "caja": seg_caja, "manual": seg_manual, "sin lesión": 0}
t_asistido = sum(costo[a] for a in accion)
t_manual = seg_manual * sum(a != "sin lesión" for a in accion)
colores = {"aceptar": "#2ca02c", "caja": "#ff7f0e", "manual": "#d62728", "sin lesión": "#cccccc"}

fig, (a, b) = plt.subplots(2, 1, figsize=(12, 4.5), gridspec_kw={"height_ratios": [2, 1]}, sharex=True)
a.bar(range(n), np.nan_to_num(dice_propuesta), color=[colores[x] for x in accion])
a.axhline(umbral_aceptacion, color="k", ls="--", lw=1)
a.set_ylabel("Dice de la propuesta\nvs. Experto 1"); a.set_ylim(0, 1)
b.scatter(range(n), [0] * n, c=[colores[x] for x in accion], s=120, marker="s")
b.set_yticks([]); b.set_xlabel("corte"); b.axvline(inicio, color="k", lw=0.8, ls=":")
a.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=v) for v in colores.values()], labels=list(colores),
         ncol=4, loc="lower center")
plt.suptitle(f"{NOMBRES[estructura]} · corte inicial {inicio + 1}"); plt.tight_layout(); plt.show()

conteo = pd.Series(accion).value_counts()
print("Acciones del experto:", conteo.to_dict())
print(f"Tiempo estimado: asistido {t_asistido / 60:.1f} min vs. manual {t_manual / 60:.1f} min "
      f"(ahorro {100 * (1 - t_asistido / t_manual):.0f} %)")
print(f"Dice 3D del volumen final: vs. Experto 1 = {dice(final, e1):.3f} · vs. Experto 2 = {dice(final, e2):.3f} "
      f"(Experto 1 vs. Experto 2 = {dice(e1, e2):.3f})")

# %%
#@title 🖼️ Ver el volumen anotado
paso = max(1, n // 12)
idx = list(range(0, n, paso))[:12]
fig, ejes = plt.subplots(2, 6, figsize=(15, 5.5))
for eje, i in zip(ejes.flat, idx):
    eje.imshow(ventana(c["ct"][i]), cmap="gray")
    if final[i].any():
        eje.imshow(np.ma.masked_where(~final[i], final[i]), cmap="autumn_r", alpha=0.45, vmin=0, vmax=1)
    if e2[i].any():
        eje.contour(e2[i], [0.5], colors="#17becf", linewidths=1)
    eje.set_title(f"corte {i + 1} · {accion[i]}", fontsize=9, color=colores[accion[i]])
    eje.axis("off")
for eje in ejes.flat[len(idx):]:
    eje.axis("off")
plt.suptitle("Amarillo: resultado del flujo asistido · celeste: Experto 2 (evaluador independiente)")
plt.tight_layout(); plt.show()

# %% [markdown]
# ### ✍️ Pregunta 2
# 1. ¿En qué cortes tuvo que intervenir más el experto? ¿Por qué ahí?
# 2. Cambia el **umbral de aceptación** (0,7 y 0,9). ¿Qué pasa con el tiempo y con el Dice frente al Experto 2?
# 3. Haz lo mismo con la **lesión hepática**. ¿El ahorro es igual?
#
# _Respuesta:_
#
# ## Paso 3 · ✍️ Decisión
# Tu servicio debe segmentar **200 volúmenes** de masas pulmonares para entrenar un modelo propio.
# - ¿Usarías este flujo? ¿Con qué umbral de aceptación?
# - El volumen final se parece mucho al Experto 1 porque él corrigió. ¿Cómo demostrarías que la anotación es
#   **buena** y no solo **rápida**? (Pista: ¿para qué sirvió el Experto 2?)
#
# _Respuesta:_
#
# ### Para llevarse
# - Un modelo fundacional segmenta **sin entrenamiento específico**, pero su calidad depende del *prompt* y falla en los bordes de la estructura (cortes extremos, lesiones pequeñas).
# - En un flujo *human-in-the-loop* el experto **corrige** en vez de dibujar: el ahorro depende de cuántas propuestas acepta.
# - Los modelos fundacionales **abaratan anotar, no validar**: la evaluación necesita una referencia independiente.
