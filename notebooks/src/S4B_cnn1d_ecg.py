# %% [markdown]
# # Sesión 4 · Actividad B — Entrena tu primera red convolucional 1D con latidos de Holter
#
# **Duración:** 25–30 min · **Cómputo:** CPU (≈1–2 min por entrenamiento; si hay GPU, se usa sola)
# · **Datos:** latidos de Holter de 24 pacientes (ITMed, anonimizados)
#
# Una **CNN 1D** desliza filtros a lo largo del tiempo, igual que una CNN 2D los desliza sobre una imagen.
# Aquí aprende, **a partir de la señal cruda**, a clasificar cada latido como
# **N** (normal), **S** (extrasístole supraventricular) o **V** (extrasístole ventricular).
# Nadie le dice qué mirar: las características las extrae la red.
#
# Los 24 pacientes están separados **por paciente**: 14 para entrenar, 4 para validar y 6 para la prueba final.
#
# | Paso | Qué haces | Tiempo |
# |---|---|---|
# | 1 | Miras los latidos | 3 min |
# | 2 | Conoces la red | 2 min |
# | 3 | Entrenas con los valores por defecto | 4 min |
# | 4 | Retos: tasa de aprendizaje, sobreajuste, fuga de datos | 15 min |
# | 5 | Revisas los errores | 3 min |

# %%
# INCLUDE: _setup.py

# %%
#@title 📦 Cargar los latidos
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from IPython.display import clear_output, display

RAIZ = obtener_datos("holter")
datos = np.load(RAIZ / "windows.npz")
FS, PRE = int(datos["fs"]), int(datos["pre"])
senal = datos["x"].astype(np.float32)                 # mV, ventanas de 4 s (2,5 s antes y 1,5 s después del pico R)
CLASES = ["N", "S", "V"]
NOMBRE = {"N": "normal", "S": "supraventricular", "V": "ventricular"}
COLOR = {"N": "#7f7f7f", "S": "#1f77b4", "V": "#d62728"}
etiqueta = pd.Series(datos["y"]).map({c: i for i, c in enumerate(CLASES)}).to_numpy()
paciente = datos["record"]
pacientes = pd.read_csv(RAIZ / "records.csv").set_index("record")
grupo = pacientes.loc[paciente, "split"].to_numpy()
DISPOSITIVO = "cuda" if torch.cuda.is_available() else "cpu"

conteo = pd.crosstab(pd.Series(grupo, name="conjunto"), pd.Series(datos["y"], name="clase"))
display(conteo.loc[["train", "val", "test"]].rename(index={"train": "entrenamiento", "val": "validación", "test": "prueba"}))
print(f"Dispositivo: {DISPOSITIVO}")

# %% [markdown]
# ## Paso 1 · Mira los latidos
# Cada ejemplo es una ventana de ECG centrada en el latido (línea punteada).

# %%
#@title 👀 Ejemplos de cada clase
rng = np.random.default_rng(1)
t = (np.arange(senal.shape[1]) - PRE) / FS
fig, ejes = plt.subplots(3, 4, figsize=(14, 6.5), sharex=True)
for fila, c in enumerate(CLASES):
    for eje, i in zip(ejes[fila], rng.choice(np.flatnonzero(etiqueta == fila), 4, replace=False)):
        eje.plot(t, senal[i], color=COLOR[c], lw=0.9)
        eje.axvline(0, color="k", ls=":", lw=0.8)
        eje.set_title(f"{c} · {NOMBRE[c]} · {paciente[i]}", fontsize=9)
        eje.set_yticks([])
for eje in ejes[-1]:
    eje.set_xlabel("tiempo desde el pico R (s)")
plt.tight_layout(); plt.show()

# %% [markdown]
# ### ✍️ Pregunta 1
# ¿Qué distingue a un latido **V** de uno normal? ¿Y a uno **S**? ¿Se ve en la forma del latido o en el **momento** en que aparece?
#
# _Respuesta:_
#
# ## Paso 2 · La red
# Tres bloques **convolución → normalización → ReLU → *pooling*** y una capa final que entrega
# una probabilidad para N, S y V. La **función de costo** es la entropía cruzada y el **optimizador**, Adam.

# %%
#@title 🧠 Definir la red y el entrenamiento


def crear_red():
    bloque = lambda entrada, salida: [nn.Conv1d(entrada, salida, 7, padding=3), nn.BatchNorm1d(salida),
                                      nn.ReLU(), nn.MaxPool1d(2)]
    return nn.Sequential(*bloque(1, 16), *bloque(16, 32), *bloque(32, 64),
                         nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(64, 3))


print(crear_red())
print(f"Parámetros entrenables: {sum(p.numel() for p in crear_red().parameters()):,}".replace(",", "."))


VENTANAS = {  # segundos antes y después del pico R
    "1,5 s (el latido y el anterior)": (1.0, 0.5),
    "1 s (solo el latido)": (0.5, 0.5),
    "4 s (el latido y su contexto)": (PRE / FS, (senal.shape[1] - PRE) / FS),
}


def preparar(ventana):
    """Recorta la ventana y normaliza la amplitud de cada latido."""
    antes, despues = VENTANAS[ventana]
    x = senal[:, PRE - int(antes * FS): PRE + int(despues * FS)]
    x = x - np.median(x, axis=1, keepdims=True)
    x = x / (np.percentile(np.abs(x), 99, axis=1, keepdims=True) + 1e-3)
    return torch.tensor(x[:, None], dtype=torch.float32)


def separar(modo, fraccion, rng):
    ent, val, pru = (np.flatnonzero(grupo == g) for g in ("train", "val", "test"))
    if modo.startswith("por latido"):        # mezcla latidos de todos los pacientes: ¡fuga de datos!
        todos = rng.permutation(len(etiqueta))
        ent, val, pru = todos[:len(ent)], todos[len(ent):len(ent) + len(val)], todos[len(ent) + len(val):]
    if fraccion < 1:
        ent = np.concatenate([rng.choice(ent[etiqueta[ent] == c], max(1, int(fraccion * (etiqueta[ent] == c).sum())),
                                         replace=False) for c in range(3)])
    return ent, val, pru


def sensibilidades(pred, real):
    return np.array([(pred[real == c] == c).mean() for c in range(3)])


def predecir(red, x):
    red.eval()
    with torch.no_grad():
        return torch.cat([red(x[i:i + 2048].to(DISPOSITIVO)).argmax(1).cpu() for i in range(0, len(x), 2048)]).numpy()


bitacora = []

# %% [markdown]
# ## Paso 3 · Entrenar
#
# - **Tasa de aprendizaje**: el largo de cada paso del optimizador.
# - **Épocas**: cuántas veces la red recorre todo el conjunto de entrenamiento.
# - **Fracción de entrenamiento**: qué parte de los datos de entrenamiento se usa.
# - **Separación**: *por paciente* (correcta) o *por latido*, mezclando latidos de un mismo paciente entre entrenamiento y prueba.
# - **Balancear clases**: da más peso en el costo a las clases poco frecuentes (S y V).
#
# Ejecuta primero con los valores por defecto. Se guarda la red de la **época con mejor validación**.

# %%
#@title 🚀 Entrenar la red
tasa_aprendizaje = "1e-3"  #@param ["1e-5", "1e-4", "1e-3", "1e-2", "1e-1"]
epocas = 15  #@param {type:"slider", min:1, max:40, step:1}
fraccion_entrenamiento = 1.0  #@param {type:"slider", min:0.02, max:1.0, step:0.02}
separacion = "por paciente"  #@param ["por paciente", "por latido (mezclando pacientes)"]
ventana = "1,5 s (el latido y el anterior)"  #@param ["1,5 s (el latido y el anterior)", "1 s (solo el latido)", "4 s (el latido y su contexto)"]
balancear_clases = True  #@param {type:"boolean"}
semilla = 0  #@param {type:"integer"}

torch.manual_seed(semilla)
rng = np.random.default_rng(semilla)
x = preparar(ventana)
y = torch.tensor(etiqueta)
ent, val, pru = separar(separacion, fraccion_entrenamiento, rng)
pesos = len(ent) / (3 * np.bincount(etiqueta[ent], minlength=3)) if balancear_clases else np.ones(3)
costo = nn.CrossEntropyLoss(weight=torch.tensor(pesos, dtype=torch.float32).to(DISPOSITIVO))
red = crear_red().to(DISPOSITIVO)
optimizador = torch.optim.Adam(red.parameters(), lr=float(tasa_aprendizaje))

historia = {"costo_ent": [], "costo_val": [], "acc_bal_val": []}
mejor = (-1.0, None, 0)
t0 = time.time()
for epoca in range(1, epocas + 1):
    red.train()
    orden = rng.permutation(ent)
    suma = 0.0
    for i in range(0, len(orden), 128):
        lote = orden[i:i + 128]
        optimizador.zero_grad()
        perdida = costo(red(x[lote].to(DISPOSITIVO)), y[lote].to(DISPOSITIVO))
        perdida.backward()
        optimizador.step()
        suma += perdida.item() * len(lote)
    red.eval()
    with torch.no_grad():
        salida = torch.cat([red(x[val[i:i + 2048]].to(DISPOSITIVO)).cpu() for i in range(0, len(val), 2048)])
    costo_val = nn.functional.cross_entropy(salida, y[val], weight=torch.tensor(pesos, dtype=torch.float32)).item()
    acc_bal = sensibilidades(salida.argmax(1).numpy(), etiqueta[val]).mean()
    historia["costo_ent"].append(suma / len(ent)); historia["costo_val"].append(costo_val)
    historia["acc_bal_val"].append(acc_bal)
    if acc_bal > mejor[0]:
        mejor = (acc_bal, {k: v.detach().clone() for k, v in red.state_dict().items()}, epoca)

    clear_output(wait=True)
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 3.5))
    ep = range(1, epoca + 1)
    a.plot(ep, historia["costo_ent"], "o-", label="entrenamiento"); a.plot(ep, historia["costo_val"], "o-", label="validación")
    a.set_xlabel("época"); a.set_ylabel("costo (entropía cruzada)"); a.legend(); a.grid(alpha=0.3)
    b.plot(ep, historia["acc_bal_val"], "o-", color="#2ca02c"); b.set_ylim(0, 1)
    b.axvline(mejor[2], color="k", ls=":", lw=1)
    b.set_xlabel("época"); b.set_ylabel("accuracy balanceada (validación)"); b.grid(alpha=0.3)
    plt.suptitle(f"Época {epoca}/{epocas} · {time.time() - t0:.0f} s · mejor época: {mejor[2]}")
    plt.tight_layout(); plt.show()

red.load_state_dict(mejor[1])
pred = predecir(red, x[pru])
sens = sensibilidades(pred, etiqueta[pru])
bitacora.append({"tasa": tasa_aprendizaje, "épocas": epocas, "fracción": fraccion_entrenamiento,
                 "separación": separacion.split(" (")[0], "ventana": ventana.split(" (")[0],
                 "balanceo": balancear_clases, "semilla": semilla, "mejor época": mejor[2],
                 "acc. bal. validación": mejor[0], "sens. N": sens[0], "sens. S": sens[1], "sens. V": sens[2],
                 "acc. bal. prueba": sens.mean()})

matriz = pd.crosstab(pd.Series(etiqueta[pru], name="real").map(dict(enumerate(CLASES))),
                     pd.Series(pred, name="predicción").map(dict(enumerate(CLASES)))).reindex(index=CLASES, columns=CLASES, fill_value=0)
print("Prueba (pacientes que la red no vio)" if separacion == "por paciente" else "Prueba (¡latidos de pacientes que la red sí vio!)")
display(matriz)
print("📒 Bitácora de experimentos")
display(pd.DataFrame(bitacora).style.format(precision=2).background_gradient(subset=["acc. bal. prueba"], cmap="Greens", vmin=0.3, vmax=1))

# %% [markdown]
# ### ✍️ Pregunta 2
# ¿Qué clase reconoce peor la red? Mira la matriz de confusión: ¿con qué la confunde?
#
# _Respuesta:_
#
# ## Paso 4 · Retos
# Cambia los parámetros de la celda **🚀 Entrenar la red** y vuelve a ejecutarla. Cada resultado queda en la bitácora.
#
# **Reto 1 · Tasa de aprendizaje** (5 min). Entrena con `1e-5`, `1e-3` y `1e-1` (15 épocas).
# Mira las **curvas**, no solo el resultado final: ¿con cuál el costo baja tan lento que no alcanza a aprender?
# ¿Con cuál la validación salta de una época a otra? Si con `1e-1` la prueba sale bien, ¿confiarías en ese
# resultado? (Pista: se guarda la mejor época de validación… ¿y si fue suerte?)
#
# _Respuesta:_
#
# **Reto 2 · Sobreajuste** (5 min). Usa `fraccion_entrenamiento = 0.04` y `epocas = 40`.
# ¿Qué pasa con el costo de entrenamiento y el de validación? ¿En qué época conviene detenerse?
#
# _Respuesta:_
#
# **Reto 3 · Fuga de datos** (5 min). Vuelve a los valores por defecto y cambia la separación a
# *por latido (mezclando pacientes)*. ¿Cuánto "mejora" la prueba? ¿Cuál de los dos resultados le
# informarías a un hospital que quiere usar la red con **pacientes nuevos**?
#
# _Respuesta:_
#
# **Reto 4 · ¿Qué ve la red?** (si te sobra tiempo). Prueba la ventana de **1 s**, que no alcanza a mostrar el
# latido anterior, y la de **4 s**. ¿Qué pasa con la sensibilidad para **S**? Cambia también la semilla:
# ¿cuánto varía el resultado solo por azar?
#
# _Respuesta:_
#
# **Para discutir.** En la actividad C, un *random forest* con 7 características diseñadas a mano
# (prematuridad, ancho del QRS…) detecta S y V mucho mejor que esta red. ¿Por qué? Piensa en cuántos
# **pacientes** tenemos para entrenar y en la diapositiva "El deep learning necesita muchos datos".
#
# ## Paso 5 · Revisa los errores
# Latidos de **prueba** que la última red clasificó mal.

# %%
#@title 🔍 Errores de la última red
errores = pru[pred != etiqueta[pru]]
muestra = np.random.default_rng(0).choice(errores, min(8, len(errores)), replace=False)
xv = x[:, 0].numpy()
tv = np.arange(xv.shape[1]) / FS - VENTANAS[ventana][0]
fig, ejes = plt.subplots(2, 4, figsize=(14, 5), sharex=True)
for eje, i in zip(ejes.flat, muestra):
    eje.plot(tv, xv[i], color=COLOR[CLASES[etiqueta[i]]], lw=0.9)
    eje.axvline(0, color="k", ls=":", lw=0.8)
    eje.set_title(f"real {CLASES[etiqueta[i]]} → red {CLASES[pred[pru == i][0]]} · {paciente[i]}", fontsize=9)
    eje.set_yticks([])
plt.tight_layout(); plt.show()

# %% [markdown]
# ### ✍️ Pregunta 3
# ¿Algunos de estos errores te parecen razonables? ¿Podría haber errores en las **etiquetas**?
#
# _Respuesta:_
#
# ### Para llevarse
# - La red aprende sus propias características; la **tasa de aprendizaje** y las **épocas** deciden si llega a un buen mínimo.
# - Con pocos datos, la red **memoriza** (sobreajuste): la curva de validación avisa cuándo detenerse.
# - Si la prueba comparte pacientes con el entrenamiento, el resultado es **optimista**: separa siempre **por paciente**.
