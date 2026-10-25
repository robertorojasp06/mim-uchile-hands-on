# %% [markdown]
# # Sesión 4 · Actividad corta 1 — La trampa del *accuracy*
#
# **Duración:** 10 min · **Cómputo:** CPU · **Datos:** Holter de 24 h de 6 pacientes (ITMed, anonimizados)
#
# Tres algoritmos analizan **todos los latidos** de 24 h de Holter de 6 pacientes y marcan cada uno como
# *normal* o *ectópico* (extrasístole supraventricular **S** o ventricular **V**). Hay que elegir cuál
# comprar para el servicio de cardiología.
#
# | Paso | Qué haces | Tiempo |
# |---|---|---|
# | 1 | Miras solo el *accuracy* y votas | 3 min |
# | 2 | Ves la matriz de confusión y las demás métricas | 4 min |
# | 3 | Cambias de paciente y ves cómo cambia el *accuracy* | 3 min |

# %%
# INCLUDE: _setup.py

# %%
#@title 📦 Cargar los datos y los tres algoritmos
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as w

latidos = pd.read_parquet(obtener_datos("holter") / "features.parquet", columns=["record", "split", "code", "ectopico", "score_rf"])
latidos = latidos[latidos.split == "test"].reset_index(drop=True)
y = latidos.ectopico.to_numpy()

ALGORITMOS = {
    "A": np.zeros(len(y), dtype=int),                    # siempre "normal"
    "B": (latidos.score_rf >= 0.30).to_numpy().astype(int),
    "C": (latidos.score_rf >= 0.97).to_numpy().astype(int),
}


def metricas(y, p):
    vp, fp = int(((p == 1) & (y == 1)).sum()), int(((p == 1) & (y == 0)).sum())
    fn, vn = int(((p == 0) & (y == 1)).sum()), int(((p == 0) & (y == 0)).sum())
    div = lambda a, b: a / b if b else float("nan")
    sens, espec = div(vp, vp + fn), div(vn, vn + fp)
    vpp = div(vp, vp + fp)
    return {"Accuracy": div(vp + vn, len(y)), "Sensibilidad": sens, "Especificidad": espec,
            "VPP (precisión)": vpp, "F1": div(2 * vp, 2 * vp + fp + fn),
            "Accuracy balanceada": np.nanmean([sens, espec]),
            "VP": vp, "FP": fp, "FN": fn, "VN": vn}


print(f"{len(y):,} latidos · {y.sum():,} ectópicos ({100 * y.mean():.1f} %)".replace(",", "."))

# %% [markdown]
# ## Paso 1 · Solo con el *accuracy*
#
# *Accuracy* = proporción de latidos bien clasificados.

# %%
#@title 📊 Accuracy de cada algoritmo
tabla = pd.DataFrame({k: metricas(y, p) for k, p in ALGORITMOS.items()}).T
display(tabla[["Accuracy"]].style.format("{:.2%}"))

# %% [markdown]
# ### ✍️ Decisión 1
# ¿Qué algoritmo comprarías? Vota **antes** de seguir y escribe por qué en una línea.
#
# _Mi elección:_

# %% [markdown]
# ## Paso 2 · La matriz de confusión
#
# - **Sensibilidad** = VP / (VP + FN): de los ectópicos reales, ¿cuántos detecta?
# - **Especificidad** = VN / (VN + FP): de los normales, ¿cuántos deja tranquilos?
# - **VPP** = VP / (VP + FP): cuando el algoritmo marca un ectópico, ¿cuántas veces acierta?

# %%
#@title 🔎 Revelar todas las métricas
fig, ejes = plt.subplots(1, 3, figsize=(13, 3.6))
for eje, (k, p) in zip(ejes, ALGORITMOS.items()):
    m = metricas(y, p)
    cm = np.array([[m["VN"], m["FP"]], [m["FN"], m["VP"]]])
    eje.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            eje.text(j, i, f"{cm[i, j]:,}".replace(",", "."), ha="center", va="center",
                     color="white" if cm[i, j] / cm[i].sum() > 0.5 else "black", fontsize=12)
    eje.set_xticks([0, 1], ["normal", "ectópico"]); eje.set_yticks([0, 1], ["normal", "ectópico"])
    eje.set_xlabel("predicción"); eje.set_ylabel("realidad")
    eje.set_title(f"Algoritmo {k} · accuracy {m['Accuracy']:.1%}")
plt.tight_layout(); plt.show()

display(tabla.drop(columns=["VP", "FP", "FN", "VN"]).style.format("{:.1%}", na_rep="— (no marca nada)"))

# %% [markdown]
# ### ✍️ Decisión 2
# 1. ¿Qué hace realmente el algoritmo **A**? ¿Por qué su *accuracy* es tan alta?
# 2. El algoritmo **C** tiene el mejor *accuracy*. ¿Cuántos ectópicos se le escapan?
# 3. ¿Cambias tu voto? ¿Qué métricas pedirías siempre a un proveedor?
#
# _Respuesta:_

# %% [markdown]
# ## Paso 3 · El *accuracy* depende de la prevalencia
#
# Elige un paciente. En un Holter **normal** casi no hay ectópicos: ¿qué pasa con el *accuracy* de A?

# %%
#@title 🧑‍⚕️ Métricas por paciente
pacientes = ["todos"] + sorted(latidos.record.unique())


def por_paciente(paciente):
    sel = np.ones(len(y), bool) if paciente == "todos" else (latidos.record == paciente).to_numpy()
    print(f"{paciente}: {sel.sum():,} latidos · prevalencia de ectópicos {100 * y[sel].mean():.2f} %".replace(",", "."))
    t = pd.DataFrame({k: metricas(y[sel], p[sel]) for k, p in ALGORITMOS.items()}).T
    t = t[["Accuracy", "Sensibilidad", "VPP (precisión)", "FN", "FP"]].astype({"FN": int, "FP": int})
    display(t.rename(columns={"FN": "ectópicos perdidos", "FP": "falsas alarmas"})
            .style.format({"Accuracy": "{:.2%}", "Sensibilidad": "{:.1%}", "VPP (precisión)": "{:.1%}"}, na_rep="—"))


w.interact(por_paciente, paciente=w.Dropdown(options=pacientes, description="Paciente"));

# %% [markdown]
# ### Para llevarse
# - Con clases desbalanceadas, el *accuracy* premia al que **nunca** detecta nada.
# - Reporta siempre **sensibilidad, especificidad y VPP**, junto con la **prevalencia** del conjunto de prueba.
# - El mismo algoritmo puede verse excelente o inútil según **en qué pacientes** se evalúe.
