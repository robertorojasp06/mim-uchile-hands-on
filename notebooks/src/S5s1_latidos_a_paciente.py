# %% [markdown]
# # Sesión 5 · Actividad corta 1 — De latidos a paciente (y de parches a lámina)
#
# **Duración:** 10 min · **Cómputo:** CPU · **Datos:** Holter de 24 h de 24 pacientes (ITMed, anonimizados)
#
# Muchos modelos clínicos predicen sobre **pequeñas unidades**: un parche de una lámina, un corte de una CT,
# un latido de un Holter. Pero la decisión clínica es sobre **el paciente**: ¿la lámina tiene tumor?
# ¿el Holter es anormal? Hay que **agregar** decenas de miles de predicciones en una sola.
#
# Aquí, un *random forest* entrega para **cada latido** la probabilidad de ser ectópico (S o V). Cada paciente
# fue evaluado por un modelo que **no lo vio** al entrenar. Definimos **Holter anormal** como el de los
# pacientes con ≥ 1 % de latidos ectópicos (12 de los 24).
#
# | Paso | Qué haces | Tiempo |
# |---|---|---|
# | 1 | Comparas cuatro reglas de agregación | 4 min |
# | 2 | Ajustas dos umbrales: el del latido y el del paciente | 4 min |
# | 3 | Trasladas la idea a patología digital | 2 min |

# %%
# INCLUDE: _setup.py

# %%
#@title 📦 Cargar las predicciones por paciente
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as w
from sklearn.metrics import roc_auc_score

p = np.load(obtener_datos("holter") / "pacientes.npz")
anormal = p["grupo"] == "anormal"
hist, horas, n = p["hist"], p["horas"], p["n_latidos"]


def sobre(t):
    """Número de latidos con probabilidad ≥ t, por paciente (a partir del histograma)."""
    return hist[:, int(round(t * 100)):].sum(axis=1)


REGLAS = {
    "Máximo: el latido más sospechoso": p["score_max"],
    "Promedio de todos los latidos": p["score_medio"],
    "Fracción de latidos con prob. ≥ 0,5": sobre(0.5) / n,
    "Latidos con prob. ≥ 0,5 por hora": sobre(0.5) / horas,
}
print(f"{len(n)} pacientes · {n.sum():,} latidos en total".replace(",", "."))

# %% [markdown]
# ## Paso 1 · Cuatro reglas para pasar de latidos a paciente
# Cada punto es un paciente. Una buena regla deja a los **anormales** (rojo) separados de los **normales** (gris).

# %%
#@title 📊 Comparar reglas
fig, ejes = plt.subplots(1, 4, figsize=(16, 3.8))
rng = np.random.default_rng(0)
for eje, (nombre, valor) in zip(ejes, REGLAS.items()):
    for g, color, x in ((False, "#7f7f7f", 0), (True, "#d62728", 1)):
        v = valor[anormal == g]
        eje.scatter(x + rng.uniform(-0.12, 0.12, len(v)), v, color=color, alpha=0.8, s=30)
    eje.set_xticks([0, 1], ["normal", "anormal"]); eje.set_xlim(-0.5, 1.5)
    if nombre.startswith("Máximo"):
        eje.set_ylim(0.99, 1.0005)
    else:
        eje.set_yscale("log")
    eje.set_title(f"{nombre}\nAUC = {roc_auc_score(anormal, valor):.2f}", fontsize=10)
    eje.grid(alpha=0.3)
plt.tight_layout(); plt.show()

# %% [markdown]
# ### ✍️ Pregunta 1
# ¿Por qué la regla del **máximo** no sirve aquí, si el modelo de latidos es muy bueno (AUC ≈ 0,99 por latido)?
# Pista: ¿cuántos latidos tiene un Holter de 24 h?
#
# _Respuesta:_
#
# ## Paso 2 · Dos umbrales
# 1. **Umbral del latido**: desde qué probabilidad un latido cuenta como ectópico.
# 2. **Umbral del paciente**: cuántos latidos ectópicos **por hora** hacen anormal un Holter.

# %%
#@title 🎚️ Ajustar los dos umbrales


def dos_umbrales(umbral_latido, umbral_por_hora):
    por_hora = sobre(umbral_latido) / horas
    pred = por_hora >= umbral_por_hora
    vp, fp = (pred & anormal).sum(), (pred & ~anormal).sum()
    fn, vn = (~pred & anormal).sum(), (~pred & ~anormal).sum()
    orden = np.argsort(por_hora)
    plt.figure(figsize=(11, 3.6))
    plt.bar(range(len(orden)), por_hora[orden] + 0.1, color=np.where(anormal[orden], "#d62728", "#7f7f7f"))
    plt.axhline(umbral_por_hora, color="k", ls="--", lw=1)
    plt.yscale("log"); plt.xticks(range(len(orden)), p["record"][orden], rotation=90, fontsize=8)
    plt.ylabel("latidos 'ectópicos' por hora"); plt.title("Rojo: anormal · gris: normal · línea: umbral del paciente")
    plt.tight_layout(); plt.show()
    print(f"Pacientes anormales detectados: {vp}/{vp + fn} · normales marcados como anormales: {fp}/{fp + vn}")


w.interact(dos_umbrales,
           umbral_latido=w.FloatSlider(value=0.5, min=0.05, max=0.99, step=0.01, description="Latido",
                                       continuous_update=False),
           umbral_por_hora=w.FloatLogSlider(value=30, base=10, min=0, max=3.5, step=0.05, description="Por hora",
                                            continuous_update=False));

# %% [markdown]
# ### ✍️ Pregunta 2
# Sube el **umbral del latido** a 0,95. ¿Qué pasa con los pacientes normales? ¿Tienes que mover el umbral por hora?
#
# _Respuesta:_
#
# ## Paso 3 · ✍️ Decisión: ¿y en patología digital?
# Una lámina de biopsia de ganglio se divide en ~20.000 parches y un modelo da la probabilidad de **metástasis**
# en cada uno. Una micrometástasis puede ocupar **solo 3–4 parches**.
#
# 1. ¿Usarías el **máximo**, el **promedio** o un **conteo**? ¿Qué riesgo tiene cada uno?
# 2. ¿Qué agregarías para que un solo parche ruidoso no marque la lámina como positiva?
#    (Pista: ¿los parches positivos están juntos o dispersos?)
#
# _Respuesta:_
#
# ### Para llevarse
# - Un buen modelo por **instancia** no garantiza una buena decisión por **paciente**: la **regla de agregación** es parte del modelo.
# - Con miles de instancias por paciente, el **máximo** casi siempre encuentra un falso positivo.
# - Hay **dos umbrales** que calibrar, y el del paciente depende de la definición clínica (p. ej., ectópicos por hora).
