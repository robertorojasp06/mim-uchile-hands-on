# %% [markdown]
# # Sesión 4 · Actividad C — El umbral es una decisión clínica
#
# **Duración:** 20 min · **Cómputo:** CPU · **Datos:** Holter de 24 h de 24 pacientes (ITMed, anonimizados)
#
# Entrenamos un clasificador **clásico** (*machine learning* con características diseñadas a mano) que decide
# si cada latido es **normal** o **ectópico** (extrasístole supraventricular **S** o ventricular **V**).
# El modelo entrega una **probabilidad**; el **umbral** que la convierte en alarma lo eliges tú.
#
# - Entrenamiento: 18 pacientes (muestra de latidos).
# - Prueba: 6 pacientes **distintos**, con **todos** sus latidos de 24 h (3 con muchos ectópicos y 3 normales).
#
# | Paso | Qué haces | Tiempo |
# |---|---|---|
# | 1 | Exploras las características de cada latido | 4 min |
# | 2 | Eliges características y modelo, y entrenas | 4 min |
# | 3 | Mueves el umbral y ves sensibilidad, VPP y falsas alarmas | 8 min |
# | 4 | Eliges un umbral para dos escenarios clínicos | 4 min |

# %%
# INCLUDE: _setup.py

# %%
#@title 📦 Cargar las características de los latidos
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as w
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score

RAIZ = obtener_datos("holter")
latidos = pd.read_parquet(RAIZ / "features.parquet")
pacientes = pd.read_csv(RAIZ / "records.csv").set_index("record")
CARACTERISTICAS = {
    "rr_prev_ms": "Intervalo RR con el latido anterior (ms)",
    "rr_next_ms": "Intervalo RR con el latido siguiente (ms)",
    "prematuridad": "RR anterior / RR típico de los vecinos (<1 = prematuro)",
    "pausa_compensatoria": "RR siguiente / RR anterior (>1 = pausa después del latido)",
    "ancho_qrs_ms": "Ancho estimado del QRS (ms)",
    "amplitud_relativa": "Amplitud del QRS / amplitud típica del paciente",
    "correlacion_plantilla": "Parecido con el latido típico del paciente (1 = idéntico)",
}
COLORES = {"N": "#7f7f7f", "S": "#1f77b4", "V": "#d62728"}
entrena = latidos[latidos.split != "test"]
prueba = latidos[latidos.split == "test"].reset_index(drop=True)
display(pd.DataFrame(CARACTERISTICAS.items(), columns=["característica", "qué mide"]).style.hide(axis="index"))
print("Entrenamiento:", entrena.code.value_counts().to_dict(), "· Prueba:", prueba.code.value_counts().to_dict())

# %% [markdown]
# ## Paso 1 · ¿Qué característica separa los latidos?

# %%
#@title 📈 Distribución por tipo de latido


def histograma(caracteristica):
    x = entrena[caracteristica]
    lo, hi = np.nanpercentile(x, [0.5, 99.5])
    bins = np.linspace(lo, hi, 60)
    plt.figure(figsize=(8, 3.5))
    for code, nombre in [("N", "normal"), ("S", "supraventricular"), ("V", "ventricular")]:
        plt.hist(entrena.loc[entrena.code == code, caracteristica], bins=bins, density=True,
                 histtype="stepfilled", alpha=0.35, color=COLORES[code], label=f"{code} · {nombre}")
    plt.xlabel(CARACTERISTICAS[caracteristica]); plt.ylabel("densidad"); plt.legend(); plt.tight_layout(); plt.show()


w.interact(histograma, caracteristica=w.Dropdown(options=list(CARACTERISTICAS), description="Característica"));

# %% [markdown]
# ### ✍️ Pregunta 1
# ¿Qué característica distingue mejor un latido **S** de uno normal? ¿Y un **V**? ¿Por qué tiene sentido fisiológico?
#
# _Respuesta:_
#
# ## Paso 2 · Entrenar
# Marca las características que quieres usar y elige el modelo. Luego ejecuta la celda.

# %%
#@title 🛠️ Elegir características y modelo
modelo_tipo = "Random forest"  #@param ["Regresión logística", "Random forest"]
rr_prev_ms = True  #@param {type:"boolean"}
rr_next_ms = True  #@param {type:"boolean"}
prematuridad = True  #@param {type:"boolean"}
pausa_compensatoria = True  #@param {type:"boolean"}
ancho_qrs_ms = True  #@param {type:"boolean"}
amplitud_relativa = True  #@param {type:"boolean"}
correlacion_plantilla = True  #@param {type:"boolean"}

usar = [c for c in CARACTERISTICAS if globals()[c]]
if modelo_tipo == "Random forest":
    modelo = RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=5,
                                    class_weight="balanced", n_jobs=-1, random_state=0)
else:
    modelo = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))

t0 = time.time()
modelo.fit(entrena[usar], entrena.ectopico)
prob = modelo.predict_proba(prueba[usar])[:, 1]
y = prueba.ectopico.to_numpy()
fpr, tpr, umbrales_roc = roc_curve(y, prob)
auc = roc_auc_score(y, prob)
print(f"{modelo_tipo} con {len(usar)} características · entrenado en {time.time() - t0:.0f} s · AUC en prueba = {auc:.4f}")

if modelo_tipo == "Random forest":
    peso = pd.Series(modelo.feature_importances_, usar)
    titulo = "Importancia (random forest)"
else:
    peso = pd.Series(modelo[-1].coef_[0], usar)
    titulo = "Coeficiente (regresión logística, datos estandarizados)"
peso.sort_values().plot.barh(figsize=(6, 0.35 * len(usar) + 1), color="#4c72b0", title=titulo)
plt.tight_layout(); plt.show()

# %% [markdown]
# ## Paso 3 · Mueve el umbral
#
# Un latido se marca como **ectópico** si su probabilidad ≥ umbral. A la derecha, el punto muestra dónde
# queda ese umbral en la curva ROC. La tabla cuenta, **para cada paciente de prueba (24 h)**, cuántos
# ectópicos se pierden y cuántas **falsas alarmas** genera el modelo.

# %%
#@title 🎚️ Umbral interactivo


def con_umbral(umbral):
    p = prob >= umbral
    vp, fp = (p & (y == 1)).sum(), (p & (y == 0)).sum()
    fn, vn = (~p & (y == 1)).sum(), (~p & (y == 0)).sum()
    sens, espec, vpp = vp / (vp + fn), vn / (vn + fp), vp / max(vp + fp, 1)

    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.2))
    fracc = np.array([[vn / (vn + fp), fp / (vn + fp)], [fn / (fn + vp), vp / (fn + vp)]])
    a.imshow(fracc, cmap="Blues", vmin=0, vmax=1)
    for (i, j), v in np.ndenumerate(np.array([[vn, fp], [fn, vp]])):
        a.text(j, i, f"{v:,}".replace(",", "."), ha="center", va="center", fontsize=13,
               color="white" if fracc[i, j] > 0.5 else "black")
    a.set_xticks([0, 1], ["normal", "ectópico"]); a.set_yticks([0, 1], ["normal", "ectópico"])
    a.set_xlabel("predicción"); a.set_ylabel("realidad"); a.set_title("Matriz de confusión (6 pacientes, 24 h)")
    b.plot(fpr, tpr, color="#4c72b0", lw=1.5, label=f"ROC (AUC = {auc:.3f})")
    b.scatter([1 - espec], [sens], color="#d62728", s=80, zorder=3, label=f"umbral = {umbral:.2f}")
    b.set_xscale("log"); b.set_xlim(1e-5, 1); b.set_ylim(0, 1.02); b.grid(alpha=0.3)
    b.set_xlabel("1 − especificidad (escala log)"); b.set_ylabel("sensibilidad"); b.legend(loc="lower right")
    plt.tight_layout(); plt.show()
    print(f"Sensibilidad {sens:.1%} · Especificidad {espec:.2%} · VPP {vpp:.1%} · "
          f"de cada 100 alarmas, {100 * vpp:.0f} son ectópicos reales")

    filas = []
    for rec, g in prueba.assign(p=p).groupby("record"):
        filas.append({"paciente": rec, "grupo": pacientes.loc[rec, "grupo"],
                      "ectópicos reales": int(g.ectopico.sum()),
                      "detectados": int((g.p & (g.ectopico == 1)).sum()),
                      "perdidos": int((~g.p & (g.ectopico == 1)).sum()),
                      "falsas alarmas / 24 h": int((g.p & (g.ectopico == 0)).sum())})
    display(pd.DataFrame(filas).style.hide(axis="index")
            .background_gradient(subset=["falsas alarmas / 24 h", "perdidos"], cmap="Reds"))


w.interact(con_umbral, umbral=w.FloatSlider(value=0.5, min=0.01, max=0.99, step=0.01, description="Umbral",
                                            continuous_update=False));

# %% [markdown]
# ### ✍️ Pregunta 2
# 1. El AUC es altísimo. ¿Cuántas falsas alarmas genera igual en los pacientes **normales**?
# 2. ¿Por qué el VPP es mucho más bajo que la especificidad?
#
# _Respuesta:_
#
# ## Paso 4 · ✍️ Decisión: dos escenarios, dos umbrales
#
# | Escenario | Quién mira las alarmas | Qué duele más |
# |---|---|---|
# | **A · Informe de Holter** | Un cardiólogo revisa las marcas del día siguiente | Perder ectópicos (el informe queda incompleto) |
# | **B · Alarma en la unidad coronaria** | Una enfermera, en tiempo real, para 8 pacientes | Interrumpir sin motivo (fatiga de alarmas) |
#
# Para cada escenario, anota el umbral que elegirías, su sensibilidad y las falsas alarmas / 24 h del peor paciente normal.
#
# - **Escenario A** · umbral: ___ · sensibilidad: ___ · falsas alarmas / 24 h: ___
# - **Escenario B** · umbral: ___ · sensibilidad: ___ · falsas alarmas / 24 h: ___
#
# ¿Cambia tu decisión si usas **regresión logística** o **menos características**? Vuelve al Paso 2 y prueba.
#
# ### Para llevarse
# - El modelo entrega probabilidades; **el umbral es una decisión clínica**, no técnica.
# - Un AUC alto no garantiza pocas falsas alarmas cuando la condición es **poco frecuente**: mira el **VPP**.
# - Evalúa **por paciente** y en pacientes **que el modelo no vio**.
