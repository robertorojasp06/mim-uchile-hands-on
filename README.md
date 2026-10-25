# Hands-on · Procesamiento de Imágenes y Bioseñales II (MIM, U. de Chile, 2026)

Actividades prácticas en Google Colab para las **Sesiones 4 y 5** (Clases 1 y 3, *Deep learning para apoyo
diagnóstico*). Todas corren **solo con CPU**, son en español y usan **datos locales anonimizados** (todos los estudiantes
usan exactamente el mismo conjunto en cada actividad).

| Actividad | Notebook | Duración | Cómputo | Datos | Diapositivas relacionadas |
|---|---|---|---|---|---|
| **4A** · U-Net preentrenada en patología digital | [`S4A_unet_patologia.ipynb`](notebooks/S4A_unet_patologia.ipynb) | 25 min | CPU (~1 min de inferencia) | `wsi` | 17–18, 25, 31, 37 |
| **4B** · Entrena una CNN 1D con latidos de Holter | [`S4B_cnn1d_ecg.ipynb`](notebooks/S4B_cnn1d_ecg.ipynb) | 25–30 min | CPU (~20–60 s por entrenamiento) | `holter` | 20, 24–27, 34–36 |
| **4C** · El umbral es una decisión clínica | [`S4C_umbral_holter.ipynb`](notebooks/S4C_umbral_holter.ipynb) | 20 min | CPU | `holter` | 8–13, 37–38 |
| **4-s1** · La trampa del *accuracy* | [`S4s1_trampa_accuracy.ipynb`](notebooks/S4s1_trampa_accuracy.ipynb) | 10 min | CPU | `holter` | 37 |
| **4-s2** · Dice, IoU y Hausdorff entre dos expertos | [`S4s2_metricas_segmentacion.ipynb`](notebooks/S4s2_metricas_segmentacion.ipynb) | 10 min | CPU | `ct` | 34, 37 |
| **5A** · MedSAM con un humano en el ciclo | [`S5A_medsam_hitl.ipynb`](notebooks/S5A_medsam_hitl.ipynb) | 25–30 min | CPU (~0,3 s por corte) | `medsam` | 40–44, 46 |
| **5B** · *Pipeline* de una lámina completa (WSI) | [`S5B_pipeline_wsi.ipynb`](notebooks/S5B_pipeline_wsi.ipynb) | 25 min | CPU (~1 min en el Paso 4) | `lamina` | 31 |
| **5C** · Ajuste fino a datos locales | [`S5C_ajuste_fino.ipynb`](notebooks/S5C_ajuste_fino.ipynb) | 25 min | CPU (~15 s–2 min por entrenamiento) | `ajuste` | 27, 40, 43–44 |
| **5-s1** · De latidos a paciente (y de parches a lámina) | [`S5s1_latidos_a_paciente.ipynb`](notebooks/S5s1_latidos_a_paciente.ipynb) | 10 min | CPU | `holter` | 31, 37 |
| **5-s2** · Galería de errores de nnU-Net | [`S5s2_galeria_errores.ipynb`](notebooks/S5s2_galeria_errores.ipynb) | 10 min | CPU | `galeria` | 37–38 |

Cada notebook tiene la misma estructura: tabla de pasos con tiempos, celdas con formularios de Colab
(el código queda oculto, pero se puede abrir), gráficos interactivos y celdas **✍️ Decisión / Pregunta**
donde el estudiante escribe su elección usando la terminología técnica.

## Puesta en marcha (equipo docente)

1. **Publicar los datos.** Subir los `dist/*.zip` (7 archivos, ~190 MB) a un lugar desde el que
   Colab pueda descargarlos con una URL directa del tipo `<DATA_URL>/<nombre>.zip`
   (por ejemplo, los *assets* de un *release* de GitHub, o un servidor web del laboratorio).
   ⚠️ Son datos clínicos **anonimizados**: elegir el lugar según lo que permitan el comité de ética y los
   convenios con HCUCH e ITMed. Un enlace "compartido" de Google Drive no sirve tal cual (no es una URL directa).
2. **Pegar la URL** en la constante `DATA_URL` de [`notebooks/src/_setup.py`](notebooks/src/_setup.py) y
   regenerar los notebooks (`python prep/build_notebooks.py`), o entregarla a los estudiantes para que la
   peguen en la primera celda.
3. **Abrir en Colab.** Si el repositorio está en GitHub:
   `https://colab.research.google.com/github/<usuario>/<repo>/blob/main/notebooks/<notebook>.ipynb`.
   Los estudiantes deben usar **Archivo → Guardar una copia en Drive** para escribir sus respuestas.

Ninguna actividad instala paquetes: usan lo que ya trae Colab (numpy, pandas, scipy, scikit-learn,
PyTorch, matplotlib, ipywidgets, Pillow).

## Datos

| Conjunto | Origen | Contenido | Tamaño |
|---|---|---|---|
| `holter` | 24 Holter de 24 h (ITMed, filtrados en Spacelabs Pathfinder y validados por expertos) | `records.csv`, `windows.npz` (24.087 latidos N/S/V, ventanas de 4 s, 128 Hz, mV), `features.parquet` (7 características por latido) | 35 MB |
| `wsi` | Proyecto DeepPath (SCIAN-Lab): parches H&E de 300 × 300 de colon y gástrico, del *split* de prueba | 225 parches + máscaras (fondo/tejido/artefacto) a 10x, 20x y 40x; U-Nets de colon a 10x, 20x y 40x | 25 MB |
| `ct` | HCUCH: 4 estudios PET-CT segmentados por **dos radiólogos** de forma independiente | 8 cortes axiales (HU) por estructura con ambas máscaras, y métricas 3D | 0,8 MB |
| `medsam` | Mismas CT que `ct`; MedSAM (`wanglab/medsam-vit-base`, Apache-2.0) | Masa pulmonar (27 cortes) y lesión hepática (14 cortes) con ambas máscaras; *embeddings* de MedSAM precalculados; decodificador en TorchScript | 95 MB |
| `lamina` | DeepPath: una lámina H&E de colon (NDPI, 40x) | Miniatura a 0,625x, dos regiones de 3.000 × 3.000 a 10x, pirámide (`info.json`) y U-Net de colon 10x | 5,6 MB |
| `ajuste` | DeepPath: parches gástricos a 20x | 72 parches de entrenamiento (6 por región, en orden fijo) y 48 de prueba; U-Net de colon 20x; curvas de referencia | 20 MB |
| `galeria` | HCUCH: predicciones de nnU-Net (lesiones hepáticas y nódulos pulmonares) | 10 casos: corte axial con referencia y predicción, y vistas 3D | 6,3 MB |

**Selección de los Holter.** De 211 registros se descartaron los que tenían marcapasos, FA, >5 % de artefacto,
>5 % de latidos ruido, >1 % de latidos no clasificables, <20 h de registro, o conteos de S/V en `beats.csv`
que no coincidían (±5 %) con el informe firmado. De los 118 restantes se eligieron 12 **anormales** (≥1 % de
latidos S+V) y 12 **normales** (<50 S+V), pareados por edad y sexo. La separación entrenamiento/validación/prueba
es **por paciente** (14/4/6).

**Anonimización.** Los registros se renombran H01–H24 (la tabla de correspondencia queda en
`prep/private/holter_id_map.csv`, fuera de git). Nunca se copia el encabezado ISHNE (nombre, RUT, fecha de
nacimiento): se descartan todas las muestras que se superponen con él. La edad se entrega en tramos
(<65, 65–79, ≥80) y no se entregan fechas. Las CT se identifican solo por la estructura y los expertos como
"Experto 1/2". Los nombres de archivo de los parches (con identificadores de ROI) y de las predicciones de
nnU-Net (UID DICOM) se reemplazan. De la lámina se exporta solo la mitad izquierda del vidrio: la derecha tiene
anotaciones a mano que podrían ser un número de caso; nunca se leen las imágenes asociadas (macro, etiqueta).

## Regenerar todo

```bash
python prep/export_holter.py                                   # data/holter  (requiere acceso a ISATEC/mvp-holter)
python prep/export_wsi.py                                      # data/wsi     (requiere deeppath-ml-tissue-segmentation)
~/anaconda3/envs/nnunet/bin/python prep/export_ct.py           # data/ct      (requiere nibabel y el disco "Nuevo vol2")
<venv-medsam>/bin/python prep/export_medsam.py                 # data/medsam  (torch==2.5.1, transformers==4.46.3, nibabel)
<deeppath>/.venv/bin/python prep/export_slide.py               # data/lamina  (requiere openslide)
python prep/export_finetune.py                                 # data/ajuste  (entrena las curvas de referencia, ~10 min)
python prep/export_gallery.py                                  # data/galeria (requiere ~/nnunet-final-predictions)
python prep/package_data.py                                    # dist/*.zip
python prep/build_notebooks.py                                 # notebooks/*.ipynb desde notebooks/src/*.py
```

Los notebooks se editan en `notebooks/src/*.py` (formato *percent*: `# %%` celda de código,
`# %% [markdown]` celda de texto) y se generan con `build_notebooks.py`. Las celdas que empiezan con
`#@title` quedan como formularios de Colab con el código oculto.

## Guía para docentes: qué deberían encontrar

Valores obtenidos al ejecutar los notebooks con los parámetros por defecto (CPU, semilla 0).

**4A · U-Net en patología**
- Matriz modelo × aumento (Dice agrupado): 0,85–0,96. El aumento importa **menos de lo esperado** para este
  modelo; el mejor par es modelo 40x sobre datos 40x (tejido 0,96, artefacto 0,92).
- El Dice **promedio por parche** (0,5–0,8) es mucho más bajo que el **agrupado** (0,85–0,96): los parches con
  muy pocos píxeles de una clase arrastran el promedio. Buena discusión sobre cómo reportar.
- Costo: una lámina de 15 × 20 mm son ~4.000 parches a 10x y ~63.000 a 40x (16×).
- Validación externa en gástrico: el tejido se mantiene, pero el artefacto baja (0,92 → 0,79 a 40x);
  peor en **desenfoque (−0,26)**, **burbujas (−0,22)** y **sombras (−0,14)**.
- Umbral de artefacto: bajar de 0,5 a 0,2 reduce el artefacto colado (24 % → 13 % en gástrico) a costa de
  descartar algo más de tejido (0,2 % → 0,8 %).

**4B · CNN 1D** (15 épocas, ventana de 1,5 s; sensibilidad por clase en los 6 pacientes de prueba)

| Experimento | Sens. N | Sens. S | Sens. V | Acc. balanceada | Tiempo CPU |
|---|---|---|---|---|---|
| Por defecto (tasa 1e-3) | 0,61 | 0,48 | 0,99 | 0,69 | ~30 s |
| Semilla 1 | 0,80 | 0,58 | 0,91 | 0,76 | ~30 s |
| Tasa 1e-5 | 0,55 | 0,25 | 0,90 | 0,57 | ~30 s |
| Tasa 1e-1 | 0,93 | 0,49 | 0,96 | 0,80 | ~30 s |
| Fracción 0,04 y 40 épocas | 0,76 | 0,24 | 0,90 | 0,63 | ~15 s |
| Ventana de 1 s | 0,92 | 0,15 | 0,99 | 0,69 | ~20 s |
| **Separación por latido (fuga)** | 0,92 | 0,93 | 0,97 | **0,94** | ~30 s |

- **V** se aprende bien (forma ancha y distinta); **S** es difícil: es un latido de forma normal que llega
  antes de tiempo. Con la ventana de 1 s la red ni siquiera ve el latido anterior (S 0,15).
- Con 14 pacientes de entrenamiento el resultado **varía bastante con la semilla** (0,69 vs 0,76): conviene
  que distintos grupos usen semillas distintas y comparen.
- Con tasa 1e-1 la validación salta de época en época (costo entre 0,7 y 3,2); el buen resultado de prueba
  depende de haber guardado una época con suerte. Con 1e-5 el costo apenas baja.
- Fracción 0,04 + 40 épocas: el costo de entrenamiento baja sin parar y el de validación sube desde la época ~15.
- La **fuga de datos** sube la accuracy balanceada de 0,69 a 0,94: el resultado más llamativo de la actividad.
- Contraste con 4C: el *random forest* con características hechas a mano (AUC 0,996) supera a la CNN con
  tan pocos pacientes. Conecta con "el deep learning necesita muchos datos".

**4C · Umbral**
- Random forest con las 7 características: AUC 0,996 en los 6 pacientes de prueba.
- Con umbral 0,5: sensibilidad 99 %, especificidad 99 %, pero **VPP 78 %** y entre 43 y 1.837 falsas
  alarmas / 24 h por paciente; el paciente normal H09 tiene 4 ectópicos reales y **714 falsas alarmas**.
- Con umbral 0,95: falsas alarmas caen a 7–891 / 24 h, pero la sensibilidad baja a 67 %.
- Las características más útiles: prematuridad y RR (para S), correlación con plantilla y ancho del QRS (para V).

**4-s1 · Trampa del accuracy** (586.568 latidos, 3,4 % ectópicos)

| Algoritmo | Accuracy | Sensibilidad | VPP |
|---|---|---|---|
| A · siempre "normal" | 96,6 % | 0 % | — |
| B · sensible (umbral 0,30) | 97,9 % | 99,3 % | 62,5 % |
| C · conservador (umbral 0,97) | **98,1 %** | 48,6 % | 91,7 % |

Por *accuracy* gana C, que pierde la mitad de los ectópicos. En el paciente normal H09, A tiene 100 % de
*accuracy* y B genera 2.331 falsas alarmas con VPP 0,2 %.

**4-s2 · Dos expertos**

| Estructura | Dice 3D | Volumen experto 1 / 2 |
|---|---|---|
| Masa pulmonar | 0,93 | 219 / 213 ml |
| Lesiones hepáticas grandes | 0,87 | 2.291 / 1.916 ml |
| Lesiones hepáticas pequeñas | 0,77 | 67 / 91 ml |
| Adenopatías | 0,70 | 324 / 516 ml |

En adenopatías, el Experto 2 incluyó además las mesentéricas y retroperitoneales: la diferencia es de
**definición** de la estructura, no solo de borde. En el laboratorio, un falso positivo lejano de 3 px deja el
Dice en 0,995 pero lleva el Hausdorff de 0 a 112 mm; dilatar 1 px baja el Dice a 0,95 en la masa y a 0,82 en
las lesiones pequeñas.

**5A · MedSAM *human-in-the-loop*** (masa pulmonar, valores por defecto)
- Con una caja ajustada en el corte central, MedSAM logra Dice ≈ 0,85 con cada experto (expertos entre sí: 0,94).
  En los cortes extremos, donde la masa es pequeña, cae a 0–0,5.
- Flujo asistido (umbral de aceptación 0,8): 12 cortes aceptados, 3 con caja nueva y 10 a mano;
  **11 vs. 25 min (ahorro 56 %)**. Dice 3D del resultado vs. Experto 2 (evaluador independiente) = 0,89,
  cercano a la concordancia entre expertos (0,93).
- La lesión hepática (más pequeña) tiene propuestas peores (mediana 0,74): el ahorro es menor.

**5B · *Pipeline* WSI**
- La lámina a 40x son 15.075 megapíxeles: 45 GB RGB sin comprimir.
- Otsu sobre saturación detecta ~48 % de la lámina como tejido; sobre brillo incluye los bordes del vidrio.
  Ninguno de los dos captura bien la grasa (pálida).
- Con parches de 300 px y ≥10 % de tejido: ~5.600 parches a 10x (~6 min en CPU de Colab) vs. ~93 min a 40x.
- Región A: la U-Net marca **la mitad inferior como artefacto** porque está **desenfocada** en el escaneo
  (buen ejemplo de control de calidad: habría que reescanear); Dice Otsu–U-Net = 0,64. Región B: 0,97.
- La superposición del 50 % cambia solo ~1 % de los píxeles y cuesta 3,6× más tiempo.

**5C · Ajuste fino** (curvas de referencia, promedio de 3 semillas, 8 épocas)

| N parches | Ajuste fino: tejido / artefacto | Desde cero: tejido / artefacto |
|---|---|---|
| 0 (modelo de colon) | 0,91 / 0,88 | — |
| 4 | 0,90 / **0,80** | 0,35 / 0,15 |
| 16 | 0,95 / 0,91 | 0,58 / 0,00 |
| 64 | 0,96 / 0,91 | 0,72 / 0,00 |

- Con **4 parches el artefacto empeora** (sobreajuste a muy pocos ejemplos); desde 16 parches se estabiliza.
- Desde cero no converge en 8 épocas: buena discusión sobre transferencia y sobre si la comparación es justa.
- Regiones más difíciles para el modelo de colon en estómago: tinta china (0,73), pliegues (0,74), sombras (0,83).

**5-s1 · De latidos a paciente** (24 pacientes, puntajes fuera de muestra)
- Máximo: AUC 0,62 (todos los Holter tienen algún latido con probabilidad ≈ 1). Promedio: 0,98.
  Fracción ≥ 0,5 y conteo por hora: 1,00.
- Con umbral de latido 0,5 y 30 ectópicos/h: 12/12 anormales detectados y 4/12 normales mal marcados;
  con umbral de latido 0,95: 0/12 normales mal marcados.

**5-s2 · Galería nnU-Net**: no tiene respuesta única; sirve para discutir. Casos útiles: el 7 (masa grande no
anotada en la referencia ni segmentada: ¿error de referencia o fuera de la tarea?), el 3 (lesiones necróticas y
lesiones pequeñas agrupadas) y los nódulos pulmonares pequeños omitidos.
