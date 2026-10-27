# Hands-on · Procesamiento de Imágenes y Bioseñales II (MIM, U. de Chile, 2026)

Actividades prácticas en Google Colab para las **Sesiones 4 y 5** del programa (unidades 2 y 3; diapositivas
"Clases 1 y 3 — Deep learning para apoyo diagnóstico"). Todas corren **solo con CPU**, son en español y usan **datos locales anonimizados** (todos los estudiantes
usan exactamente el mismo conjunto en cada actividad).

| Actividad | Abrir | Notebook | Duración | Cómputo | Datos | Diapositivas relacionadas |
|---|---|---|---|---|---|---|
| **4A** · U-Net preentrenada en patología digital | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/robertorojasp06/mim-uchile-hands-on/blob/main/notebooks/S4A_unet_patologia.ipynb) | [`S4A_unet_patologia.ipynb`](notebooks/S4A_unet_patologia.ipynb) | 25 min | CPU (~30 s de inferencia) | `wsi` | 17–18, 25, 31, 37 |
| **4B** · Entrena una CNN 1D con latidos de Holter | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/robertorojasp06/mim-uchile-hands-on/blob/main/notebooks/S4B_cnn1d_ecg.ipynb) | [`S4B_cnn1d_ecg.ipynb`](notebooks/S4B_cnn1d_ecg.ipynb) | 25–30 min | CPU (~20–60 s por entrenamiento) | `holter` | 20, 24–27, 34–36 |
| **4C** · El umbral es una decisión clínica | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/robertorojasp06/mim-uchile-hands-on/blob/main/notebooks/S4C_umbral_holter.ipynb) | [`S4C_umbral_holter.ipynb`](notebooks/S4C_umbral_holter.ipynb) | 20 min | CPU | `holter` | 8–13, 37–38 |
| **4-s1** · La trampa del *accuracy* | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/robertorojasp06/mim-uchile-hands-on/blob/main/notebooks/S4s1_trampa_accuracy.ipynb) | [`S4s1_trampa_accuracy.ipynb`](notebooks/S4s1_trampa_accuracy.ipynb) | 10 min | CPU | `holter` | 37 |
| **4-s2** · Dice, IoU y Hausdorff entre dos expertos | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/robertorojasp06/mim-uchile-hands-on/blob/main/notebooks/S4s2_metricas_segmentacion.ipynb) | [`S4s2_metricas_segmentacion.ipynb`](notebooks/S4s2_metricas_segmentacion.ipynb) | 10 min | CPU | `ct` | 34, 37 |
| **5A** · MedSAM con un humano en el ciclo | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/robertorojasp06/mim-uchile-hands-on/blob/main/notebooks/S5A_medsam_hitl.ipynb) | [`S5A_medsam_hitl.ipynb`](notebooks/S5A_medsam_hitl.ipynb) | 25–30 min | CPU (~0,3 s por corte) | `medsam` | 40–44, 46 |
| **5B** · *Pipeline* de una lámina completa (WSI) | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/robertorojasp06/mim-uchile-hands-on/blob/main/notebooks/S5B_pipeline_wsi.ipynb) | [`S5B_pipeline_wsi.ipynb`](notebooks/S5B_pipeline_wsi.ipynb) | 25 min | CPU (~1 min en el Paso 4) | `lamina` | 31 |
| **5C** · Ajuste fino a datos locales | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/robertorojasp06/mim-uchile-hands-on/blob/main/notebooks/S5C_ajuste_fino.ipynb) | [`S5C_ajuste_fino.ipynb`](notebooks/S5C_ajuste_fino.ipynb) | 25 min | CPU (~15 s–2 min por entrenamiento) | `ajuste` | 27, 40, 43–44 |
| **5-s1** · De latidos a paciente (y de parches a lámina) | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/robertorojasp06/mim-uchile-hands-on/blob/main/notebooks/S5s1_latidos_a_paciente.ipynb) | [`S5s1_latidos_a_paciente.ipynb`](notebooks/S5s1_latidos_a_paciente.ipynb) | 10 min | CPU | `holter` | 31, 37 |
| **5-s2** · Galería de errores de nnU-Net | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/robertorojasp06/mim-uchile-hands-on/blob/main/notebooks/S5s2_galeria_errores.ipynb) | [`S5s2_galeria_errores.ipynb`](notebooks/S5s2_galeria_errores.ipynb) | 10 min | CPU | `galeria` | 37–38 |

Cada notebook tiene la misma estructura: tabla de pasos con tiempos, celdas con formularios de Colab
(el código queda oculto, pero se puede abrir), gráficos interactivos y celdas **✍️ Decisión / Pregunta**
donde el estudiante escribe su elección usando la terminología técnica.

## Puesta en marcha (equipo docente)

1. **Datos (ya publicados).** Los `dist/*.zip` (7 archivos, ~190 MB) están en una carpeta de **sasiba**
   (ownCloud de la U. de Chile) compartida con un enlace público **protegido con clave**:
   `https://sasiba.uchile.cl/index.php/s/MC5VkrEeFNEZV30`. El enlace ya está en `DATA_URL`
   ([`notebooks/src/_setup.py`](notebooks/src/_setup.py)). Los notebooks descargan cada zip por WebDAV
   (`/public.php/webdav/<nombre>.zip`, usuario = código del enlace, contraseña = clave).
2. **La clave no se guarda en el repositorio.** Se entrega en clase y cada estudiante la escribe en el campo
   `CLAVE_DATOS` de la primera celda. Con la clave vacía o incorrecta, la celda avisa en español.
   Para actualizar los datos: regenerar los zips (`python prep/package_data.py`) y reemplazar los archivos en
   la carpeta de sasiba **con el mismo nombre**; el enlace y la clave no cambian.
3. **Abrir en Colab.** El repositorio es **público**: cada notebook se abre con el botón *Open in Colab* de la
   tabla de arriba, o con `https://colab.research.google.com/github/robertorojasp06/mim-uchile-hands-on/blob/main/notebooks/<notebook>.ipynb`.
   Basta una **cuenta de Google**; no se necesita cuenta de GitHub. El repositorio no contiene datos ni la clave.
   Los enlaces muestran siempre la versión más reciente de `main`.

Ninguna actividad instala paquetes: usan lo que ya trae Colab (numpy, pandas, pyarrow, scipy, scikit-learn,
scikit-image, PyTorch, matplotlib, ipywidgets, Pillow).

## Instrucciones para los estudiantes

Texto listo para copiar en U-Cursos, el chat de la clase o una diapositiva (cambiar el enlace según la actividad):

> 1. Abre el enlace de la actividad (por ejemplo, la actividad 4A:
>    <https://colab.research.google.com/github/robertorojasp06/mim-uchile-hands-on/blob/main/notebooks/S4A_unet_patologia.ipynb>)
>    e inicia sesión con tu cuenta de Google.
> 2. Haz clic en **Archivo → Guardar una copia en Drive** (*File → Save a copy in Drive*). Trabaja siempre en esa copia: ahí quedan tus respuestas.
> 3. En la primera celda (⚙️ *Preparar el entorno*), escribe la **clave de los datos** que entregamos en clase en
>    el campo `CLAVE_DATOS`.
> 4. Ejecuta las celdas **en orden**, de arriba hacia abajo (▶ a la izquierda de cada celda, o Mayús + Enter).
>    Si Colab avisa que el notebook no fue creado por Google, elige **Ejecutar de todos modos** (*Run anyway*).
> 5. Responde en las celdas ✍️ (doble clic para editar).

No se necesita GPU: si Colab ofrece una, no hace falta cambiar el tipo de entorno.

## Probar una actividad (equipo docente)

1. Abrir el notebook con su botón *Open in Colab* (tabla de arriba).
2. **Archivo → Guardar una copia en Drive** (*File → Save a copy in Drive*).
3. Usar CPU, como los estudiantes: **Entorno de ejecución → Cambiar tipo de entorno de ejecución → CPU**
   (*Runtime → Change runtime type → Hardware accelerator: CPU → Save*).
4. Escribir la clave en `CLAVE_DATOS` (primera celda) y ejecutar las celdas **en orden**.
5. Comparar con la [guía para docentes](#guía-para-docentes-qué-deberían-encontrar).

Prueba rápida sin Colab (en un computador con los datos en `data/`):

```bash
cd notebooks
python ../prep/smoke_test.py src/S4A_unet_patologia.py   # ejecuta todas las celdas; figuras en smoke_test_figs/
```


## Datos

| Conjunto | Origen | Contenido | Tamaño |
|---|---|---|---|
| `holter` | 24 Holter de 24 h (ITMed, filtrados en Spacelabs Pathfinder y validados por expertos) | `records.csv`, `windows.npz` (24.087 latidos N/S/V, ventanas de 4 s, 128 Hz, mV), `features.parquet` (7 características por latido), `pacientes.npz` (histograma por paciente de las probabilidades por latido, para 5-s1) | 35 MB |
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
nnU-Net (UID DICOM) se reemplazan. De la lámina de 5B se exporta solo la mitad izquierda del vidrio: la derecha tiene
anotaciones a mano que podrían ser un número de caso. Por la misma razón, la figura introductoria de 4A
(`notebooks/img/S4A_lamina_colon.jpg`, otra lámina de colon) muestra solo la parte superior del vidrio. Nunca se leen
las imágenes asociadas (macro, etiqueta).

## Regenerar todo

```bash
python prep/export_holter.py                                   # data/holter  (requiere acceso a ISATEC/mvp-holter)
python prep/export_wsi.py                                      # data/wsi     (requiere deeppath-ml-tissue-segmentation)
~/anaconda3/envs/nnunet/bin/python prep/export_ct.py           # data/ct      (requiere nibabel y el disco "Nuevo vol2")
<venv-medsam>/bin/python prep/export_medsam.py                 # data/medsam  (torch==2.5.1, transformers==4.46.3, nibabel)
<deeppath>/.venv/bin/python prep/export_slide.py               # data/lamina  (requiere openslide)
<deeppath>/.venv/bin/python prep/export_intro_figure.py       # notebooks/img/S4A_lamina_colon.jpg (figura de 4A)
python prep/export_finetune.py                                 # data/ajuste  (entrena las curvas de referencia, ~10 min)
python prep/export_gallery.py                                  # data/galeria (requiere ~/nnunet-final-predictions)
python prep/package_data.py                                    # dist/*.zip
python prep/build_notebooks.py                                 # notebooks/*.ipynb desde notebooks/src/*.py
```

Para revisar un notebook sin Colab, ver `prep/smoke_test.py` (sección *Probar una actividad*).

Los notebooks se editan en `notebooks/src/*.py` (formato *percent*: `# %%` celda de código,
`# %% [markdown]` celda de texto) y se generan con `build_notebooks.py`. Las celdas que empiezan con
`#@title` quedan como formularios de Colab con el código oculto.

## Guía para docentes: qué deberían encontrar

Valores obtenidos al ejecutar los notebooks con los parámetros por defecto (CPU, semilla 0).

**4A · U-Net en patología**
- Paso 2: el panel de texto muestra las tres probabilidades del píxel elegido y su suma (= 1,000, por la softmax).
  En el parche por defecto (burbujas de aire) la confianza baja (~0,35–0,5) en un **anillo en el borde de la
  burbuja**, que es justo donde la predicción difiere de la anotación.
- Paso 3a: con los tamaños típicos, el tejido, las burbujas, los pliegues y las criptas miden ≥ 60 px incluso a
  10x; un núcleo mide ~8 px a 10x y ~31 px a 40x, y los cromosomas de una mitosis (~1,5 µm) solo llegan a ~7 px a 40x.
- Paso 3b (cada modelo en su aumento, Dice agrupado): tejido 0,93 / 0,88 / 0,96 y artefacto 0,90 / 0,87 / 0,92 a
  10x / 20x / 40x. El aumento **casi no cambia** la segmentación de tejido y artefacto; 20x incluso queda algo por
  debajo (pliegues 0,47), y con 3 parches por región parte de eso es ruido.
- El Dice **promedio por parche** (0,62–0,80) es mucho más bajo que el **agrupado** (0,87–0,96): los parches con
  muy pocos píxeles de una clase arrastran el promedio. Buena discusión sobre cómo reportar.
- Paso 3c: una lámina de 15 × 20 mm son ~4.100 parches a 10x (~6 min en CPU) y ~65.000 a 40x (~1,5 h): 16×.
- Decisión 1 esperada: control de calidad a **10x** (mismo Dice, 16× más barato); mitosis a **40x**, pero solo
  dentro del tejido que dejó la máscara de 10x (flujo multiescala, que se retoma en 5B).
- Validación externa en gástrico: el tejido se mantiene, pero el artefacto baja (0,92 → 0,79 a 40x);
  peor en **desenfoque (−0,26)**, **burbujas (−0,22)** y **sombras (−0,14)**.
- Umbral de artefacto (20x): bajar de 0,5 a 0,2 reduce el artefacto colado (colon 24 % → 12 %; gástrico
  27 % → 13 %) a costa de descartar algo más de tejido (~0,1–0,2 % → 0,8 %). Además **sube el Dice** de ambas
  clases (colon: artefacto 0,86 → 0,92): el umbral 0,5 no es el óptimo.
- En el Paso 2, el parche por defecto (burbujas de aire) muestra Dice tejido = 0,00: el experto no anotó tejido
  y la red predice algunos píxeles de tejido. Es un buen ejemplo de por qué el Dice por parche es inestable.

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
