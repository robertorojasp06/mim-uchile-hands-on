#@title ⚙️ Preparar el entorno (ejecuta esta celda primero)
#@markdown URL de la carpeta con los datos del curso (la entrega el equipo docente).
DATA_URL = "PEGAR_AQUI_LA_URL_DE_LOS_DATOS"  #@param {type:"string"}

import urllib.request
import zipfile
from pathlib import Path


def obtener_datos(nombre):
    """Devuelve la carpeta local del conjunto de datos `nombre`; si no existe, la descarga."""
    for base in (Path("data"), Path("../data"), Path("/content/data")):
        if (base / nombre).is_dir():
            return base / nombre
    if "PEGAR_AQUI" in DATA_URL:
        raise RuntimeError("Falta DATA_URL: pega la URL que entregó el equipo docente y vuelve a ejecutar.")
    destino = Path("/content/data") if Path("/content").exists() else Path("data")
    destino.mkdir(parents=True, exist_ok=True)
    archivo = destino / f"{nombre}.zip"
    print(f"Descargando {nombre}.zip ...")
    urllib.request.urlretrieve(f"{DATA_URL.rstrip('/')}/{nombre}.zip", archivo)
    with zipfile.ZipFile(archivo) as z:
        z.extractall(destino)
    archivo.unlink()
    return destino / nombre
