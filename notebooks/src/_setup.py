#@title ⚙️ Preparar el entorno (ejecuta esta celda primero)
#@markdown Escribe la **clave de los datos** que entregó el equipo docente y ejecuta la celda.
CLAVE_DATOS = ""  #@param {type:"string"}
DATA_URL = "https://sasiba.uchile.cl/index.php/s/MC5VkrEeFNEZV30"

import base64
import shutil
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


def _url_y_cabeceras(nombre):
    """Enlace público de ownCloud (sasiba): se descarga por WebDAV con el código del enlace y la clave."""
    if "/index.php/s/" in DATA_URL:
        servidor, codigo = DATA_URL.rstrip("/").split("/index.php/s/")
        credencial = base64.b64encode(f"{codigo}:{CLAVE_DATOS}".encode()).decode()
        return f"{servidor}/public.php/webdav/{nombre}.zip", {"Authorization": f"Basic {credencial}"}
    return f"{DATA_URL.rstrip('/')}/{nombre}.zip", {}


def obtener_datos(nombre):
    """Devuelve la carpeta local del conjunto de datos `nombre`; si no existe, la descarga."""
    for base in (Path("data"), Path("../data"), Path("/content/data")):
        if (base / nombre).is_dir():
            return base / nombre
    destino = Path("/content/data") if Path("/content").exists() else Path("data")
    destino.mkdir(parents=True, exist_ok=True)
    archivo = destino / f"{nombre}.zip"
    url, cabeceras = _url_y_cabeceras(nombre)
    print(f"Descargando {nombre}.zip ...")
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=cabeceras)) as r, open(archivo, "wb") as f:
            shutil.copyfileobj(r, f)
    except urllib.error.HTTPError as e:
        archivo.unlink(missing_ok=True)
        if e.code == 401:
            raise RuntimeError("Clave incorrecta o vacía: escribe en CLAVE_DATOS la clave que entregó el equipo docente.") from None
        raise
    with zipfile.ZipFile(archivo) as z:
        z.extractall(destino)
    archivo.unlink()
    return destino / nombre
