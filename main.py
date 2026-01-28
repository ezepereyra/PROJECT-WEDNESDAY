import pandas as pd
import os
from datetime import datetime
import logging

# Crear carpeta logs
os.makedirs("logs", exist_ok=True)

# Timestamp válido
fecha = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
nombre_log = f"log_{fecha}.txt"
ruta_log = f"logs/{nombre_log}"

# Configurar logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(lineno)s - %(message)s",
    handlers=[
        logging.FileHandler(ruta_log, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Función para cargar el dataset
def cargar_dataset(path: str) -> pd.DataFrame | None:
    logger.info(f"Cargando dataset desde {path}")
    try:
        df = pd.read_csv(path)
        logger.info(f"Dataset cargado correctamente con {df.shape[0]} filas y {df.shape[1]} columnas")
        return df
    except Exception as e:
        logger.exception(f"Error al cargar el dataset {path}: {e}")
        raise

# Función principal
def main():

    logger.info("Inicio de ejecución")

    path = "data/competencia_01_crudo_filtrado.csv"

    df = cargar_dataset(path)

    logger.debug(f"Primeras filas:\n{df.head()}")

    logger.info(f"Fin de ejecución. Logs en {ruta_log}")


if __name__ == "__main__":
    main()

