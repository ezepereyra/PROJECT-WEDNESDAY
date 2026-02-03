import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Funcion para cargar el dataset
def cargar_dataset(path: str) -> pd.DataFrame | None:
    logger.info(f"Cargando dataset desde {path}")
    try:
        df = pd.read_csv(path)
        logger.info(f"Dataset cargado correctamente con {df.shape[0]} filas y {df.shape[1]} columnas")
        return df
    except Exception as e:
        logger.exception(f"Error al cargar el dataset {path}: {e}")
        raise