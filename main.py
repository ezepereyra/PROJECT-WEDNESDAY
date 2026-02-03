import pandas as pd
import os
from datetime import datetime
import logging
# cargo las funciones en el archivo loader dentro de src
# para eso necesito el archivo __init__.py en src
from src.loader import cargar_dataset 
from src.features import feature_engineering_lag

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

# Función principal
def main():
    # Cargar datos 
    logger.info("Inicio de ejecución")
    os.makedirs("data", exist_ok=True)
    path = "data/competencia_01_crudo_filtrado.csv"
    df = cargar_dataset(path)

    # Feature engineering
    columnas = ["ctrx_quarter", "mrentabilidad"]
    cant_lag = 2
    df = feature_engineering_lag(df, columnas=columnas, cant_lag=cant_lag)

    # Guardar los datos 
    path = "data/competencia_01_crudo_filtrado_lag.csv"
    df.to_csv(path, index=False)
    logger.info(f"Datos guardados en {path}")

    logger.info(f"Fin de ejecución. Logs en {ruta_log}")


if __name__ == "__main__":
    main()

