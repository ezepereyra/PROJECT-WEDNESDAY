import pandas as pd
import os
from datetime import datetime
import logging
# cargo las funciones en el archivo loader dentro de src
# para eso necesito el archivo __init__.py en src
from src.loader import cargar_dataset, convertir_clase_ternaria_a_target, crear_clase_ternaria
from src.features import feature_engineering_lag
from src.conf import *

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

    # Crear clase ternaria
    df = crear_clase_ternaria(df)

    # Feature engineering
    columnas = ["ctrx_quarter", "mrentabilidad", "mcuentas_saldo", "mtarjeta_visa_consumo", "cproductos"]
    cant_lag = 2
    df_fe = feature_engineering_lag(df, columnas=columnas, cant_lag=cant_lag)
    logger.info(f"Feature engineering completado con {cant_lag} lags para {len(columnas) if columnas else 0} atributos")

    # Convertir clase ternaria a target binaria
    df_fe = convertir_clase_ternaria_a_target(df_fe)

    # Guardar los datos 
    path = "data/competencia_01_crudo_filtrado_lag_binaria.csv"
    df_fe.to_csv(path, index=False)
    logger.info(f"Datos guardados en {path}")

    logger.info(f"Fin de ejecución. Logs en {ruta_log}")


    print("SEMILLA:", SEMILLA)
    print("DATA_PATH:", DATA_PATH)
    print("MES_TRAIN:", MES_TRAIN)
    print("MES_VALIDACION:", MES_VALIDACION)
    print("MES_TEST:", MES_TEST)
    print("GANANCIA_ACIERTO:", GANANCIA_ACIERTO)
    print("COSTO_ESTIMULO:", COSTO_ESTIMULO)


if __name__ == "__main__":
    main()

