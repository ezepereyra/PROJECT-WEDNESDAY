import pandas as pd
import os
from datetime import datetime
import logging
# cargo las funciones en el archivo loader dentro de src
# para eso necesito el archivo __init__.py en src
from src.loader import cargar_dataset, convertir_clase_ternaria_a_target, crear_clase_ternaria
from src.features import feature_engineering_lag
from src.optimization import optimizar, evaluar_en_test
from src.best_params import cargar_los_mejores_hiperparametros
from src.final_training import preparar_datos_entrenamiento_final, entrenar_modelo_final, generar_predicciones_finales, guardar_predicciones_finales, guardar_modelo_final
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

# Manejo de configuración en YAML
logger.info("Configuración cargada desde YAML")
logger.info(f"STUDY_NAME: {STUDY_NAME}")
logger.info(f"SEMILLA: {SEMILLA}")
logger.info(f"DATA_PATH: {DATA_PATH}")
logger.info(f"MES_TRAIN: {MES_TRAIN}")
logger.info(f"MES_VALIDACION: {MES_VALIDACION}")
logger.info(f"MES_TEST: {MES_TEST}")
logger.info(f"GANANCIA_ACIERTO: {GANANCIA_ACIERTO}")
logger.info(f"COSTO_ESTIMULO: {COSTO_ESTIMULO}")

# Función principal
def main():
    # Cargar datos 
    logger.info("Inicio de ejecución")
    os.makedirs("data", exist_ok=True)
    df = cargar_dataset(DATA_PATH)

    # Crear clase ternaria
    df = crear_clase_ternaria(df)

    # Feature engineering
    columnas = ["ctrx_quarter", "mrentabilidad", "mcuentas_saldo", "mtarjeta_visa_consumo", "cproductos"]
    cant_lag = 2
    df_fe = feature_engineering_lag(df, columnas=columnas, cant_lag=cant_lag)
    logger.info(f"Feature engineering completado con {cant_lag} lags para {len(columnas) if columnas else 0} atributos")

    # Convertir clase ternaria a target binaria
    df_fe = convertir_clase_ternaria_a_target(df_fe)

    """
    # Guardar los datos 
    path = "data/competencia_01_crudo_filtrado_lag_binaria.csv"
    df_fe.to_csv(path, index=False)
    logger.info(f"Datos guardados en {path}")
    """

    # Ejecutar la optimización de hiperparámetros
    study = optimizar(df_fe, n_trials=100)

    # Análisis adicional
    logger.info("===ANÁLISIS DE RESULTADOS===")
    trials_df = study.trials_dataframe()
    if len(trials_df) > 0:
        top_5 = trials_df.nlargest(5, "value")
        logger.info("Top 5 mejores trials:")
        for idx, trial in top_5.iterrows():
            logger.info(f"Trial {trial['number']}: {trial['value']:,.0f}")

    logger.info("===OPTIMIZACIÓN COMPLETADA===")
    logger.info(f"Mejores hiperparámetros: {study.best_params}")
    logger.info(f"Ganancia en validación: {study.best_value:,.0f}")
    logger.info("===EVALUACIÓN EN EL CONJUNTO DE TEST===")
    mejores_params = cargar_los_mejores_hiperparametros()
    logger.info(f"Mejores hiperparámetros cargados: {mejores_params}")
    ganancia_test = evaluar_en_test(df_fe, mejores_params)
    logger.info(f"Ganancia en test: {ganancia_test:,.0f}")

    # Entrenar modelo final
    X_train, y_train, X_predict, clientes_predict = preparar_datos_entrenamiento_final(df_fe)
    modelo = entrenar_modelo_final(X_train, y_train, mejores_params)

    # Guardar el modelo entrenado (podría ser útil para futuras predicciones) como .txt
    guardar_modelo_final(modelo)

    # Generar predicciones finales
    predicciones = generar_predicciones_finales(modelo, X_predict, clientes_predict)

    # Guardar predicciones finales
    salida = guardar_predicciones_finales(predicciones)
    logger.info(f"Predicciones guardadas en {salida}")
    
    logger.info(f"=== FIN DE EJECUCIÓN ===. Revisar logs para más detalle. {nombre_log}")

if __name__ == "__main__":
    main()

