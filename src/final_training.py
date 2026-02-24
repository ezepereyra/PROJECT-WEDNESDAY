import pandas as pd
import lightgbm as lgb
import numpy as np
import logging
import os
from datetime import datetime
from .conf import FINAL_TRAIN, FINAL_PREDICT, SEMILLA
from .best_params import cargar_los_mejores_hiperparametros
from .gain_function import calcular_ganancia, ganancia_lgb_binary

logger = logging.getLogger(__name__)

def preparar_datos_entrenamiento_final(df):
    """
    Prepara los datos para el entrenamiento final usando todos los meses de FINAL_TRAIN

    Args:
        df: DataFrame con los datos
    
    Returns:
        tuple: (X_train, y_train, X_predict, clientes_predict)
    """
    logger.info(f"Preparando datos para el entrenamiento final usando los meses {FINAL_TRAIN}")
    logger.info(f"Preparando datos para la predicción usando los meses {FINAL_PREDICT}")
    
    # Filtrar los datos para el entrenamiento final
    X_train = df[df['foto_mes'].isin(FINAL_TRAIN)]
    y_train = X_train['clase_ternaria']
    X_train = X_train.drop('clase_ternaria', axis=1)
    
    # Filtrar los datos para la predicción
    X_predict = df[df['foto_mes'].isin(FINAL_PREDICT)]
    X_predict = X_predict.drop('clase_ternaria', axis=1)

    # Filtrar los datos para la predicción
    clientes_predict = X_predict['numero_de_cliente'].unique()
    
    return X_train, y_train, X_predict, clientes_predict


def entrenar_modelo_final(X_train, y_train, mejores_params):
    """
    Entrena el modelo final usando los mejores hiperparámetros encontrados

    Args:
        X_train: DataFrame con los datos de entrenamiento
        y_train: Array con las etiquetas de entrenamiento
        mejores_params: Diccionario con los mejores hiperparámetros encontrados por Optuna

    Returns:
        lgb.Booster: Modelo entrenado
    """
    logger.info("Iniciando entrenamiento del modelo final con los mejores hiperparámetros")

    # Configurar los parámetros del modelo
    params ={
        'objective': 'binary',
        'metric': None, # Usamos nuestra métrica personalizada
        'random_state': SEMILLA[0],
        'verbose': -1,
        **mejores_params
    }

    logger.info(f"Parámetros del modelo: {params}")

    # Crear el dataset de entrenamiento
    train_data = lgb.Dataset(X_train, label=y_train)

    # Estimar el modelo
    logger.info("Entrenando el modelo final")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=mejores_params.get("num_boost_round", 1000),
        feval=ganancia_lgb_binary
    )
    
    return model

def generar_predicciones_finales(modelo, X_predict, clientes_predict, umbral=0.025):
    """
    Genera las predicciones finales usando el modelo entrenado para el mes objetivo
    
    Args:
        modelo: Modelo entrenado
        X_predict: DataFrame con los datos de predicción
        clientes_predict: Array con los IDs de los clientes
        umbral: Umbral de probabilidad para clasificar como positivo (clasificación binaria)
    
    Returns:
        pd.DataFrame: DataFrame con numero_cliente y predict
    """
    logger.info("Generando predicciones finales")
    
    # Predecir probabilidades
    y_pred_proba = modelo.predict(X_predict)

    # binarizar la probabilidad de y_pred_proba
    predict = (y_pred_proba > umbral).astype(int)
    
    # Crear DataFrame con las predicciones
    predicciones = pd.DataFrame({
        'numero_de_cliente': clientes_predict,
        'Predicted': predict
    })
    
    logger.info(f"Predicciones generadas para {len(clientes_predict)} clientes")
    logger.info(f"Predicciones positivas: {predict.sum()}")
    logger.info(f"Predicciones negativas: {(1 - predict).sum()}")
    logger.info(f"Umbral de probabilidad: {umbral}")
    
    return predicciones

def guardar_predicciones_finales(predicciones, nombre_archivo=None):
    """
    Guarda las predicciones finales en un archivo CSV en la carpeta predict
    
    Args:
        predicciones: DataFrame con las predicciones
        nombre_archivo: Nombre del archivo (si es None, usa STUDY_NAME)

    Returns: 
        str: Ruta del archivo guardado
    """
    DIR_GUARDADO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    carpeta_predict = os.path.join(DIR_GUARDADO, "src","predict")
    
    if nombre_archivo is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        nombre_archivo = f"predicciones_{timestamp}.csv"
    
    ruta_archivo = os.path.join(carpeta_predict, nombre_archivo)
    
    # Guardar el archivo
    predicciones.to_csv(ruta_archivo, index=False, columns = ["numero_de_cliente","Predicted"])
    
    logger.info(f"Predicciones guardadas en: {ruta_archivo}")
    
    return ruta_archivo

def guardar_modelo_final(modelo, nombre_archivo=None):
    """
    Guarda el modelo entrenado en un archivo .txt
    
    Args:
        modelo: Modelo entrenado
        nombre_archivo: Nombre del archivo (si es None, usa STUDY_NAME)
    
    Returns:
        str: Ruta del archivo guardado
    """
    DIR_GUARDADO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    carpeta_modelos = os.path.join(DIR_GUARDADO, "src","models")
    
    if nombre_archivo is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        nombre_archivo = f"modelo_{timestamp}.txt"
    
    ruta_archivo = os.path.join(carpeta_modelos, nombre_archivo)
    
    # Guardar el archivo
    modelo.save_model(ruta_archivo)
    
    logger.info(f"Modelo guardado en: {ruta_archivo}")
    
    return ruta_archivo