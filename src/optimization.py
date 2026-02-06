import optuna
import pandas as pd
import lightgbm as lgb
import numpy as np
import logging
import json
import os
from datetime import datetime
from .conf import *
from .gain_function import calcular_ganancia, ganancia_lgb_binary

def objetivo_ganancia(trial,df) -> float:
    """
    Parameters:
        trial: Trial de Optuna
        df: DataFrame con los datos

    Description:
    Función objetivo que maximiza ganancia en mes de validación
    Utiliza conf YAML para periodos y semilla
    Define parámetros para el modelo LightGBM
    Preparar dataset para entrenamiento y validación
    Entrenar modelo con función de ganancia personalizada
    Predecir y calcular ganancia
    Guardar cada iteración en JSON

    Returns:
        float: Ganancia total
    """
    # Hiperparámetros a optimizar 
    params = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "metric": "None",
        "num_leaves": trial.suggest_int("num_leaves", PARAMETROS_LGB['num_leaves'][0], PARAMETROS_LGB['num_leaves'][1]),
        "learning_rate": trial.suggest_float("learning_rate", PARAMETROS_LGB['learning_rate'][0], PARAMETROS_LGB['learning_rate'][1]),
        "feature_fraction": trial.suggest_float("feature_fraction", PARAMETROS_LGB['feature_fraction'][0], PARAMETROS_LGB['feature_fraction'][1]),
        "bagging_fraction": trial.suggest_float("bagging_fraction", PARAMETROS_LGB['bagging_fraction'][0], PARAMETROS_LGB['bagging_fraction'][1]),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", PARAMETROS_LGB['min_data_in_leaf'][0], PARAMETROS_LGB['min_data_in_leaf'][1]),
        "max_depth": trial.suggest_int("max_depth", PARAMETROS_LGB['max_depth'][0], PARAMETROS_LGB['max_depth'][1]),
        "lambda_l1": trial.suggest_float("lambda_l1", PARAMETROS_LGB['lambda_l1'][0], PARAMETROS_LGB['lambda_l1'][1]),
        "lambda_l2": trial.suggest_float("lambda_l2", PARAMETROS_LGB['lambda_l2'][0], PARAMETROS_LGB['lambda_l2'][1]),
        "min_gain_to_split": 0.0,
        "verbose": -1,
        "silent": True,
        "bin": 31,
        "random_state": SEMILLA[0]
    }

    # Preparar datos usando conf YAML
    if isinstance(MES_TRAIN, list):
        df_train = df[df["foto_mes"].isin(MES_TRAIN)]
    else:
        df_train = df[df["foto_mes"] == MES_TRAIN]
    
    df_val = df[df["foto_mes"] == MES_VALIDACION]

    X_train = df_train.drop(columns=["clase_ternaria"])
    y_train = df_train["clase_ternaria"].values

    X_val = df_val.drop(columns=["clase_ternaria"])
    y_val = df_val["clase_ternaria"].values

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        params, 
        train_data, 
        valid_sets=[val_data],
        feval = ganancia_lgb_binary, # Función de ganancia personalizada
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)]
    )

    # Predecir y calcular ganancia
    y_pred_proba = model.predict(X_val)
    y_pred_binary = (y_pred_proba >= 0.025).astype(int)

    ganancia_total = calcular_ganancia(y_val, y_pred_binary)

    # Guardar cada iteración en JSON 
    guardar_iteracion(trial, ganancia_total)

    logger.info(f"Trial {trial.number}: Ganancia = {ganancia_total}")

    return ganancia_total


def guardar_iteracion(trial, ganancia, archivo_base=None):
    """
    Guarda cada iteración de la optimización en un único archivo JSON

    Args: 
    trial: Trial de Optuna
    ganancia: Valor de ganancia obtenido
    archivo_base: Nombre base del archivo (si es None, usa el de config.yaml)
    """
    if archivo_base is None: 
        archivo_base = STUDY_NAME

    # Nombre del archivo único para todas las iteraciones
    archivo = f"resultados_{archivo_base}_iteraciones.json"
    
    #Datos de esta iteración
    iteracion_data = {
        "trial_number": trial.number,
        "params": trial.params,
        "value": float(ganancia),
        "datetime": datetime.now().isoformat(),
        "state": 'COMPLETE', # si llega aquí es porque terminó exitosamente
        "configuración": {
            "semilla": SEMILLA,
            "mes_train": MES_TRAIN,
            "mes_validación": MES_VALIDACION
        }
    }
    
    # Cargar datos existentes si el archivo ya existe 
    if os.path.exists(archivo):
        with open(archivo, "r") as f:
            try: 
                datos_existentes = json.load(f)
                if not isinstance(datos_existentes, list):
                    datos_existentes = []
            except json.JSONDecodeError:
                datos_existentes = []
    else:
        datos_existentes = []


    # Agregar una nueva iteración
    datos_existentes.append(iteracion_data)
    
    # Guardar el archivo
    with open(archivo, "w") as f:
        json.dump(datos_existentes, f, indent=2)

    logger.info(f"Iteración {trial.number} guardada en {archivo}")
    logger.info(f"Ganancia: {ganancia:,}" + "---" + f"Parámetros:{trial.params}")
    

from src.conf import STUDY_NAME

def optimizar(df: pd.DataFrame, n_trials: int, study_name: str = None) -> optuna.Study:
    """
    Args:
        df: DataFrame con los datos
        n_trial: Número de trials para la optimización
        study_name: Nombre del study

    Descripción:
        Ejecuta optimización bayesiana de hiperparámetros usando configuración YAML
        Guarda cada iteración en un archivo JSON separado
        Pasos:
        1. Crear estudio de Optuna
        2. Ejecutar optimización
        3. Retornar estudio

    Returns:
        optuna.Study: Estudio de Optuna con resultados
    """

    study_name = STUDY_NAME

    logger.info(f"Iniciando optimización con {n_trials} trials")
    logger.info(f"Configuración: TRIAN = {MES_TRAIN}, VALID = {MES_VALIDACION}, SEMILLA = {SEMILLA}")

    study = optuna.create_study(direction="maximize", study_name=study_name)

    # Función objetivo parcial con datos 
    objetive_with_data = lambda trial : objetivo_ganancia(trial, df)

    # Ejecutar optimización
    study.optimize(
        objetive_with_data,
        n_trials=n_trials,
        show_progress_bar=True
    )

    # Resultados
    logger.info(f"Mejor ganancia: {study.best_value:,.0f}")
    logger.info(f"Mejores hiperparámetros: {study.best_params}")
    
    return study
    