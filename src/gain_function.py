import numpy as np
import pandas as pd
from .conf import GANANCIA_ACIERTO, COSTO_ESTIMULO
import logging

logger = logging.getLogger(__name__)

def calcular_ganancia(y_true, y_pred):
    """
    Calcula la ganancia total usando las funciones de ganancia de la competencia.
    
    Args:
        y_true: Array con las etiquetas reales (0 o 1)
        y_pred: Array con las predicciones binarias (0 o 1)
        
    Returns:
        float: Ganancia total
    """
    # Convertir a arrays numpy si no lo son
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
    
    # Calcular ganancia vectorizada usando config
    # Verdaderos positivos: y_true = 1 y y_pred = 1 -> ganancia
    # Falsos positivos: y_true = 0 y y_pred = 1 -> costo
    # Verdaderos negativos y falsos negativos: ganancia = 0

    ganancia_total = np.sum(
        ((y_true == 1) & (y_pred == 1)) * GANANCIA_ACIERTO +
        ((y_true == 0) & (y_pred == 1)) * (-COSTO_ESTIMULO)
    )

    logger.debug(f"Ganancia total calculada: {ganancia_total:,0f}"
        f"(GANANCIA_ACIERTO: {GANANCIA_ACIERTO}, COSTO_ESTIMULO: {COSTO_ESTIMULO})")
    
    return ganancia_total

def ganancia_lgb_binary(y_pred, y_true):
    """
    Función de ganancia para LightGBM con predicciones binarias.
    Compatible con callbacks de LightGBM.
    
    Args:
        y_pred: Array con las predicciones binarias (0 o 1)
        y_true: Array con las etiquetas reales (0 o 1)
        
    Returns:
        tuple: (eval_name, eval_result, is_higher_better)
    """
    # Obtener labels verdaderos
    y_true_labels = y_true.get_label()

    # Convertir probabilidades a predicciones binarias (umbral 0.025)
    y_pred_binary = (y_pred >= 0.025).astype(int)
    
    # Calcular ganancia usando configuración
    ganancia_total = calcular_ganancia(y_true_labels, y_pred_binary)
    
    # Retornar tuple para LightGBM
    return "ganancia", ganancia_total, True # True = higher is better

