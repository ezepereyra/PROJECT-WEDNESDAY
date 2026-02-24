import json
from src.conf import *
import logging

logger = logging.getLogger(__name__)

def cargar_los_mejores_hiperparametros(archivo_base=None):
    """
    Carga los mejores hiperparámetros desde el archivo JSON de iteraciones de Optuna.
    
    Args:
        archivo_base: Nombre base (si es None, usa STUDY_NAME)
    
    Returns:
        dict: Mejores hiperparámetros encontrados
    """
    if archivo_base is None:
        archivo_base = STUDY_NAME
    
    archivo = f"resultados_{archivo_base}_iteraciones.json"
    
    try:
        with open(archivo, "r") as f:
            iteraciones = json.load(f)
            
        if not iteraciones: 
            raise ValueError("No se encontraron iteraciones en el archivo JSON")

        # Encontrar la iteración con la mayor ganancia
        mejor_iteracion = max(iteraciones, key=lambda x: x["value"])
        mejores_params = mejor_iteracion["params"]
        mejor_ganancia = mejor_iteracion["value"]

        logger.info(f"Mejores hiperparámetros cargados desde {archivo}")
        logger.info(f"Mejor ganancia encontrada: {mejor_ganancia}")
        logger.info(f"Trial número: {mejor_iteracion['trial_number']}")
        logger.info(f"Hiperparámetros: {mejores_params}")
        
        return mejores_params

    except FileNotFoundError:
        logger.error(f"El archivo {archivo} no existe")
        logger.error("Asegúrate de haber ejecutado la optimización antes de cargar los mejores hiperparámetros")
        raise

    except Exception as e:
        logger.error(f"Error al cargar los mejores hiperparámetros: {str(e)}")
        raise

def obtener_estadisticas_optuna(archivo_base=None):
    """
    Obtiene estadísticas de la optimización de Optuna. 

    Args:
        archivo_base: Nombre base (si es None, usa STUDY_NAME)
    
    Returns:
        dict: Estadísticas de la optimización
    """

    if archivo_base is None:
        archivo_base = STUDY_NAME
    
    archivo = f"resultados_{archivo_base}_iteraciones.json"

    try:
        with open(archivo, 'r') as f:
            iteraciones = json.load(f)
            
        ganancias = [iter['value'] for iter in iteraciones]

        estadisticas = {
            'total_trials': len(iteraciones),
            'mejor_ganancia': max(ganancias),
            'peor_ganancia': min(ganancias),
            'ganancia_media': sum(ganancias) / len(ganancias),
            'top_5_trials': sorted(iteraciones, key=lambda x: x['value'], reverse=True)[:5]
        }

        logger.info("Estadísticas de la optimización:")
        logger.info(f"Total de trials: {estadisticas['total_trials']}")
        logger.info(f"Mejor ganancia: {estadisticas['mejor_ganancia']}")
        logger.info(f"Peor ganancia: {estadisticas['peor_ganancia']}")
        logger.info(f"Ganancia media: {estadisticas['ganancia_media']}")

        return estadisticas

    except Exception as e:
        logger.error(f"Error al obtener estadísticas de Optuna: {str(e)}")
        raise