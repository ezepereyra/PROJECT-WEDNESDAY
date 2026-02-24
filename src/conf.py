import yaml
import os
import logging

logger = logging.getLogger(__name__)

# Ruta de archivo de configuración
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH_CONFIG = os.path.join(BASE_DIR, "conf.yaml")

try: 
    with open(PATH_CONFIG, "r") as f:
        _cfgGeneral = yaml.safe_load(f)
        _cfg = _cfgGeneral["competencia01"]

        # Valores por defecto ? 
        STUDY_NAME = _cfgGeneral.get("STUDY_NAME","Wednesday")
        PARAMETROS_LGB = _cfgGeneral["parametros_lgb"]
        DATA_PATH = os.path.join(
            BASE_DIR,
            _cfg.get("DATA_PATH", "data/competencia.csv")
        )
        SEMILLA = _cfg.get("SEMILLA",[42])
        MES_TRAIN = _cfg.get("MES_TRAIN",[])
        MES_VALIDACION = _cfg.get("MES_VALIDACION",[])
        MES_TEST = _cfg.get("MES_TEST",[])
        GANANCIA_ACIERTO = _cfg.get("GANANCIA_ACIERTO",None)
        COSTO_ESTIMULO = _cfg.get("COSTO_ESTIMULO",None)
        FINAL_TRAIN = _cfg.get("FINAL_TRAIN", [])
        FINAL_PREDICT = _cfg.get("FINAL_PREDICT",[])

except Exception as e:
    logger.exception(f"Error al cargar el archivo de configuración {PATH_CONFIG}: {e}")
    raise
