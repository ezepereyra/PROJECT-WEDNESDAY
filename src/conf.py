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
        DATA_PATH = _cfg.get("DATA_PATH","../data/competencia.csv")
        SEMILLA = _cfg.get("SEMILLA",[42])
        MES_TRAIN = _cfg.get("MES_TRAIN","202102")
        MES_VALIDACION = _cfg.get("MES_VALIDACION","202103")
        MES_TEST = _cfg.get("MES_TEST","202104")
        GANANCIA_ACIERTO = _cfg.get("GANANCIA_ACIERTO",None)
        COSTO_ESTIMULO = _cfg.get("COSTO_ESTIMULO",None)

except Exception as e:
    logger.exception(f"Error al cargar el archivo de configuración {PATH_CONFIG}: {e}")
    raise
