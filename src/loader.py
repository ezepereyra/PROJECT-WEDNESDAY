import pandas as pd
import logging
import numpy as np

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

def crear_clase_ternaria(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea la clase ternaria CONTINUA, BAJA+1 o BAJA+2
    """
    logger.info("Creando clase ternaria")

    required = {"numero_de_cliente", "foto_mes"}

    if not required.issubset(df.columns):
        raise ValueError(f"Faltan columnas: {required - set(df.columns)}")

    df_out = df.copy()
    df_out["_pos"] = df_out.index

    # Ordenar
    df_out = df_out.sort_values(
        ["numero_de_cliente", "foto_mes"]
    )

    grp = df_out.groupby("numero_de_cliente")

    # Próximo mes
    next_mes = grp["foto_mes"].shift(-1)
    next2_mes = grp["foto_mes"].shift(-2)

    # Diferencias
    diff1 = next_mes - df_out["foto_mes"]
    diff2 = next2_mes - df_out["foto_mes"]

    df_out["clase_ternaria"] = np.nan

    # CONTINUA: tiene continuidad normal
    df_out.loc[diff1 == 1, "clase_ternaria"] = "CONTINUA"

    # BAJA+1: falta el siguiente mes
    df_out.loc[
        (diff1.isna()) | (diff1 > 1),
        "clase_ternaria"
    ] = "BAJA+1"

    # BAJA+2: falta dos meses seguidos
    df_out.loc[
        (diff1 == 1) & ((diff2.isna()) | (diff2 > 2)),
        "clase_ternaria"
    ] = "BAJA+2"

    # Restaurar orden
    df_out = df_out.sort_values("_pos").drop(columns="_pos")

    logger.info("Clase ternaria creada")

    return df_out
    

def convertir_clase_ternaria_a_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte la clase ternaria a target binario reemplazando en el mismo atributo: 
    - CONTINUA -> 0
    - BAJA+1 -> 1
    - BAJA+2 -> 1

    Args: 
        df: DataFrame con columna 'clase_ternaria'
    Returns: 
        df: DataFrame con columna 'clase_ternaria' convertida a valores binarios (0, 1)
    """

    logger.info("Convertiendo clase ternaria a target binaria")

    df_results = df.copy()

    n_continua_orig = (df_results['clase_ternaria'] == "CONTINUA").sum()
    n_baja1_orig = (df_results['clase_ternaria'] == "BAJA+1").sum()
    n_baja2_orig = (df_results['clase_ternaria'] == "BAJA+2").sum()

    # Converitr clase_ternaria a binaria mapeando
    df_results['clase_ternaria'] = df_results['clase_ternaria'].map({
        "CONTINUA": 0, 
        "BAJA+1": 1, 
        "BAJA+2": 1
    })

    # Log de la conversión 
    n_ceros = (df_results['clase_ternaria'] == 0).sum()
    n_unos = (df_results['clase_ternaria'] == 1).sum()

    logger.info("Conversión completada")
    logger.info(f"Original - CONTINUA: {n_continua_orig}, BAJA+1: {n_baja1_orig}, BAJA+2: {n_baja2_orig}")
    logger.info(f"Binario - 0: {n_ceros}, 1: {n_unos}")
    logger.info(f"Distribución : {n_unos / (n_ceros + n_unos) * 100:.2f}% casos positivos")
    
    return df_results