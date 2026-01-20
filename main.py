import pandas as pd
import os
import dateline

def main():
    print("Inicio de ejecución: ")
    #cargar datos
    try:
        df = pd.read_csv("data/competencia_01_crudo.csv")
    except Exception as e:
        print(f"Error al cargar el dataset: {e}")
        return
    
    #mostrar datos
    print(df.head())
    print(f"Filas: {df.shape[0]}")
    print(f"Columnas: {df.shape[1]}")
    print("Dataset cargado")

    with open("logs/logs.txt", "a", enconding = "utf-8") as f:
        f.write(f"{dateline.dateline.now()} - Dataset cargado con {df.shape[0]} filas y {df.shape[1]} columnas\n")

    print("Fin de ejecución. Revisar logs/logs.txt")

if __name__ == "__main__":
    main()  