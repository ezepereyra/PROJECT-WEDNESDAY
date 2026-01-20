import pandas as pd

def main():
    print("Inicio de ejecución: ")
    #cargar datos
    df = pd.read_csv("data/competencia_01_crudo.csv")
    #mostrar datos
    print(df.head())
    print(f"Filas: {df.shape[0]}")
    print(f"Columnas: {df.shape[1]}")

if __name__ == "__main__":
    main()  