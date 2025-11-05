import pandas as pd
import numpy as np
from util.load_dataframes import load_dataframe_vacas


def fill_missing_data(file_path: str) -> pd.DataFrame:
    df = load_dataframe_vacas(file_path)
    
    # Crear una copia para evitar modificar el dataframe original
    df_filled = df.copy()
    
    # Ordenar por ID de vaca y fecha para facilitar el cálculo de ventanas
    df_filled = df_filled.sort_values([('ID', 'ID Vaca'), ('Fecha y hora de inicio', 'fecha')])
    df_filled = df_filled.reset_index(drop=True)
    
    # Definir las columnas a rellenar con el promedio de la vaca
    flow_production_columns = [
        ('Media de los flujos (kg/min)', 'TI'),
        ('Media de los flujos (kg/min)', 'TD'),
        ('Media de los flujos (kg/min)', 'DI'),
        ('Media de los flujos (kg/min)', 'DD'),
        ('Flujos maximos (kg/min)', 'TI'),
        ('Flujos maximos (kg/min)', 'TD'),
        ('Flujos maximos (kg/min)', 'DI'),
        ('Flujos maximos (kg/min)', 'DD'),
        ('Producciones (kg)', 'TI'),
        ('Producciones (kg)', 'TD'),
        ('Producciones (kg)', 'DI'),
        ('Producciones (kg)', 'DD')
    ]
    
    conductivity_columns = [
        ('Conductividad (mS / cm)', 'TI'),
        ('Conductividad (mS / cm)', 'TD'),
        ('Conductividad (mS / cm)', 'DI'),
        ('Conductividad (mS / cm)', 'DD')
    ]
    
    blood_columns = [
        ('Sangre (ppm)', 'TI'),
        ('Sangre (ppm)', 'TD'),
        ('Sangre (ppm)', 'DI'),
        ('Sangre (ppm)', 'DD')
    ]
    
    # Función auxiliar para calcular el promedio en ventana de ±3 días
    def fill_with_window(df, col, window_days=3):
        """Rellena valores NaN usando el promedio de la vaca en una ventana de ±window_days días"""
        filled_values = df[col].copy()
        
        # Encontrar índices con valores NaN
        nan_indices = df[df[col].isna()].index
        
        for idx in nan_indices:
            cow_id = df.loc[idx, ('ID', 'ID Vaca')]
            date = df.loc[idx, ('Fecha y hora de inicio', 'fecha')]
            
            # Definir ventana de fechas (±3 días)
            date_min = date - pd.Timedelta(days=window_days)
            date_max = date + pd.Timedelta(days=window_days)
            
            # Filtrar datos de la misma vaca dentro de la ventana de tiempo
            mask = (
                (df[('ID', 'ID Vaca')] == cow_id) & 
                (df[('Fecha y hora de inicio', 'fecha')] >= date_min) & 
                (df[('Fecha y hora de inicio', 'fecha')] <= date_max) &
                (df[col].notna())  # Solo valores no-NaN
            )
            
            window_data = df.loc[mask, col]
            
            # Si hay datos en la ventana, usar el promedio
            if len(window_data) > 0:
                filled_values.loc[idx] = window_data.mean()
        
        return filled_values
    
    # Rellenar Flujos y Producciones con el promedio de la vaca en ventana de ±3 días
    for col in flow_production_columns:
        if col in df_filled.columns:
            # Rellenar con ventana de ±3 días
            df_filled[col] = fill_with_window(df_filled, col, window_days=3)
            # Si todavía hay NaN (la vaca no tiene datos en la ventana), usar el promedio general de la vaca
            cow_means = df_filled.groupby(('ID', 'ID Vaca'))[[col]].transform('mean')
            df_filled[col] = df_filled[col].fillna(cow_means[col])
            # Si todavía hay NaN (la vaca no tiene datos válidos), rellenar con la media general de la columna
            df_filled[col] = df_filled[col].fillna(df_filled[col].mean())
    
    # Rellenar Conductividad con el promedio de la vaca en ventana de ±3 días
    for col in conductivity_columns:
        if col in df_filled.columns:
            # Rellenar con ventana de ±3 días
            df_filled[col] = fill_with_window(df_filled, col, window_days=3)
            # Si todavía hay NaN (la vaca no tiene datos en la ventana), usar el promedio general de la vaca
            cow_means = df_filled.groupby(('ID', 'ID Vaca'))[[col]].transform('mean')
            df_filled[col] = df_filled[col].fillna(cow_means[col])
            # Si todavía hay NaN (la vaca no tiene datos válidos), rellenar con la media general de la columna
            df_filled[col] = df_filled[col].fillna(df_filled[col].mean())
    
    # Rellenar Sangre (ppm) con 0
    for col in blood_columns:
        if col in df_filled.columns:
            df_filled[col] = df_filled[col].fillna(0)
    
    return df_filled


if __name__ == "__main__":
    # Ejemplo de uso
    file_path = "datos/registros_sesiones_merged.csv"
    df_filled = fill_missing_data(file_path)
    print(f"\nForma final del dataframe: {df_filled.shape}")
    print(f"Valores faltantes restantes: {df_filled.isna().sum().sum()}")
