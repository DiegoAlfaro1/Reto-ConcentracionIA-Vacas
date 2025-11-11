import pandas as pd
import numpy as np

def fill_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    
    # Crear una copia para evitar modificar el dataframe original
    df_original = df.copy()
    
    # Separar filas de "Ordeño" y "Rechazada"
    mask_ordeno = df_original[('Main', 'Accion')] == 'Ordeño'
    df_ordeno = df_original[mask_ordeno].copy()
    df_rechazada = df_original[~mask_ordeno].copy()
    
    print(f"Total de filas: {len(df_original)}")
    print(f"Filas con 'Ordeño' (se rellenarán): {len(df_ordeno)}")
    print(f"Filas con 'Rechazada' (se mantendrán sin cambios): {len(df_rechazada)}\n")
    
    # Ordenar por ID de vaca y fecha para facilitar el cálculo de ventanas (solo filas de Ordeño)
    df_ordeno = df_ordeno.sort_values([('ID', 'ID Vaca'), ('Fecha y hora de inicio', 'fecha')])
    df_ordeno = df_ordeno.reset_index(drop=True)
    
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
    
    # Reemplazar 0 con NaN en columnas de flujo/producción y conductividad (solo en filas de Ordeño)
    for col in flow_production_columns:
        if col in df_ordeno.columns:
            df_ordeno[col] = df_ordeno[col].replace(0, np.nan)
    
    for col in conductivity_columns:
        if col in df_ordeno.columns:
            df_ordeno[col] = df_ordeno[col].replace(0, np.nan)
    
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
    
    # Rellenar Flujos y Producciones con el promedio de la vaca en ventana de ±3 días (solo filas de Ordeño)
    for col in flow_production_columns:
        if col in df_ordeno.columns:
            # Rellenar con ventana de ±3 días
            df_ordeno[col] = fill_with_window(df_ordeno, col, window_days=3)
            # Si todavía hay NaN (la vaca no tiene datos en la ventana), usar el promedio general de la vaca
            cow_means = df_ordeno.groupby(('ID', 'ID Vaca'))[[col]].transform('mean')
            df_ordeno[col] = df_ordeno[col].fillna(cow_means[col])
            # Si todavía hay NaN (la vaca no tiene datos válidos), rellenar con la media general de la columna
            df_ordeno[col] = df_ordeno[col].fillna(df_ordeno[col].mean())
    
    # Rellenar Conductividad con el promedio de la vaca en ventana de ±3 días (solo filas de Ordeño)
    for col in conductivity_columns:
        if col in df_ordeno.columns:
            # Rellenar con ventana de ±3 días
            df_ordeno[col] = fill_with_window(df_ordeno, col, window_days=3)
            # Si todavía hay NaN (la vaca no tiene datos en la ventana), usar el promedio general de la vaca
            cow_means = df_ordeno.groupby(('ID', 'ID Vaca'))[[col]].transform('mean')
            df_ordeno[col] = df_ordeno[col].fillna(cow_means[col])
            # Si todavía hay NaN (la vaca no tiene datos válidos), rellenar con la media general de la columna
            df_ordeno[col] = df_ordeno[col].fillna(df_ordeno[col].mean())
    
    # Rellenar Sangre (ppm) con 0 (solo filas de Ordeño)
    for col in blood_columns:
        if col in df_ordeno.columns:
            df_ordeno[col] = df_ordeno[col].fillna(0)
    
    # Combinar de nuevo las filas de Ordeño (rellenadas) con las de Rechazada (sin cambios)
    df_final = pd.concat([df_ordeno, df_rechazada], ignore_index=False)
    
    # Ordenar por el índice original para mantener el orden
    df_final = df_final.sort_index()
    df_final = df_final.reset_index(drop=True)
    
    print(f"Total de filas en el dataset final: {len(df_final)}")
    
    return df_final


# if __name__ == "__main__":
#     # Ejemplo de uso
#     file_path = "datos/registros_sesiones_merged.csv"
    
#     # Cargar datos originales (después de load_dataframe_vacas)
#     df_original = load_dataframe_vacas(file_path)
    
#     # Exportar CSV de datos originales
#     output_path_original = "datos/datos_despues_de_load.csv"
#     df_original.to_csv(output_path_original, index=False)
#     print(f"✓ Datos originales exportados a: {output_path_original}\n")
    
#     # Aplicar fill_missing_data
#     print("Aplicando fill_missing_data...")
#     df_filled = fill_missing_data(file_path)

#     # Exportar CSV de datos rellenados
#     output_path_filled = "datos/datos_despues_de_fill.csv"
#     df_filled.to_csv(output_path_filled, index=False)
#     print(f"✓ Datos rellenados exportados a: {output_path_filled}")
