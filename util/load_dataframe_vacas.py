import pandas as pd

def load_dataframe_vacas(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, header=[0,1], skiprows=[1])

    # O si necesitas crear la estructura manualmente, puedes usar esto:
    columns = pd.MultiIndex.from_tuples([
        ('ID', 'ID Vaca'),
        ('Main', 'Hora de inicio'),
        ('Main', 'Acción'),
        ('Main', 'Duración (mm:ss)'),
        ('Main', 'Producción (kg)'),
        ('Estado', 'Número de ordeño'),
        ('Estado', 'Patada'),
        ('Estado', 'Incompleto'),
        ('Estado', 'Pezones no encontrados'),
        ('Estado', 'Ubre'),
        ('Estado', 'Pezón'),
        ('Media de los flujos (kg/min)', 'DI'),
        ('Media de los flujos (kg/min)', 'DD'),
        ('Media de los flujos (kg/min)', 'TI'),
        ('Media de los flujos (kg/min)', 'TD'),
        ('Sangre (ppm)', 'DI'),
        ('Sangre (ppm)', 'DD'),
        ('Sangre (ppm)', 'TI'),
        ('Sangre (ppm)', 'TD'),
        ('Conductividad (mS / cm)', 'DI'),
        ('Conductividad (mS / cm)', 'DD'),
        ('Conductividad (mS / cm)', 'TI'),
        ('Conductividad (mS / cm)', 'TD'),
        ('Misc', 'EO/PO'),
        ('Misc', 'Destino Leche'),
        ('Flujos máximos (kg/min)', 'DI'),
        ('Flujos máximos (kg/min)', 'DD'),
        ('Flujos máximos (kg/min)', 'TI'),
        ('Flujos máximos (kg/min)', 'TD'),
        ('Producciones (kg)', 'DI'),
        ('Producciones (kg)', 'DD'),
        ('Producciones (kg)', 'TI'),
        ('Producciones (kg)', 'TD'),
        ('Misc', 'Razón de la desviación')
    ])

    df.columns = columns

    return df