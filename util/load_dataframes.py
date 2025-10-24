import pandas as pd

def load_dataframe_vacas(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, header=[0,1], skiprows=[1])

    # Definir nombres de columnas multiíndice
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
    # Eliminar columna innecesaria
    df = df.drop([('Misc','Razón de la desviación')], axis = 1)

    # Separar columnas de fecha y hora
    
    df[('Fecha y hora de inicio','fecha')] = df[('Main', 'Hora de inicio')].str.split(' ').str[0]
    df[('Fecha y hora de inicio','fecha')] = pd.to_datetime(df[('Fecha y hora de inicio','fecha')], format='%d/%m/%Y')
    df[('Fecha y hora de inicio','hora')] = (
        df[('Main', 'Hora de inicio')]
        .str.split(' ')
        .str[1:]
        .str.join(' ')
        .str.replace("a. m.", "AM")
        .str.replace("p. m.", "PM")
        
    )
    df[('Fecha y hora de inicio','hora')] = pd.to_datetime(df[('Fecha y hora de inicio','hora')], format="%I:%M %p").dt.time
    df = df.drop([('Main', 'Hora de inicio')], axis = 1)

    # Llenar valores nulos con 0
    df = df.fillna(0)
    
    # Crear columnas de totales para las columnas que involucran a cada cuarto.
    mediaFlujosColumns = df.xs('Media de los flujos (kg/min)', axis=1, level=0)
    df[('Media de los flujos (kg/min)', 'Total')] = mediaFlujosColumns.sum(axis=1)

    conductividadColumns = df.xs('Conductividad (mS / cm)', axis=1, level=0)
    df[('Conductividad (mS / cm)', 'Total')] = conductividadColumns.sum(axis=1)
        
    maxFlujosColumns = df.xs('Flujos máximos (kg/min)', axis=1, level=0)
    df[('Flujos máximos (kg/min)', 'Total')] = maxFlujosColumns.sum(axis=1)

    produccionesColumns = df.xs('Producciones (kg)', axis=1, level=0)
    df[('Producciones (kg)', 'Total')] = produccionesColumns.sum(axis=1)


    # Reordenar las columnas para mayor claridad

    columnOrder = [
        'ID',
        'Fecha y hora de inicio',
        'Main',
        'Estado',
        'Media de los flujos (kg/min)',
        'Sangre (ppm)',
        'Conductividad (mS / cm)',
        'Flujos máximos (kg/min)',
        'Producciones (kg)',
        'Misc'
    ]

    new_cols = []
    for top in columnOrder:
        new_cols.extend([c for c in df.columns if c[0] == top])

    # Añadir cualquier columna sobrante (que no esté en columnOrder) al final
    remaining = [c for c in df.columns if c not in new_cols]
    new_cols.extend(remaining)

    df = df.loc[:, new_cols]



    return df



def load_dataframe_patadas(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    return df

def load_dataframe_inventario(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    return df

def load_dataframe_reporte(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    return df