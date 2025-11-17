import pandas as pd

from statsmodels.nonparametric.smoothers_lowess import lowess


def load_dataframe_vacas(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, header=[0,1], skiprows=[1])

    # Definir nombres de columnas multiíndice
    columns = pd.MultiIndex.from_tuples([
        ('ID', 'ID Vaca'),
        ('Main', 'Hora de inicio'),
        ('Main', 'Accion'),
        ('Main', 'Duracion (mm:ss)'),
        ('Main', 'Produccion (kg)'),
        ('Estado', 'Numero de ordeño'),
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
        ('Flujos maximos (kg/min)', 'DI'),
        ('Flujos maximos (kg/min)', 'DD'),
        ('Flujos maximos (kg/min)', 'TI'),
        ('Flujos maximos (kg/min)', 'TD'),
        ('Producciones (kg)', 'DI'),
        ('Producciones (kg)', 'DD'),
        ('Producciones (kg)', 'TI'),
        ('Producciones (kg)', 'TD'),
        ('Misc', 'Razon de la desviacion')
    ])

    df.columns = columns
    # Eliminar columna innecesaria

    df[('ID', 'ID Vaca')] = df[('ID', 'ID Vaca')].astype(int)

    df = df.drop([('Misc','Razon de la desviacion')], axis = 1)

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


    # Convertir columna de duración a time de pandas

    df[('Main', 'Duracion (mm:ss)')] = pd.to_timedelta('00:' + df[('Main', 'Duracion (mm:ss)')])

    # Llenar valores nulos con 0
    df = df.fillna(0)
    
    # Crear columnas de totales para las columnas que involucran a cada cuarto.
    mediaFlujosColumns = df.xs('Media de los flujos (kg/min)', axis=1, level=0)
    df[('Media de los flujos (kg/min)', 'Total')] = mediaFlujosColumns.sum(axis=1)

    conductividadColumns = df.xs('Conductividad (mS / cm)', axis=1, level=0)
    df[('Conductividad (mS / cm)', 'Total')] = conductividadColumns.sum(axis=1)
        
    maxFlujosColumns = df.xs('Flujos maximos (kg/min)', axis=1, level=0)
    df[('Flujos maximos (kg/min)', 'Total')] = maxFlujosColumns.sum(axis=1)

    produccionesColumns = df.xs('Producciones (kg)', axis=1, level=0)
    df[('Producciones (kg)', 'Total')] = produccionesColumns.sum(axis=1)



    # df[('Producciones (kg)', 'Suavizado')] = df[('Producciones (kg)', 'Total')].rolling(window=7, center=True).mean()

    df[('Producciones (kg)', 'Suavizado')] = lowess(df['Producciones (kg)', 'Total'], df[('Fecha y hora de inicio','fecha')], frac=0.1)[:,1]

    sangreColumns = df.xs('Sangre (ppm)', axis=1, level=0)
    df[('Sangre (ppm)', 'Total')] = sangreColumns.sum(axis=1)

    # Reordenar las columnas para mayor claridad

    columnOrder = [
        'ID',
        'Fecha y hora de inicio',
        'Main',
        'Estado',
        'Media de los flujos (kg/min)',
        'Sangre (ppm)',
        'Conductividad (mS / cm)',
        'Flujos maximos (kg/min)',
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


def load_dataframe_ranking(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    return df