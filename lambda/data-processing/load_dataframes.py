import pandas as pd

from statsmodels.nonparametric.smoothers_lowess import lowess

def load_dataframe_vacas(df: pd.DataFrame) -> pd.DataFrame:
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

    df[('ID', 'ID Vaca')] = df[('ID', 'ID Vaca')].astype(int)
    df = df.drop([('Misc','Razon de la desviacion')], axis=1)

    # Procesar fecha y hora
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

    df = df.drop([('Main', 'Hora de inicio')], axis=1)

    # Convertir duración
    df[('Main', 'Duracion (mm:ss)')] = pd.to_timedelta('00:' + df[('Main', 'Duracion (mm:ss)')])

    df = df.fillna(0)

    # Totales
    df[('Media de los flujos (kg/min)', 'Total')] = df.xs('Media de los flujos (kg/min)', axis=1, level=0).sum(axis=1)
    df[('Conductividad (mS / cm)', 'Total')] = df.xs('Conductividad (mS / cm)', axis=1, level=0).sum(axis=1)
    df[('Flujos maximos (kg/min)', 'Total')] = df.xs('Flujos maximos (kg/min)', axis=1, level=0).sum(axis=1)
    df[('Producciones (kg)', 'Total')] = df.xs('Producciones (kg)', axis=1, level=0).sum(axis=1)

    df[('Producciones (kg)', 'Suavizado')] = lowess(
        df['Producciones (kg)', 'Total'],
        df[('Fecha y hora de inicio','fecha')],
        frac=0.1
    )[:,1]

    df[('Sangre (ppm)', 'Total')] = df.xs('Sangre (ppm)', axis=1, level=0).sum(axis=1)

    # Ordenar columnas
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

    remaining = [c for c in df.columns if c not in new_cols]
    new_cols.extend(remaining)

    return df.loc[:, new_cols]
