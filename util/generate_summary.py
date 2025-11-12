import pandas as pd
import numpy as np

from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.signal import find_peaks

from scipy import stats


def generate_summary(dataFrame: pd.DataFrame) -> pd.DataFrame:
    columnasNumericas = dataFrame.select_dtypes(include=['float64', 'int64']).columns
    columnasNumericas = columnasNumericas.drop(('ID','ID Vaca'))

    promediosPorVaca = dataFrame.groupby(('ID', 'ID Vaca'))[columnasNumericas].mean()

    # Obtener fecha min y max por vaca
    datosPorVaca = dataFrame.groupby(('ID', 'ID Vaca'))[[('Fecha y hora de inicio', 'fecha')]].agg(['min', 'max']) #[('Fecha y hora de inicio', 'fecha')]
    datosPorVaca.columns = ['PrimeraFecha', 'UltimaFecha']

    # Obtener cantidad de días entre la primera y última fecha
    datosPorVaca['DiasTotales'] = (datosPorVaca['UltimaFecha'] - datosPorVaca['PrimeraFecha']).dt.days

    # Obtener total de ordeños
    datosPorVaca['OrdeñosTotales'] = dataFrame.groupby(('ID', 'ID Vaca'))[[('Main', 'Accion')]].count()

    # Obtener la duración promedio de ordeño
    datosPorVaca['DuracionPromedio'] = dataFrame.groupby(('ID', 'ID Vaca'))[[('Main', 'Duracion (mm:ss)')]].mean()

    # Obtener suma y promedio de índice de mastitis
    datosPorVaca['MastitisTotal'] = dataFrame.groupby(('ID', 'ID Vaca'))[[('Estado', 'Ubre')]].sum()
    datosPorVaca['MastitisPromedio'] = dataFrame.groupby(('ID', 'ID Vaca'))[[('Estado', 'Ubre')]].mean()


    datosPorVaca = datosPorVaca.reset_index()

    datosPorVaca.columns = ['ID', 'PrimeraFecha', 'UltimaFecha', 'DiasTotales', 'OrdeñosTotales','DuracionPromedio', 'MastitisTotal', 'MastitisPromedio']

    # Inicializar columna de tasa de decaimiento de producción
    datosPorVaca['TasaDecaemientoProduccion'] = 0

    datosPorVaca.sort_values(by='DiasTotales', ascending=False)





    if (('Producciones (kg)', 'Total')) in dataFrame.columns:
        prod_col = ('Producciones (kg)', 'Total')
    # Preparar dataframe con id_vaca, fecha y producción
    df_plot = dataFrame[[('ID','ID Vaca'), ('Fecha y hora de inicio','fecha'), prod_col]].copy()
    df_plot.columns = ['id_vaca', 'fecha', 'produccion']

    # Asegurar tipo datetime para fecha (si viene como datetime ya lo mantiene)
    df_plot['fecha'] = pd.to_datetime(df_plot['fecha']).dt.date

    # Agregar por día y vaca (suma de producciones en el mismo día)
    daily = df_plot.groupby(['id_vaca', 'fecha'], as_index=False)['produccion'].sum()


    # Graficar una figura por vaca
    for vaca in sorted(daily['id_vaca'].unique()):
        sub = daily[daily['id_vaca'] == vaca].sort_values('fecha')






    # Asegurar datetime real, no date
    df_plot['fecha'] = pd.to_datetime(df_plot['fecha'])

    # Agregar por día y vaca (suma)
    daily = df_plot.groupby(['id_vaca', 'fecha'], as_index=False)['produccion'].sum()



    for vaca in sorted(daily['id_vaca'].unique()):
        sub = daily[daily['id_vaca'] == vaca].sort_values('fecha').copy()

        # Aplicar LOWESS (ajusta frac según qué tan suave quieres la curva)
        smoothed = lowess(sub['produccion'], sub['fecha'], frac=0.20)
        sub['smooth'] = smoothed[:,1]


        # Convertimos fechas a valores numéricos para find_peaks
        x = np.arange(len(sub))               # índice numérico
        y = sub['smooth'].values              # serie suavizada

        # Detectar máximos locales (picos)
        peaks, _ = find_peaks(y, distance=10, prominence=0.5)

        if len(peaks) > 0:
            peak_index = peaks[0]   # Primer máximo local
        else:
            peak_index = y.argmax() # Si no encuentra picos, usar máximo global

        peak_date = sub.iloc[peak_index]['fecha']
        peak_value = sub.iloc[peak_index]['smooth']

        # Detectar mínimos después del pico (inicio decaimiento)
        y_post = y[peak_index:]
        min_candidates, _ = find_peaks(-y_post, distance=5, prominence=0.5)

        if len(min_candidates) > 0:
            min_index = peak_index + min_candidates[0]
        else:
            min_index = y.argmin()  # Se usa mínimo global si no encuentra local

        min_date = sub.iloc[min_index]['fecha']
        min_value = sub.iloc[min_index]['smooth']

        # Cálculo de pendiente de decaimiento (kg/día)
        days = (min_date - peak_date).days if (min_date - peak_date).days != 0 else 1
        slope = (min_value - peak_value) / days

        # Guardar pendiente en el DataFrame de resumen
        datosPorVaca.loc[datosPorVaca['ID'] == vaca, 'TasaDecaemientoProduccion'] = slope



    
    trends_and_stats = []

    for vaca in sorted(daily['id_vaca'].unique()):
        sub = daily[daily['id_vaca'] == vaca].sort_values('fecha').copy()
        
        # Convertir fecha a número de días desde el inicio
        sub['dias_transcurridos'] = (sub['fecha'] - sub['fecha'].min()).dt.days.astype(float)
        
        # Calcular estadísticas de producción
        prod_mean = sub['produccion'].mean()
        prod_std = sub['produccion'].std()
        prod_min = sub['produccion'].min()
        prod_max = sub['produccion'].max()
        prod_cv = prod_std / prod_mean if prod_mean != 0 else 0  # Coeficiente de variación
        n_ordeños = len(sub)
        
        # Regresión lineal: producción vs tiempo
        if len(sub) > 2:  # mínimo 2 puntos para regresión
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                sub['dias_transcurridos'], 
                sub['produccion']
            )
            trends_and_stats.append({
                'id_vaca': vaca,
                # Tasa de cambio
                'tasa_cambio_media_kg_dia': slope,
                'tasa_cambio_r2': r_value**2,
                'tasa_cambio_pvalor': p_value,
                # Desviación estándar y variabilidad
                'prod_mean': prod_mean,
                'prod_std': prod_std,
                'prod_min': prod_min,
                'prod_max': prod_max,
                'prod_cv': prod_cv,
                'n_ordeños': n_ordeños
            })
        else:
            trends_and_stats.append({
                'id_vaca': vaca,
                'tasa_cambio_media_kg_dia': 0,
                'tasa_cambio_r2': 0,
                'tasa_cambio_pvalor': 1,
                'prod_mean': prod_mean,
                'prod_std': prod_std,
                'prod_min': prod_min,
                'prod_max': prod_max,
                'prod_cv': prod_cv,
                'n_ordeños': n_ordeños
            })

    trends_df = pd.DataFrame(trends_and_stats)

    # Merge con datosPorVaca
    datosPorVaca = pd.merge(
        datosPorVaca, 
        trends_df, 
        left_on='ID', 
        right_on='id_vaca', 
        how='left'
    )

    # Eliminar columna duplicada
    datosPorVaca = datosPorVaca.drop('id_vaca', axis=1)



    cols = [('ID', 'ID Vaca'), ('Estado', 'Pezones no encontrados')]
    df_pezones = dataFrame[cols].copy()

    # Aplanar nombres de columnas
    df_pezones.columns = ['_'.join(col) if isinstance(col, tuple) else col for col in df_pezones.columns]

    # Ahora las columnas se llaman 'ID_ID Vaca' y 'Estado_Pezones no encontrados'

    # Separar y explotar
    df_expandido = (
        df_pezones
        .assign(temp=df_pezones['Estado_Pezones no encontrados'].str.split(','))
        .explode('temp')
    )

    # Limpiar espacios
    df_expandido['temp'] = df_expandido['temp'].str.strip()

    # Agrupar y contar
    conteos = (
        df_expandido
        .groupby('ID_ID Vaca')['temp']
        .value_counts()
        .unstack(fill_value=0)
    )

    conteos.reset_index(inplace=True)
    conteos = conteos.drop(['Todos'], axis=1)
    conteos['Total'] = conteos['DD']+ conteos['DI'] + conteos['TD'] + conteos['TI']
    conteos = conteos.add_prefix('NoEncontrados')


    datosPorVaca = pd.merge(
        datosPorVaca, 
        conteos, 
        left_on='ID', 
        right_on='NoEncontradosID_ID Vaca', 
        how='left'
    )








    df_kicks = dataFrame[[('ID', 'ID Vaca'), ('Estado', 'Patada')]].copy()
    df_kicks.columns = ['id_vaca', 'patada_cuarto']

    # Reemplazar NaN con 'Sin patada'
    df_kicks['patada_cuarto'] = df_kicks['patada_cuarto'].fillna('Sin patada')

    # Contar ocurrencias de cada cuarto por vaca
    kicks_counts = df_kicks.groupby('id_vaca')['patada_cuarto'].value_counts().unstack(fill_value=0)

    # Renombrar columnas para claridad
    kicks_counts.columns = [f'patadas_{col}' for col in kicks_counts.columns]

    # Calcular total de patadas (excluyendo 'Sin patada')


    kicks_counts['patadas_total'] = kicks_counts['patadas_TI']+ kicks_counts['patadas_TD']+kicks_counts['patadas_DI']+kicks_counts['patadas_DD']
    # Resetear índice
    kicks_counts = kicks_counts.reset_index()
    kicks_counts.columns = ['id_vaca'] + list(kicks_counts.columns[1:])

    # Asegurar que existan todas las columnas de cuartos (TI, TD, DI, DD)
    cuartos_esperados = ['TI', 'TD', 'DI', 'DD']
    for cuarto in cuartos_esperados:
        if f'patadas_{cuarto}' not in kicks_counts.columns:
            kicks_counts[f'patadas_{cuarto}'] = 0

    # Reordenar columnas
    kicks_counts = kicks_counts[['id_vaca', 'patadas_TI', 'patadas_TD', 'patadas_DI', 'patadas_DD', 'patadas_total']]

    kicks_counts.head(10)

    # Merge con datosPorVaca
    datosPorVaca = pd.merge(
        datosPorVaca, 
        kicks_counts, 
        left_on='ID', 
        right_on='id_vaca', 
        how='left'
    )







    datosPorVaca['PatadasPorDia'] = datosPorVaca['patadas_total'] / datosPorVaca['DiasTotales']

    datosPorVaca['PatadasPorOrdeño'] = datosPorVaca['patadas_total'] / datosPorVaca['OrdeñosTotales']

    datosPorVaca['PatadasPorHora'] = datosPorVaca['patadas_total'] / (datosPorVaca['DuracionPromedio'].dt.total_seconds()* datosPorVaca['OrdeñosTotales'] / 3600)



    df1 = promediosPorVaca.reset_index().copy()
    df2 = datosPorVaca.copy()


    cols = [str(c) for c in datosPorVaca.columns]

    # Crear MultiIndex con un único nivel superior 'Datos Generados'
    # df2.columns = pd.MultiIndex.from_tuples([('Datos Generados', c) for c in cols])
    df2.columns = pd.MultiIndex.from_product([['Datos Generados'], df2.columns])

    df1 = df1.set_index(('ID', 'ID Vaca')) 
    df2 = df2.set_index(('Datos Generados', 'ID')) 

    df_final = df1.join(df2, how='left')

    df_final = df_final.reset_index()

    return df_final




def save_dataframe(df: pd.DataFrame, file_path: str):
    df_final = df.copy()
    df_final.columns = ['_'.join(col).strip() for col in df_final.columns.values]

    df_final.to_csv(file_path, encoding='utf-8')

    # Verificar que se guardó
    print("Archivo guardado exitosamente")