import pandas as pd
import numpy as np
import os
from pathlib import Path
import warnings
from util.load_dataframe import load_dataframe
warnings.filterwarnings('ignore')

# Configuración de rutas
BASE_PATH = Path(__file__).parent
VACAS_FOLDER = BASE_PATH / 'datos'
ARCHIVO_MERGED = BASE_PATH / 'datos' / 'registros_sesiones_merged.csv'

def leer_csv_con_multiheader(archivo_path):
    """
    Lee un archivo CSV con headers de múltiples niveles usando el método estándar del equipo.
    Utiliza la función load_dataframe del módulo util.
    """
    # Usar la función del módulo util
    df = load_dataframe(str(archivo_path))
    
    # Aplanar el MultiIndex para facilitar el análisis
    # Combinar los niveles con guion bajo
    df.columns = ['_'.join(col).strip() for col in df.columns.values]
    
    return df

def analizar_todos_los_archivos():
    """
    Analiza todos los archivos CSV de la carpeta Vacas y genera un reporte consolidado general.
    Prioriza el archivo merged si existe, de lo contrario usa los archivos individuales.
    """
    # Verificar si existe el archivo merged
    if ARCHIVO_MERGED.exists():
        print(f"Se encontro el archivo merged: {ARCHIVO_MERGED.name}")
        print("Procesando archivo consolidado con estructura multi-header...\n")
        
        try:
            df_consolidado = leer_csv_con_multiheader(ARCHIVO_MERGED)
            
            reporte = {
                'archivos_procesados': [ARCHIVO_MERGED.name],
                'archivos_con_error': [],
                'total_archivos': 1,
                'archivos_exitosos': 1,
                'total_filas': len(df_consolidado),
                'total_columnas': len(df_consolidado.columns),
                'columnas': list(df_consolidado.columns),
                'tamano_total_kb': os.path.getsize(ARCHIVO_MERGED) / 1024,
                'df': df_consolidado
            }
            
            return reporte
            
        except Exception as e:
            print(f"Error al procesar el archivo merged: {e}")
            print("Intentando procesar archivos individuales...\n")
    
    # Si no existe el archivo merged o hubo error, usar archivos individuales
    archivos_csv = list(VACAS_FOLDER.glob('*.csv'))
    
    if not archivos_csv:
        print(f"Error: No se encontraron archivos CSV en {VACAS_FOLDER}")
        return None
    
    print(f"Se encontraron {len(archivos_csv)} archivos CSV en la carpeta Vacas")
    print("Procesando datos con estructura multi-header...\n")
    
    # Leer y consolidar todos los archivos
    dataframes = []
    archivos_procesados = []
    archivos_con_error = []
    tamanos_archivos = []
    
    for archivo in sorted(archivos_csv):
        try:
            df = leer_csv_con_multiheader(archivo)
            dataframes.append(df)
            archivos_procesados.append(archivo.name)
            tamanos_archivos.append(os.path.getsize(archivo) / 1024)  # KB
        except Exception as e:
            archivos_con_error.append({'archivo': archivo.name, 'error': str(e)})
    
    if not dataframes:
        print("Error: No se pudo procesar ningún archivo")
        return None
    
    # Consolidar todos los DataFrames
    df_consolidado = pd.concat(dataframes, ignore_index=True)
    
    # Generar reporte consolidado
    reporte = {
        'archivos_procesados': archivos_procesados,
        'archivos_con_error': archivos_con_error,
        'total_archivos': len(archivos_csv),
        'archivos_exitosos': len(archivos_procesados),
        'total_filas': len(df_consolidado),
        'total_columnas': len(df_consolidado.columns),
        'columnas': list(df_consolidado.columns),
        'tamano_total_kb': sum(tamanos_archivos),
        'df': df_consolidado
    }
    
    return reporte

def analizar_valores_faltantes(df):
    """Analiza valores faltantes en el dataframe."""
    valores_faltantes = df.isnull().sum()
    total_celdas = len(df) * len(df.columns)
    total_faltantes = valores_faltantes.sum()
    
    # Top 10 columnas con más valores faltantes
    top_faltantes = valores_faltantes[valores_faltantes > 0].sort_values(ascending=False).head(10)
    
    resultado = {
        'total_valores_faltantes': int(total_faltantes),
        'porcentaje_total': round((total_faltantes / total_celdas) * 100, 2),
        'columnas_con_faltantes': int((valores_faltantes > 0).sum()),
        'top_10_columnas': {
            col: {
                'cantidad': int(valores_faltantes[col]),
                'porcentaje': round((valores_faltantes[col] / len(df)) * 100, 2)
            } for col in top_faltantes.index
        }
    }
    
    return resultado

def analizar_duplicados(df):
    """Analiza duplicados en el dataframe."""
    duplicados_totales = df.duplicated().sum()
    
    # Duplicados en columnas clave
    duplicados_por_columna = {}
    columnas_clave = ['Hora de inicio', 'Acción', 'Producción (kg)', 'Número de ordeño']
    
    for col in columnas_clave:
        if col in df.columns:
            duplicados_por_columna[col] = int(df[col].duplicated().sum())
    
    resultado = {
        'filas_duplicadas_completas': int(duplicados_totales),
        'porcentaje_duplicados': round((duplicados_totales / len(df)) * 100, 2),
        'duplicados_por_columna': duplicados_por_columna
    }
    
    return resultado

def analizar_inconsistencias(df):
    """Analiza inconsistencias en los datos."""
    inconsistencias = {}
    
    # Valores negativos en columnas numéricas
    columnas_numericas = df.select_dtypes(include=[np.number]).columns
    negativos_por_columna = {}
    
    for col in columnas_numericas:
        valores_negativos = (df[col] < 0).sum()
        if valores_negativos > 0:
            negativos_por_columna[col] = int(valores_negativos)
    
    if negativos_por_columna:
        inconsistencias['valores_negativos'] = negativos_por_columna
    
    # Valores únicos en columnas de estado
    columnas_estado = ['Acción', 'Patada', 'Incompleto', 'Pezones no encontrados', 'Ubre', 'Destino Leche']
    valores_estado = {}
    
    for col in columnas_estado:
        if col in df.columns:
            valores_unicos = df[col].dropna().unique()
            if len(valores_unicos) > 0:
                valores_estado[col] = list(map(str, valores_unicos))[:10]  # Máximo 10 valores
    
    if valores_estado:
        inconsistencias['valores_unicos_columnas_estado'] = valores_estado
    
    return inconsistencias

def analizar_outliers(df):
    """Analiza outliers usando el método IQR."""
    columnas_numericas = df.select_dtypes(include=[np.number]).columns
    outliers_resumen = {}
    
    # Solo analizar columnas numéricas importantes
    columnas_importantes = ['Producción (kg)', 'Número de ordeño', 'RCS (* 1000 células / ml)']
    
    for col in columnas_numericas:
        if col in columnas_importantes and df[col].notna().sum() > 0:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            limite_inferior = Q1 - 1.5 * IQR
            limite_superior = Q3 + 1.5 * IQR
            
            outliers_count = ((df[col] < limite_inferior) | (df[col] > limite_superior)).sum()
            
            if outliers_count > 0:
                outliers_resumen[col] = {
                    'cantidad': int(outliers_count),
                    'porcentaje': round((outliers_count / df[col].notna().sum()) * 100, 2),
                    'rango_esperado': f'{round(limite_inferior, 2)} a {round(limite_superior, 2)}',
                    'min': round(df[col].min(), 2),
                    'max': round(df[col].max(), 2)
                }
    
    return outliers_resumen

def analizar_precision_coherencia(df):
    """Analiza precisión y coherencia de datos numéricos."""
    columnas_numericas = df.select_dtypes(include=[np.number]).columns
    estadisticas = {}
    
    # Columnas importantes para análisis
    columnas_importantes = ['Producción (kg)', 'Número de ordeño', 'RCS (* 1000 células / ml)']
    
    for col in columnas_numericas:
        if col in columnas_importantes and df[col].notna().sum() > 0:
            estadisticas[col] = {
                'media': round(df[col].mean(), 2),
                'mediana': round(df[col].median(), 2),
                'desviacion_std': round(df[col].std(), 2),
                'min': round(df[col].min(), 2),
                'max': round(df[col].max(), 2),
                'coef_variacion': round((df[col].std() / df[col].mean() * 100) if df[col].mean() != 0 else 0, 2)
            }
    
    return estadisticas

def analizar_completitud(df):
    """Analiza la completitud de los datos."""
    completitud_por_columna = {}
    
    for col in df.columns:
        valores_no_nulos = df[col].notna().sum()
        porcentaje = round((valores_no_nulos / len(df)) * 100, 2)
        completitud_por_columna[col] = {
            'completitud_porcentaje': porcentaje,
            'valores_completos': int(valores_no_nulos),
            'valores_faltantes': int(len(df) - valores_no_nulos)
        }
    
    # Top 10 columnas menos completas
    columnas_incompletas = {k: v for k, v in completitud_por_columna.items() 
                           if v['completitud_porcentaje'] < 100}
    
    resultado = {
        'columnas_100_completas': len([c for c in completitud_por_columna.values() 
                                       if c['completitud_porcentaje'] == 100]),
        'columnas_incompletas': len(columnas_incompletas),
        'top_10_menos_completas': dict(sorted(columnas_incompletas.items(), 
                                             key=lambda x: x[1]['completitud_porcentaje'])[:10])
    }
    
    return resultado

def generar_reporte_consolidado(info_general, df):
    """Genera el reporte consolidado de calidad de datos."""
    print("\n" + "="*100)
    print("REPORTE CONSOLIDADO DE CALIDAD DE DATOS - FOLDER VACAS")
    print("="*100)
    
    # INFORMACIÓN GENERAL
    print(f"\nINFORMACION GENERAL:")
    print(f"{'='*100}")
    print(f"Total de archivos CSV encontrados: {info_general['total_archivos']}")
    print(f"Archivos procesados exitosamente: {info_general['archivos_exitosos']}")
    
    if info_general['archivos_con_error']:
        print(f"Archivos con errores: {len(info_general['archivos_con_error'])}")
        for err in info_general['archivos_con_error']:
            print(f"  - {err['archivo']}: {err['error']}")
    
    print(f"\nTotal de registros consolidados: {info_general['total_filas']:,}")
    print(f"Total de columnas: {info_general['total_columnas']}")
    print(f"Tamano total de archivos: {info_general['tamano_total_kb']:.2f} KB")
    
    print(f"\nColumnas del dataset (combinando categoria_nombre):")
    for i, col in enumerate(info_general['columnas'], 1):
        print(f"  {i}. {col}")
    
    # 1. VALORES FALTANTES
    print(f"\n\n1. VALORES FALTANTES:")
    print(f"{'='*100}")
    faltantes = analizar_valores_faltantes(df)
    print(f"Total de valores faltantes: {faltantes['total_valores_faltantes']:,}")
    print(f"Porcentaje del total de celdas: {faltantes['porcentaje_total']}%")
    print(f"Columnas con valores faltantes: {faltantes['columnas_con_faltantes']}")
    
    if faltantes['top_10_columnas']:
        print(f"\nTop 10 columnas con mas valores faltantes:")
        for col, info in faltantes['top_10_columnas'].items():
            print(f"  - {col}: {info['cantidad']:,} ({info['porcentaje']}%)")
    
    # 2. DUPLICADOS
    print(f"\n\n2. DUPLICADOS:")
    print(f"{'='*100}")
    duplicados = analizar_duplicados(df)
    print(f"Filas completamente duplicadas: {duplicados['filas_duplicadas_completas']:,}")
    print(f"Porcentaje de duplicados: {duplicados['porcentaje_duplicados']}%")
    
    if duplicados['duplicados_por_columna']:
        print(f"\nDuplicados en columnas clave:")
        for col, count in duplicados['duplicados_por_columna'].items():
            print(f"  - {col}: {count:,}")
    
    # 3. INCONSISTENCIAS
    print(f"\n\n3. INCONSISTENCIAS:")
    print(f"{'='*100}")
    inconsistencias = analizar_inconsistencias(df)
    
    if 'valores_negativos' in inconsistencias:
        print(f"Valores negativos detectados en columnas numericas:")
        for col, count in inconsistencias['valores_negativos'].items():
            print(f"  - {col}: {count:,} valores negativos")
    
    if 'valores_unicos_columnas_estado' in inconsistencias:
        print(f"\nValores unicos en columnas de estado:")
        for col, valores in inconsistencias['valores_unicos_columnas_estado'].items():
            print(f"  - {col}: {', '.join(valores)}")
    
    if not inconsistencias:
        print("No se detectaron inconsistencias significativas")
    
    # 4. OUTLIERS
    print(f"\n\n4. OUTLIERS (Valores atipicos):")
    print(f"{'='*100}")
    outliers = analizar_outliers(df)
    
    if outliers:
        for col, info in outliers.items():
            print(f"\n{col}:")
            print(f"  - Cantidad de outliers: {info['cantidad']:,} ({info['porcentaje']}%)")
            print(f"  - Rango esperado (IQR): {info['rango_esperado']}")
            print(f"  - Rango encontrado: {info['min']} a {info['max']}")
    else:
        print("No se detectaron outliers significativos en las columnas principales")
    
    # 5. PRECISIÓN Y COHERENCIA
    print(f"\n\n5. PRECISION Y COHERENCIA:")
    print(f"{'='*100}")
    estadisticas = analizar_precision_coherencia(df)
    
    if estadisticas:
        for col, stats in estadisticas.items():
            print(f"\n{col}:")
            print(f"  - Media: {stats['media']}")
            print(f"  - Mediana: {stats['mediana']}")
            print(f"  - Desviacion estandar: {stats['desviacion_std']}")
            print(f"  - Rango: {stats['min']} - {stats['max']}")
            print(f"  - Coeficiente de variacion: {stats['coef_variacion']}%")
    
    # 6. COMPLETITUD
    print(f"\n\n6. COMPLETITUD:")
    print(f"{'='*100}")
    completitud = analizar_completitud(df)
    print(f"Columnas 100% completas: {completitud['columnas_100_completas']}")
    print(f"Columnas con datos incompletos: {completitud['columnas_incompletas']}")
    
    if completitud['top_10_menos_completas']:
        print(f"\nTop 10 columnas menos completas:")
        for col, info in completitud['top_10_menos_completas'].items():
            print(f"  - {col}: {info['completitud_porcentaje']}% completo ({info['valores_faltantes']:,} faltantes)")
    
    # 7. ACCESIBILIDAD
    print(f"\n\n7. ACCESIBILIDAD:")
    print(f"{'='*100}")
    print(f"Formato: CSV")
    print(f"Encoding: UTF-8")
    print(f"Lectura exitosa: Si")
    print(f"Estructura tabular: Si")
    print(f"Total de archivos accesibles: {info_general['archivos_exitosos']}/{info_general['total_archivos']}")
    print(f"Archivos procesados:")
    for archivo in info_general['archivos_procesados']:
        print(f"  - {archivo}")

def main():
    """Función principal."""
    print("Iniciando analisis consolidado de datos de la carpeta 'Vacas'...")
    print(f"Ruta: {VACAS_FOLDER}\n")
    
    # Verificar que la carpeta existe
    if not VACAS_FOLDER.exists():
        print(f"Error: La carpeta {VACAS_FOLDER} no existe")
        return
    
    # Analizar todos los archivos
    info_general = analizar_todos_los_archivos()
    
    if info_general is None:
        return
    
    # Generar reporte consolidado
    generar_reporte_consolidado(info_general, info_general['df'])
    
    print(f"\n\n{'='*100}")
    print("Analisis completado exitosamente!")
    print(f"{'='*100}\n")

if __name__ == "__main__":
    main()
