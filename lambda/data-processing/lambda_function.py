import json
import logging
import boto3

# Configuración del logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Configuración de cliente
s3 = boto3.client('s3')

def lambda_handler(event, context):
    """
    Handler principal de Lambda.
    Puede ser activado de dos formas:
    1. S3 Event (cuando se sube un archivo a S3) - event['Records'] existe
    2. Direct Invocation (desde otro Lambda) - event['bucket'] existe
    
    Retorna el DataFrame merged para ser usado por otra función.
    """
    logger.info("Data Processing Lambda invoked")
    logger.info(f"Event: {json.dumps(event)}")
    
    try:
        # Determinar el bucket según el tipo de invocación
        if 'Records' in event and event['Records']:
            # Invocado por S3 Event
            bucket = event['Records'][0]['s3']['bucket']['name']
            logger.info(f"Triggered by S3 event for bucket: {bucket}")
        elif 'bucket' in event:
            # Invocado directamente desde otro Lambda
            bucket = event['bucket']
            logger.info(f"Triggered by direct invocation for bucket: {bucket}")
        else:
            raise ValueError("No valid trigger source found. Expected 'Records' (S3) or 'bucket' (direct invocation)")
        
        # Contar objetos en raw/
        response = s3.list_objects_v2(Bucket=bucket, Prefix="raw/")
        objects_len = response.get("KeyCount", 0) - 1
        
        logger.info(f"{objects_len} Objects in raw data folder")
        
        # Ejecutar merge
        merged_df, metadata = merge_csv_files_from_s3(
            bucket=bucket,
            input_prefix="raw/"
        )
        
        # Log primeras filas del DataFrame (solo para debug)
        logger.info(f"DataFrame head:\n{merged_df.head().to_string()}")
        
        # Aquí puedes llamar a otra función o procesar el DataFrame
        # Ejemplo: otra_funcion(merged_df, bucket)
        
        logger.info(f"🚀 DataFrame merged disponible: {metadata['total_rows']} filas, {metadata['total_columns']} columnas")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Merge completed successfully',
                'metadata': metadata
            }),
            'dataframe': merged_df  # Disponible para siguiente función en el pipeline
        }
    
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
