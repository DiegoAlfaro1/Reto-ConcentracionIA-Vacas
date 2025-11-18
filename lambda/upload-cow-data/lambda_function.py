"""
Lambda function para procesar y subir archivos CSV a S3.
Recibe archivos vía multipart/form-data desde API Gateway.
"""

import json
import boto3
import base64
import os
import logging

# Importar función de parsing desde el módulo separado
from multipart_parser import parse_multipart_form
# Configuración del logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Cliente de AWS S3
s3 = boto3.client('s3')
BUCKET_NAME = os.environ.get('BUCKET_NAME')
FOLDER = 'raw/'

# Cliente lambda
lambda_client = boto3.client('lambda')
MERGE_LAMBDA_NAME = os.environ.get('MERGE_LAMBDA_NAME', 'dataProcessing')

def lambda_handler(event, context):
    """
    Función principal de Lambda que procesa archivos CSV enviados vía multipart/form-data.
    
    Flujo de ejecución:
    1. Decodifica el cuerpo de la petición (base64 si es necesario)
    2. Extrae headers y valida Content-Type
    3. Parsea los archivos del multipart/form-data
    4. Sube cada archivo a S3 con su nombre original
    5. Retorna respuesta con información de archivos subidos
    
    Nota: Si se sube un archivo con el mismo nombre, se sobrescribirá el archivo existente en S3.
    
    Args:
        event (dict): Evento de API Gateway con información de la petición HTTP
        context (object): Contexto de ejecución de Lambda
    
    Returns:
        dict: Respuesta HTTP con statusCode, headers y body
              - 200: Archivos subidos exitosamente
              - 500: Error en el procesamiento
    
    Variables de entorno requeridas:
        BUCKET_NAME (str): Nombre del bucket de S3 donde se subirán los archivos
    """
    logger.info(f"Lambda invoked | RequestId: {context.aws_request_id}")

    try:
        # Validar que el bucket name esté configurado
        if not BUCKET_NAME:
            raise ValueError("BUCKET_NAME not set")

        # Decodificar el cuerpo de la petición
        # API Gateway envía el body en base64 cuando isBase64Encoded es true
        body = base64.b64decode(event["body"]) if event.get("isBase64Encoded") else event["body"].encode()
        
        logger.info(f"Body size: {len(body)} bytes")
        
        # Normalizar headers a minúsculas para facilitar acceso
        headers = {k.lower(): v for k, v in (event.get("headers") or {}).items()}
        content_type = headers.get("content-type")
        
        if not content_type:
            raise ValueError("Missing Content-Type header")

        logger.info(f"Content-Type: {content_type}")

        # Parsear multipart y obtener todos los archivos
        files = parse_multipart_form(body, content_type)
        
        if not files:
            raise ValueError("No files found in form data")
        
        logger.info(f"Found {len(files)} file(s) to upload")

        # Subir todos los archivos a S3
        uploaded_files = []
        
        for file_data in files:
            filename = file_data['filename']
            file_content = file_data['content']
            
            # Sanitizar el nombre del archivo (eliminar rutas potencialmente peligrosas)
            safe_name = os.path.basename(filename)
            
            # Nombre final en S3: folder/nombre_original.csv
            # Si existe, se sobrescribirá automáticamente
            s3_key = f"{FOLDER}{safe_name}"

            # Subir archivo a S3
            s3.put_object(
                Bucket=BUCKET_NAME,
                Key=s3_key,
                Body=file_content,
                ContentType="text/csv",
                Metadata={
                    'original-filename': safe_name
                }
            )

            logger.info(f"Uploaded file: {s3_key}")
            
            # Guardar información del archivo subido para la respuesta
            uploaded_files.append({
                'original_name': safe_name,
                's3_key': s3_key,
                'size_bytes': len(file_content)
            })
        
        logger.info("All files uploaded. invoking data processing function")

        process_payload = {
            'bucket': BUCKET_NAME,
            'trigger': 'manual',
            'files_uploaded': len(uploaded_files),
            'upload_request_id': context.aws_request_id
        }

        response = lambda_client.invoke(
            FunctionName=MERGE_LAMBDA_NAME,
            InvocationType='Event',  # Asíncrono - no bloquea
            Payload=json.dumps(process_payload)
        )
        logger.info(f"Merge Lambda invoked with status: {response['StatusCode']}")

        # Retornar respuesta exitosa con información de todos los archivos
        return {
            'statusCode': 200,
            'body': json.dumps({
                "message": f"Successfully uploaded {len(uploaded_files)} file(s)",
                "files": uploaded_files,
                "total_files": len(uploaded_files)
            })
        }

    except Exception as e:
        # Capturar cualquier error y retornar respuesta de error
        logger.error(f"Error: {str(e)}", exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }