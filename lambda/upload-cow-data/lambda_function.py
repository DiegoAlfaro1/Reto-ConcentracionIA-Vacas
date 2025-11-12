import json
import boto3
import base64
import time
import os
import logging
from io import BytesIO

# Configuración del logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Cliente de AWS S3
s3 = boto3.client('s3')
BUCKET_NAME = os.environ.get('BUCKET_NAME')
FOLDER = 'raw/'


def parse_multipart_form(body, content_type):
    """
    Analiza datos multipart/form-data usando la librería python-multipart.
    
    Args:
        body (bytes): Cuerpo de la petición HTTP decodificado
        content_type (str): Encabezado Content-Type de la petición
    
    Returns:
        list: Lista de diccionarios con información de archivos
              [{'filename': 'nombre.csv', 'content': b'...'}, ...]
    
    Raises:
        ValueError: Si no se encuentra el boundary en Content-Type
    """
    from multipart.multipart import parse_options_header, MultipartParser
    
    # Extraer el boundary del content-type
    content_type_header, options = parse_options_header(content_type)
    boundary = options.get(b'boundary')
    
    if not boundary:
        raise ValueError("No boundary found in Content-Type")
    
    logger.info(f"Parsing with boundary: {boundary.decode()}")
    
    # Almacenar todos los archivos encontrados
    files = []
    current_part = {}
    
    # Callbacks para el parser de multipart
    
    def on_part_begin():
        """Se ejecuta al inicio de cada parte del multipart"""
        nonlocal current_part
        current_part = {}
    
    def on_part_data(data, start, end):
        """
        Se ejecuta cuando se reciben datos del contenido del archivo.
        Acumula el contenido del archivo en chunks.
        """
        nonlocal current_part
        if 'content' not in current_part:
            current_part['content'] = b''
        current_part['content'] += data[start:end]
    
    def on_part_end():
        """
        Se ejecuta al finalizar cada parte del multipart.
        Guarda el archivo si tiene nombre y contenido.
        """
        nonlocal current_part
        # Solo guardar partes que tengan nombre de archivo y contenido
        if 'filename' in current_part and 'content' in current_part:
            files.append({
                'filename': current_part['filename'],
                'content': current_part['content']
            })
            logger.info(f"Part completed: {current_part['filename']} ({len(current_part['content'])} bytes)")
    
    def on_header_field(data, start, end):
        """Se ejecuta cuando se encuentra un nombre de header"""
        nonlocal current_part
        field = data[start:end]
        current_part['last_header_field'] = field
    
    def on_header_value(data, start, end):
        """
        Se ejecuta cuando se encuentra el valor de un header.
        Extrae el nombre del archivo del header Content-Disposition.
        """
        nonlocal current_part
        value = data[start:end]
        field = current_part.get('last_header_field', b'')
        
        if field.lower() == b'content-disposition':
            # Extraer el nombre del archivo del header Content-Disposition
            value_str = value.decode('utf-8', errors='ignore')
            if 'filename=' in value_str:
                if 'filename="' in value_str:
                    filename = value_str.split('filename="')[1].split('"')[0]
                else:
                    filename = value_str.split('filename=')[1].split()[0].strip('"')
                current_part['filename'] = filename
    
    def on_header_end():
        """Se ejecuta al finalizar cada header"""
        pass
    
    def on_headers_finished():
        """Se ejecuta cuando todos los headers han sido procesados"""
        pass
    
    def on_end():
        """Se ejecuta al finalizar todo el parsing"""
        pass
    
    # Crear el parser con los callbacks configurados
    parser = MultipartParser(boundary, {
        'on_part_begin': on_part_begin,
        'on_part_data': on_part_data,
        'on_part_end': on_part_end,
        'on_header_field': on_header_field,
        'on_header_value': on_header_value,
        'on_header_end': on_header_end,
        'on_headers_finished': on_headers_finished,
        'on_end': on_end
    })
    
    # Parsear el cuerpo de la petición
    parser.write(body)
    parser.finalize()
    
    return files


def lambda_handler(event, context):
    """
    Función principal de Lambda que procesa archivos CSV enviados vía multipart/form-data.
    
    Flujo de ejecución:
    1. Decodifica el cuerpo de la petición (base64 si es necesario)
    2. Extrae headers y valida Content-Type
    3. Parsea los archivos del multipart/form-data
    4. Sube cada archivo a S3 con nombres únicos
    5. Retorna respuesta con información de archivos subidos
    
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
        timestamp = int(time.time())  # Timestamp único para todos los archivos de esta petición
        
        for idx, file_data in enumerate(files):
            filename = file_data['filename']
            file_content = file_data['content']
            
            # Sanitizar el nombre del archivo (eliminar rutas potencialmente peligrosas)
            safe_name = os.path.basename(filename)
            
            # Crear nombre único: folder/timestamp_index_nombre.csv
            # El índice permite diferenciar archivos subidos en la misma petición
            final_name = f"{FOLDER}{timestamp}_{idx}_{safe_name}"

            # Subir archivo a S3
            s3.put_object(
                Bucket=BUCKET_NAME,
                Key=final_name,
                Body=file_content,
                ContentType="text/csv",
                Metadata={
                    'original-filename': safe_name,
                    'upload-timestamp': str(timestamp),
                    'file-index': str(idx)
                }
            )

            logger.info(f"Uploaded file {idx + 1}/{len(files)}: {final_name}")
            
            # Guardar información del archivo subido para la respuesta
            uploaded_files.append({
                'original_name': safe_name,
                's3_key': final_name,
                'size_bytes': len(file_content)
            })

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