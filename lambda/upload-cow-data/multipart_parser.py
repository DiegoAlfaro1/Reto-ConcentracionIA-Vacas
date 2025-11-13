"""
Módulo para parsear datos multipart/form-data.
Contiene funciones auxiliares para procesar archivos enviados vía HTTP.
"""

import logging

logger = logging.getLogger()


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