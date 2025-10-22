# Reto-ConcentracionIA-Vacas

## Equipo

Daniel Queijeiro Albo - A01710441

Diego Alfaro Pinto - A01709971

Diego Isaac Fuentes Juvera - A01705506

Jesus Ramirez Delgado - A01274723

Mauricio Anguiano Juarez - A01703337

Luis Adrián Uribe Cruz - A01783129


## Links importantes

Indice de documentos del proyecto
https://docs.google.com/document/d/1T3OYqLs6bp0mpNbWSwZWNzy4HwuqY9HIyPhlONydFH8/edit?usp=drive_link

## Configuración del entorno virtual (venv)

Este proyecto utiliza un entorno virtual de Python (`venv`) para mantener aisladas las dependencias.  
Sigue los pasos a continuación para crear y activar el entorno antes de ejecutar el código.

---

### Crear el entorno virtual

En la raíz del proyecto, ejecuta el siguiente comando:

- Windows:
<code>
python -m venv venv
</code>

- MacOS:
<code>
python3 -m venv venv
</code>.

### Activar el entorno virtual

- Windows:
<code>
venv\Scripts\activate
</code>

- MacOS:
<code>
source venv\Scripts\activate
</code>

### Instalar dependencias

Las dependencias necesarias se obtienen del archivo requirements.txt, con los siguientes
comandos puedes instalar las dependencias:

1) Activa tu venv (consultar el paso previo).

2) Instala las dependencia con el siguiente comando:

<code>
pip install -r requirements.txt
</code>

### Quiero agregar o actualizar dependencias

Sigue los siguientes pasos si quieres agregar nuevas dependencias ó actualizar las 
ya existentes.

1) Instala o actualiza las dependencias:
<code>
pip install [nombre dependencia]
</code>

2) Atualiza el archivo requirements.txt:

<code>
pip freeze > requirements.txt
</code>

## Conexion con S3
