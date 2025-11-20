# Reto-ConcentracionIA-Vacas

## Equipo

### Concentrados

Daniel Queijeiro Albo - A01710441

Diego Alfaro Pinto - A01709971

Diego Isaac Fuentes Juvera - A01705506

Jesus Ramirez Delgado - A01274723

Mauricio Anguiano Juarez - A01703337

Luis Adrián Uribe Cruz - A01783129


## Indice de links

- [`Business Understanding`](https://github.com/DiegoAlfaro1/Reto-ConcentracionIA-Vacas/blob/main/Documentaci%C3%B3n%20CRISP-DM/Business%20Understanding.pdf)
- [`Data Understanding`](https://github.com/DiegoAlfaro1/Reto-ConcentracionIA-Vacas/blob/main/Documentaci%C3%B3n%20CRISP-DM/Data%20Preparation.pdf)
- [`Data Preparation`](https://github.com/DiegoAlfaro1/Reto-ConcentracionIA-Vacas/blob/main/Documentaci%C3%B3n%20CRISP-DM/Data%20Understanding.pdf)
- [`Modelado`](https://docs.google.com/document/d/1ij_Z6rscloFkTmMU2RQvtyWRKz3KUUbD3AR-JPQDoO0/edit?usp=drive_link)
- [`Plan de valor ganado`](https://docs.google.com/spreadsheets/d/12ubunIKMkxzAnnMMdO_UpgfDZ-DNtL5uD4X8DdFFTCw/edit?usp=drive_link)
- [`Politica de Datos y Acceso`](https://github.com/DiegoAlfaro1/Reto-ConcentracionIA-Vacas/blob/main/Documentaci%C3%B3n%20CRISP-DM/Politica%20de%20Datos%20y%20Acceso.pdf)

[Link](https://drive.google.com/drive/u/1/folders/1E_7hHlKTfgszawLt3X5qVyLl47rKrEi5) a la carpeta completa de drive

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

## Quiero ejecutrar desde consola ETL

### ETl para modelos de comportamiento y sanidad
<code>
python3 data/etl.py
</code>

## Quiero entrenar desde consola modelos e merito de productividad

### Modelo comportamiento V2
<code>
python3 models/comportamiento_rf_v2.py
</code>

### Modelo de sanidad V2
<code>
python3 models/sanidad_iso_v2.py
</code>

### Merito de productvidad
<code>
python3 util/merito_productivo.py
</code>

## Quiero ejecutrar la integreacion desde consola

<code>
python3 integration_v1.py --csv [path csv ]--cow-id [id]
</code>

## Conexion con S3