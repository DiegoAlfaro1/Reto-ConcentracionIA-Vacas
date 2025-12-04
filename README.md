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

## 📂 Documentación CRISP-DM

Cada archivo puede consultarse haciendo clic en su nombre:

- [Despliegue](Documentación%20CRISP-DM/Despliegue.pdf)
- [Entendimiento de negocio](Documentación%20CRISP-DM/Entendimiento%20de%20negocio.pdf)
- [Entendimiento de datos](Documentación%20CRISP-DM/Entendimiento%20de%20datos.pdf)
- [Evaluación](Documentación%20CRISP-DM/Evaluacion.pdf)
- [Modelado](Documentación%20CRISP-DM/Modelado.pdf)
- [Política de Datos y Acceso](Documentación%20CRISP-DM/Politica%20de%20Datos%20y%20Acceso.pdf)
- [Preparación de datos](Documentación%20CRISP-DM/Preparacion%20de%20datos.pdf)
- [Reporte final](Documentación%20CRISP-DM/Reporte%20final.pdf)

## Correcciones

### Respuesta a retroalimentación y acciones correctivas implementadas

Con base en la retroalimentación recibida, se realizaron ajustes estructurales, metodológicos y documentales para asegurar alineación con CRISP-DM y claridad en la trazabilidad del proyecto.

---

### Finalización de fases CRISP-DM  
Se completaron formalmente todas las fases y sus entregables:

- Entendimiento de negocio  
- Entendimiento de datos  
- Preparación de datos  
- Modelado  
- Evaluación  
- Despliegue  

---

### Adaptaciones metodológicas documentadas  
Cada fase ahora incluye una sección donde se describen explícitamente:

- Cómo se aplicó CRISP-DM  
- Qué actividades fueron adaptadas  
- Justificación de dichas adaptaciones  

---

### Estructura documental corregida  
El índice y orden de los documentos fue reorganizado para reflejar adecuadamente el flujo de CRISP-DM y mejorar navegabilidad.

---

### Objetivos y criterios corregidos y validados  
- El **objetivo de minería de datos** fue revisado para ser una traducción técnica directa del objetivo de negocio.  
- Los **criterios de éxito de minería de datos** fueron corregidos, validados y traducidos 1 a 1 desde los criterios de negocio, asegurando capacidad de evaluación del desempeño de modelos.

---

### Documentación de riesgos y responsables ampliada  
Se incorporaron responsables por riesgo, magnitud y contexto, conforme a observaciones metodológicas.

---

### Repositorio actualizado como punto central  
El README fue actualizado para:

- Enlazar cada fase documentada  
- Facilitar acceso directo a los entregables  

---

Con estas correcciones, el proyecto cuenta ahora con trazabilidad metodológica completa, criterios medibles de evaluación, y documentación estructurada conforme a mejores prácticas.


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

1. Activa tu venv (consultar el paso previo).

2. Instala las dependencia con el siguiente comando:

<code>
pip install -r requirements.txt
</code>

### Quiero agregar o actualizar dependencias

Sigue los siguientes pasos si quieres agregar nuevas dependencias ó actualizar las
ya existentes.

1. Instala o actualiza las dependencias:
   <code>
   pip install [nombre dependencia]
   </code>

2. Atualiza el archivo requirements.txt:

<code>
pip freeze > requirements.txt
</code>

## Quiero ejecutrar desde consola ETL

### ETl para modelos de comportamiento y sanidad

<code>
python3 data/etl.py
</code>

## Quiero entrenar desde consola modelos y merito de productividad

### Modelo comportamiento V2

<code>
python3 models/comportamiento_rf_v2.py
</code>

### Modelo de sanidad V2

<code>
python models/sanidad_iso_v2.py --input datos/sessions_health.csv --contamination 0.02
</code>

### Merito de productvidad

<code>
python3 util/merito_productivo.py
</code>

## Quiero ejecutrar la integreacion desde consola

<code>
python3 integration_v1.py --csv [path csv ]--cow-id [id]
</code>
