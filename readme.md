# Antnalyzer
Proyecto final de la carrera Ingenieria Electrónica de la Universidad 
Tecnologica Nacional Facultad Regional Paraná

Titulo de Tesis: "Utilización de Inteligencia Artificial para la estimación de la actividad forrajera de las hormigas cortadoras de hojas"

Informe: [./Informe Proyecto Final_QUIROGA-VINZON.pdf](https://github.com/Alanv3514/Antnalyzer/blob/master/Informe%20Proyecto%20Final_QUIROGA-VINZON.pdf)

Guia de creacion de modelo:[./guia-creacion-modelo.pdf](https://github.com/Alanv3514/Antnalyzer/blob/master/documentacion%20extra/guia-creacion-modelo.pdf)

## Descripcion
Se desarrolló un sistema automatizado con el fin de lograr el seguimiento y análisis de comportamiento forrajero de hormigas cortadoras de hojas.
Desarrollado en python 3.9.16. 
Detección de carga realizada con la ayuda de YOLOv8. 
Trackeo realizado con un algoritmo trigonometrico.

## Ejecución del proyecto
### Para entornos de desarrollo

Actualmente se sube el codigo fuente del proyecto por lo que para correrlo 
se necesitan instalar herramientas de desarrollo. 
Herramienta utilizada para manejar entornos de python: 
    - Anaconda3: https://www.anaconda.com/download

Luego de instalar Anaconda3 y haber reiniciado la pc:
* Ejecutar anaconda prompt
* Ejecutar el comando conda create -n <- nombre del entorno -> python=3.9.16
* Colocar el archivo requirements.txt en la carpeta donde estamos posicionados (normalmente C:\Users\Usuario)
* Ejecutar el comando conda activate <- nombre del entorno ->
* Ejecutar pip install -r requeriments.txt
* Utilizando su IDE de preferencia ejecute el archivo antnalyzer.py en el entorno python correspondiente

### Para entornos de usuario final

Encontrarán la última versión en:
https://github.com/Alanv3514/Antnalyzer/releases

## Autores
### Desarrolladores
* [Quiroga, Agustin (UTN FRP)](https://github.com/quiro1297)
* [Vinzon, Eric Alan (UTN FRP)](https://github.com/Alanv3514)
### Director de tesis
* Ing. Maggiolini, Lucas (UTN FRP)
### Co-Director de tesis
* [Dr. Sabattini, Julián Alberto (UNER FCA)](https://github.com/HormigaArgentina)


## Licencia

Este proyecto está licenciado bajo los términos de la [GNU Affero General Public License v3.0](https://www.gnu.org/licenses/agpl-3.0.html).

Consulta el archivo [LICENSE](./LICENSE) para más información.
