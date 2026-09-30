# Aplicación de segmentación Online Retail

Requiere Python 3.12. Extraer todos los archivos juntos.
En una terminal, dentro de esta carpeta:
python -m pip install -r requirements.txt
python -m streamlit run app.py
Abrir http://localhost:8501 en el navegador de esa computadora.

Para Streamlit Community Cloud: subir esta carpeta completa a un repositorio
de GitHub, crear una app en https://share.streamlit.io, seleccionar repositorio,
rama y app.py; elegir Python 3.12 y desplegar. Guardar la URL real resultante.
Si la carpeta está en un subdirectorio, indicar esa ruta de app.py.
El modelo joblib requiere las versiones de requirements.txt.

Uso: seleccionar un cliente existente o ingresar R (días), F (facturas)
y M (GBP) y pulsar Asignar segmento. Consultar perfil, ubicación PCA y métricas.
No cargar modelos joblib de terceros no confiables.

Entrega académica: tomar capturas de la app funcionando, anotar URL real
si se publica y subir el ipynb a Google Colab compartiéndolo con la docente.
La URL localhost solo funciona en la computadora donde se inicia la app.
