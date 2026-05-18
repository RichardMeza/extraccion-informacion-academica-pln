# Sistema Inteligente de Extracción Automática de Información Académica

Aplicación desarrollada con Streamlit para procesar documentos académicos en PDF y extraer información mediante técnicas de Procesamiento de Lenguaje Natural.

## Funcionalidades

- Carga de documentos PDF.
- Extracción de texto con pdfplumber.
- Limpieza y preprocesamiento textual.
- Reconocimiento de entidades nombradas con spaCy.
- Extracción de años, autores, instituciones, software y métodos.
- Extracción de secciones académicas.
- Extracción de keywords mediante TF-IDF.
- Descarga de resultados en CSV.

## Ejecución local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Despliegue en Streamlit Cloud

1. Crear un repositorio en GitHub.
2. Subir los archivos `app.py`, `requirements.txt` y `README.md`.
3. Entrar a Streamlit Cloud.
4. Seleccionar el repositorio.
5. Indicar como archivo principal: `app.py`.
6. Presionar Deploy.

## Tecnologías usadas

- Python
- Streamlit
- pdfplumber
- spaCy
- NLTK
- scikit-learn
- matplotlib
