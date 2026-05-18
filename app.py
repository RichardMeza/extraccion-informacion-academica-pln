import re
import io
import pandas as pd
import streamlit as st
import pdfplumber
import spacy
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Extractor Académico PLN",
    page_icon="📄",
    layout="wide"
)

# ---------------------------------------------------------
# Descarga segura de recursos NLTK
# ---------------------------------------------------------
try:
    stop_words = set(stopwords.words("spanish"))
except LookupError:
    nltk.download("stopwords")
    stop_words = set(stopwords.words("spanish"))

# ---------------------------------------------------------
# Carga del modelo spaCy
# ---------------------------------------------------------
@st.cache_resource
def cargar_modelo_spacy():
    return spacy.load("es_core_news_sm")

nlp = cargar_modelo_spacy()

# ---------------------------------------------------------
# Funciones del proyecto
# ---------------------------------------------------------
def extraer_texto_pdf(archivo_pdf):
    texto_total = ""
    with pdfplumber.open(archivo_pdf) as pdf:
        for i, pagina in enumerate(pdf.pages):
            texto = pagina.extract_text()
            if texto:
                texto_total += f"\n--- Página {i+1} ---\n"
                texto_total += texto + "\n"
    return texto_total


def limpiar_texto(texto):
    texto = texto.lower()
    texto = texto.replace("\n", " ")
    texto = re.sub(r"\d+", "", texto)
    texto = re.sub(r"[^\w\s]", "", texto)

    doc = nlp(texto)
    tokens_limpios = []

    for token in doc:
        if token.text not in stop_words and not token.is_space:
            tokens_limpios.append(token.lemma_)

    return " ".join(tokens_limpios)


def extraer_entidades(texto):
    doc = nlp(texto)
    entidades = []

    for ent in doc.ents:
        entidades.append({
            "entidad": ent.text,
            "tipo": ent.label_,
            "inicio": ent.start_char,
            "fin": ent.end_char
        })

    return pd.DataFrame(entidades)


def extraer_anios(texto):
    anios = re.findall(r"\b(19\d{2}|20\d{2})\b", texto)
    return sorted(set(anios))


def extraer_palabras_clave(texto):
    patrones = [
        r"Palabras clave[:\-]?\s*(.*)",
        r"Palabras claves[:\-]?\s*(.*)",
        r"Keywords[:\-]?\s*(.*)"
    ]

    resultados = []
    for patron in patrones:
        coincidencias = re.findall(patron, texto, flags=re.IGNORECASE)
        resultados.extend(coincidencias)

    return resultados[:10]


def extraer_software(texto):
    software = [
        "R", "Python", "SPSS", "Stata", "SAS", "Excel",
        "Power BI", "Tableau", "NVivo", "Atlas.ti",
        "Jamovi", "JASP", "Minitab", "MATLAB"
    ]

    encontrados = []
    for s in software:
        patron = r"\b" + re.escape(s) + r"\b"
        if re.search(patron, texto, flags=re.IGNORECASE):
            encontrados.append(s)

    return encontrados


def extraer_metodos(texto):
    metodos = [
        "regresión lineal", "regresión logística", "anova",
        "chi cuadrado", "chi-cuadrado", "prueba t",
        "correlación", "clustering", "k-means", "pca",
        "análisis de componentes principales", "análisis factorial",
        "árbol de decisión", "random forest", "svm",
        "redes neuronales", "machine learning", "aprendizaje automático",
        "minería de datos", "topic modeling", "lda",
        "procesamiento de lenguaje natural", "pln",
        "bert", "tf-idf", "tfidf", "Redes neuronales convoluonales"
        "redes neuronales recurrentes"
    ]

    encontrados = []
    texto_lower = texto.lower()

    for metodo in metodos:
        if metodo.lower() in texto_lower:
            encontrados.append(metodo)

    return encontrados


def extraer_seccion(texto, inicio, finales):
    patron = rf"{inicio}\s*(.*?)(?:{'|'.join(finales)})"
    resultado = re.search(patron, texto, flags=re.IGNORECASE | re.DOTALL)

    if resultado:
        return resultado.group(1).strip()
    return "No encontrado"


def extraer_keywords_tfidf(texto, n=20):
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words=list(stop_words),
        ngram_range=(1, 2)
    )

    matriz_tfidf = vectorizer.fit_transform([texto])
    palabras = vectorizer.get_feature_names_out()
    puntajes = matriz_tfidf.toarray()[0]

    df_keywords = pd.DataFrame({
        "termino": palabras,
        "tfidf": puntajes
    })

    return df_keywords.sort_values("tfidf", ascending=False).head(n)


def convertir_lista_texto(lista):
    if isinstance(lista, list):
        if len(lista) == 0:
            return "No encontrado"
        return ", ".join([str(x) for x in lista])
    return str(lista)


# ---------------------------------------------------------
# Interfaz Streamlit
# ---------------------------------------------------------
st.title("Sistema Inteligente de Extracción Automática de Información Académica")
st.write("Aplicación de Procesamiento de Lenguaje Natural para extraer información relevante desde documentos académicos en PDF.")

archivo_pdf = st.file_uploader("Sube un documento académico en PDF", type=["pdf"])

if archivo_pdf is not None:
    with st.spinner("Procesando documento..."):
        texto_documento = extraer_texto_pdf(archivo_pdf)

        if len(texto_documento.strip()) == 0:
            st.error("No se pudo extraer texto del PDF. Puede ser un documento escaneado como imagen.")
            st.stop()

        texto_limpio = limpiar_texto(texto_documento)
        df_entidades = extraer_entidades(texto_documento)

        if not df_entidades.empty:
            df_entidades_unicas = (
                df_entidades
                .drop_duplicates(subset=["entidad", "tipo"])
                .sort_values(["tipo", "entidad"])
                .reset_index(drop=True)
            )
        else:
            df_entidades_unicas = pd.DataFrame(columns=["entidad", "tipo", "inicio", "fin"])

        anios = extraer_anios(texto_documento)
        palabras_clave = extraer_palabras_clave(texto_documento)
        software_encontrado = extraer_software(texto_documento)
        metodos_encontrados = extraer_metodos(texto_documento)

        autores = df_entidades[df_entidades["tipo"] == "PER"]["entidad"].drop_duplicates().tolist() if not df_entidades.empty else []
        instituciones = df_entidades[df_entidades["tipo"] == "ORG"]["entidad"].drop_duplicates().tolist() if not df_entidades.empty else []

        resumen = extraer_seccion(
            texto_documento,
            inicio=r"\bResumen\b",
            finales=[r"\bPalabras clave\b", r"\bAbstract\b", r"\bIntroducción\b", r"\bINTRODUCCIÓN\b"]
        )

        abstract = extraer_seccion(
            texto_documento,
            inicio=r"\bAbstract\b",
            finales=[r"\bKeywords\b", r"\bIntroducción\b", r"\bIntroduction\b"]
        )

        introduccion = extraer_seccion(
            texto_documento,
            inicio=r"\bIntroducción\b",
            finales=[r"\bMetodología\b", r"\bMateriales y métodos\b", r"\bMétodo\b", r"\bResultados\b"]
        )

        metodologia = extraer_seccion(
            texto_documento,
            inicio=r"\bMetodología\b|\bMateriales y métodos\b|\bMétodo\b",
            finales=[r"\bResultados\b", r"\bDiscusión\b", r"\bConclusiones\b"]
        )

        resultados = extraer_seccion(
            texto_documento,
            inicio=r"\bResultados\b",
            finales=[r"\bDiscusión\b", r"\bConclusiones\b", r"\bReferencias\b"]
        )

        conclusiones = extraer_seccion(
            texto_documento,
            inicio=r"\bConclusiones\b|\bConclusión\b",
            finales=[r"\bReferencias\b", r"\bBibliografía\b"]
        )

        df_keywords_tfidf = extraer_keywords_tfidf(texto_documento, n=30)
        keywords_tfidf = df_keywords_tfidf["termino"].tolist()

        resumen_final = {
            "Años detectados": convertir_lista_texto(anios),
            "Posibles autores": convertir_lista_texto(autores[:10]),
            "Posibles instituciones": convertir_lista_texto(instituciones[:10]),
            "Palabras clave originales": convertir_lista_texto(palabras_clave),
            "Software detectado": convertir_lista_texto(software_encontrado),
            "Métodos detectados": convertir_lista_texto(metodos_encontrados),
            "Keywords TF-IDF": convertir_lista_texto(keywords_tfidf),
            "Resumen encontrado": "Sí" if resumen != "No encontrado" else "No",
            "Abstract encontrado": "Sí" if abstract != "No encontrado" else "No",
            "Introducción encontrada": "Sí" if introduccion != "No encontrado" else "No",
            "Metodología encontrada": "Sí" if metodologia != "No encontrado" else "No",
            "Resultados encontrados": "Sí" if resultados != "No encontrado" else "No",
            "Conclusiones encontradas": "Sí" if conclusiones != "No encontrado" else "No"
        }

        df_resumen_final = pd.DataFrame({
            "Elemento extraído": resumen_final.keys(),
            "Resultado": resumen_final.values()
        })

    st.success("Documento procesado correctamente")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Resumen general",
        "Entidades NER",
        "Keywords TF-IDF",
        "Secciones",
        "Texto extraído"
    ])

    with tab1:
        st.subheader("Resumen académico consolidado")
        st.dataframe(df_resumen_final, use_container_width=True)

        csv_resumen = df_resumen_final.to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            label="Descargar resumen académico CSV",
            data=csv_resumen,
            file_name="resumen_academico_documento.csv",
            mime="text/csv"
        )

    with tab2:
        st.subheader("Entidades nombradas detectadas")

        if df_entidades_unicas.empty:
            st.warning("No se detectaron entidades nombradas.")
        else:
            tipo_filtrado = st.selectbox(
                "Filtrar por tipo de entidad",
                options=["Todas"] + sorted(df_entidades_unicas["tipo"].unique().tolist())
            )

            if tipo_filtrado != "Todas":
                df_mostrar = df_entidades_unicas[df_entidades_unicas["tipo"] == tipo_filtrado]
            else:
                df_mostrar = df_entidades_unicas

            st.dataframe(df_mostrar, use_container_width=True)

            csv_entidades = df_entidades_unicas.to_csv(index=False, encoding="utf-8-sig")
            st.download_button(
                label="Descargar entidades CSV",
                data=csv_entidades,
                file_name="entidades_detectadas.csv",
                mime="text/csv"
            )

    with tab3:
        st.subheader("Términos más importantes mediante TF-IDF")
        st.dataframe(df_keywords_tfidf, use_container_width=True)

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(df_keywords_tfidf["termino"], df_keywords_tfidf["tfidf"])
        ax.invert_yaxis()
        ax.set_xlabel("Puntaje TF-IDF")
        ax.set_ylabel("Término")
        ax.set_title("Términos más importantes del documento")
        st.pyplot(fig)

        csv_keywords = df_keywords_tfidf.to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            label="Descargar keywords CSV",
            data=csv_keywords,
            file_name="keywords_tfidf.csv",
            mime="text/csv"
        )

    with tab4:
        st.subheader("Secciones académicas extraídas")

        secciones = {
            "Resumen": resumen,
            "Abstract": abstract,
            "Introducción": introduccion,
            "Metodología": metodologia,
            "Resultados": resultados,
            "Conclusiones": conclusiones
        }

        for nombre, contenido in secciones.items():
            with st.expander(nombre):
                st.write(contenido[:5000] if contenido != "No encontrado" else "No encontrado")

    with tab5:
        st.subheader("Texto completo extraído del PDF")
        st.text_area("Texto extraído", texto_documento, height=500)

else:
    st.info("Sube un archivo PDF para iniciar el análisis.")

st.markdown("---")
st.caption("Proyecto de PLN: extracción automática de información académica usando pdfplumber, spaCy, NLTK, TF-IDF y Streamlit.")
