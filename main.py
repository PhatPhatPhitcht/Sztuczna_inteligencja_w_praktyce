import streamlit as st
import pandas as pd
import json
import xml.etree.ElementTree as ET
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns

def load_file_to_dataframe(uploaded_file):
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'csv':
            df = pd.read_csv(uploaded_file)
            st.success(f"Wczytano plik CSV: {uploaded_file.name}")
            
        elif file_extension == 'json':
            content = uploaded_file.read().decode('utf-8')
            data = json.loads(content)
            
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                df = pd.DataFrame([data])
            else:
                df = pd.DataFrame(data)
            
            st.success(f"Wczytano plik JSON: {uploaded_file.name}")
            
        elif file_extension == 'xml':
            tree = ET.parse(uploaded_file)
            root = tree.getroot()

            data = []
            for child in root:
                row = {}
                for elem in child:
                    row[elem.tag] = elem.text
                data.append(row)
            
            df = pd.DataFrame(data)
            st.success(f"Wczytano plik XML: {uploaded_file.name}")
            
        else:
            st.error(f"Nieobsługiwany format pliku: {file_extension}")
            return False

        st.session_state.df = df

        st.write(f"**Wymiary:** {df.shape[0]} wierszy, {df.shape[1]} kolumn")
        st.dataframe(df.head())
        
        return True
        
    except Exception as e:
        st.error(f"Błąd podczas wczytywania pliku: {str(e)}")
        return False

if "df" not in st.session_state:
        st.session_state.df = sns.load_dataset("iris")

#------------------Początek---------------------
st.title("Interaktywne Algorytmy Uczenia Maszynowego")
st.markdown("Witaj w aplikacji do interaktywnej wizualizacji algorytmów. Eksperymentuj z różnymi parametrami, wczytuj własne dane i obserwuj jak algorytmy klasteryzacji, klasyfikacji i regresji pracują w czasie rzeczywistym!")
st.markdown("**Dostępne algorytmy:**")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    **Klasteryzacja**
    - K-means
    - x
    """)
with col2:
    st.markdown("""
    **Klasyfikacja**
    - x
    - x
    """)
with col3:
    st.markdown("""
    **Regresja**
    - x
    - x
    """)

st.markdown("Możesz skorzystać z wbudowanego zbioru Irys, lub wczytać własny.")
st.caption("Obsługiwane formaty to: csv, json oraz xml")

d1, d2 = st.columns(2)

with d1:
    b1 = st.button("Załaduj własne dane")

with d2:
    b2 = st.button("Dowiedz się więcej o zbiorze Irys")

output = st.container()

with output:
    if b1:
        uploaded_file = st.file_uploader(
            "Wybierz plik (CSV, JSON lub XML)", 
            type=['csv', 'json', 'xml']
        )

        if uploaded_file is not None:
            load_file_to_dataframe(uploaded_file)

            st.header("Podgląd danych")

            st.dataframe(st.session_state.df)

    elif b2:
        #-------------------EDA---------------------------
        df = st.session_state.df

        st.markdown("""
        Zbiór Iris to klasyczny dataset używany w statystyce i uczeniu maszynowym.
        Zawiera 150 próbek trzech gatunków irysów:
        - Iris setosa
        - Iris versicolor
        - Iris virginica

        Dla każdej próbki zmierzono cztery cechy morfologiczne:
        - sepal_length — Długość kielicha (cm)
        - sepal_width — Szerokość kielicha (cm)
        - petal_length — Długość płatka (cm)
        - petal_width — Szerokość płatka (cm)

        Zbiór jest niewielki, dobrze zbalansowany (50 próbek na gatunek).
        
                    """)

        st.header("Podgląd danych")

        st.dataframe(df)

        st.subheader("Statystyki opisowe")
        st.write(df.describe())

        st.subheader("Rozkład gatunków")
        st.write(df["species"].value_counts())

        st.header("Rozłożenie danych:")
        st.header("Boxplot")
        labels = ["Długość kielicha", "Szerokość kielicha", "Długość płatka", "Szerokość płatka"]

        fig, ax = plt.subplots()
        sns.boxplot(data=df, orient="h", ax=ax)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        st.pyplot(fig)

        st.header("Scatterplot")

        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x="sepal_length", y="sepal_width", hue="species", ax=ax)
        ax.set_xlabel("Długość kielicha")
        ax.set_ylabel("Szerokość kielicha")
        st.pyplot(fig)

