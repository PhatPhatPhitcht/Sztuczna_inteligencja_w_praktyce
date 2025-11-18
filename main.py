import streamlit as st
import pandas as pd
import json
import xml.etree.ElementTree as ET
from io import StringIO
from sklearn.datasets import load_iris

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
        iris = load_iris()
        st.session_state.df = pd.DataFrame(iris.data, columns=iris.feature_names)

st.title("Tytuł") #tytuł

uploaded_file = st.file_uploader(
    "Wybierz plik (CSV, JSON lub XML)", 
    type=['csv', 'json', 'xml']
)

if uploaded_file is not None:
    load_file_to_dataframe(uploaded_file)
