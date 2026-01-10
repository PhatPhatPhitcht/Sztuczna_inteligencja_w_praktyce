import pandas as pd
import streamlit as st
import json
import xml.etree.ElementTree as ET

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
            for col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    pass
            st.success(f"Wczytano plik XML: {uploaded_file.name}")
            
        else:
            st.error(f"Nieobsługiwany format pliku: {file_extension}")
            return False
        
        # Sprawdzenie brakujących danych
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            missing_cols = df.columns[df.isnull().any()].tolist()
            st.error(f"Znaleziono {missing_count} brakujących wartości w kolumnach: {', '.join(missing_cols)}")
            st.error("Plik zawiera brakujące dane. Proszę uzupełnić dane lub usunąć wiersze z brakami.")
            return False
        
        st.session_state.df = df
        
        return True
        
    except Exception as e:
        st.error(f"Błąd podczas wczytywania pliku, spróbuj wczytać inne dane")
        return False