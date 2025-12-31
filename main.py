import streamlit as st
import pandas as pd
import json
import xml.etree.ElementTree as ET
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns
from utils.file_loader import load_file_to_dataframe
from sklearn.datasets import fetch_openml

st.set_page_config(page_title="Interaktywne algorytmy uczenia maszynowego", initial_sidebar_state="collapsed")
st.markdown("""
    <style>
        [data-testid="stSidebarNav"] {display: none;}
    </style>
""", unsafe_allow_html=True)
  
df_iris = sns.load_dataset("iris")

housing = fetch_openml(
        name="house_prices",
        as_frame=True
        )
df_house = housing.frame

if "df" not in st.session_state:
        st.session_state.df = df_iris

#------------------Początek---------------------
st.title("Interaktywne algorytmy uczenia maszynowego")
st.markdown("Witaj w aplikacji do interaktywnej wizualizacji algorytmów. Eksperymentuj z różnymi parametrami, wczytuj własne dane i obserwuj jak algorytmy klasteryzacji, klasyfikacji i regresji pracują w czasie rzeczywistym!")
st.markdown("**Dostępne algorytmy:**")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**Klasteryzacja**")
    st.page_link("pages/elbow.py", label="Metoda łokcia")
    st.page_link("pages/k-means.py", label="K-means")
    st.page_link("pages/dbscan.py", label="DBSCAN")  
with col2:
    st.markdown("**Klasyfikacja**")
    st.page_link("pages/decision_trees.py", label="Drzewa decyzyjne")
    st.page_link("pages/logistic_regression.py", label="Regresja logistyczna")
with col3:
    st.markdown("**Regresja**")
    st.page_link("pages/lreg.py", label="Regresja liniowa")
    st.page_link("pages/dec_trees_regression.py", label="Drzewa decyzyjne")
    st.page_link("pages/reg_wght.py", label="Regresja ważona")

st.divider()

st.markdown("""
Przed omówieniem samych algorytmu należy wspomnieć o przygotowywaniu danych, czyli standaryzacji danych i technice PCA  
         
**Standaryzacja (StandardScaler)**    
Standaryzacja to proces przekształcania danych tak, aby każda cecha miała średnią równą 0 i odchylenie standardowe równe 1.  
  
**PCA (Principal Component Analysis)**
PCA to technika redukcji wymiarowości, która przekształca dane do nowego układu współrzędnych, gdzie nowe osie (składowe główne) wyjaśniają maksymalną wariancję danych.

**Kiedy jest używane?**
- Gdy masz więcej niż 3 cechy i chcesz wizualizować dane w 2D lub 3D
- Gdy masz bardzo wiele cech (np. 50+) i chcesz przyspieszyć obliczenia
- Gdy cechy są skorelowane i można je zredukować bez utraty wielu informacji
            
*PCA zawsze wiąże się z ryzykiem utraty informmacji*          
            """)

st.markdown("Możesz skorzystać z wbudowanego zbioru Irys, lub wczytać własny.")

with st.expander("Wczytaj własne dane"):
    uploaded_file = st.file_uploader(
    "Wybierz plik (CSV, JSON lub XML)", 
    type=['csv', 'json', 'xml']
)
    if uploaded_file is not None:
        load_file_to_dataframe(uploaded_file)

with st.expander("Dowiedz się więcej o zbiorze Iris"):
     #-------------------EDA---------------------------
     #df = st.session_state.df
    
    if st.button("Wybierz Iris jako podstawowy zbiór"):
        st.session_state.df = df_iris
    st.caption("Iris jest domyślnym zbiorem w przypadku braku innych danych")

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

    st.dataframe(df_iris)

    st.subheader("Statystyki opisowe")
    st.write(df_iris.describe())

    st.subheader("Rozkład gatunków")
    st.write(df_iris["species"].value_counts())

    st.header("Rozłożenie danych:")
    st.header("Boxplot")
    labels = ["Długość kielicha", "Szerokość kielicha", "Długość płatka", "Szerokość płatka"]

    fig, ax = plt.subplots()
    sns.boxplot(data=df_iris, orient="h", ax=ax)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    st.pyplot(fig)

    st.header("Scatterplot")

    fig, ax = plt.subplots()
    sns.scatterplot(data=df_iris, x="sepal_length", y="sepal_width", hue="species", ax=ax)
    ax.set_xlabel("Długość kielicha")
    ax.set_ylabel("Szerokość kielicha")
    st.pyplot(fig)

with st.expander("Dowiedz się więcej o zbiorze House Prices"):
    if st.button("Wybierz *House Prices* jako podstawowy zbiór"):
        st.session_state.df = df_house

    st.markdown("""
    Zbiór House Prices (Ames Housing) to popularny dataset wykorzystywany
    w uczeniu maszynowym do zadań regresji.
    Celem jest przewidywanie ceny sprzedaży domu.

    Zbiór zawiera dane dotyczące **1460 domów** sprzedanych w mieście Ames (Iowa, USA).

    Dla każdej nieruchomości opisano **79 cech**, obejmujących m.in.:
    - lokalizację (np. Neighborhood)
    - wielkość działki i powierzchnię domu
    - jakość i stan techniczny budynku
    - rok budowy i modernizacji
    - liczbę pokoi
    - informacje o piwnicy, garażu i tarasie

    Zmienną docelową jest:
    - SalePrice — cena sprzedaży domu (w USD)

    Zbiór zawiera zarówno cechy numeryczne, jak i kategoryczne,
    a także braki danych, co czyni go dobrym przykładem do nauki
    preprocessingu i inżynierii cech.
                """)
    st.header("Podgląd danych")

    st.dataframe(df_house)

    st.subheader("Statystyki opisowe")
    st.write(df_house.describe())

    st.subheader("Rozkład ceny sprzedaży")

    fig, ax = plt.subplots()
    sns.histplot(df_house["SalePrice"], kde=True, ax=ax)
    ax.set_xlabel("Cena sprzedaży (USD)")
    ax.set_ylabel("Liczba domów")
    st.pyplot(fig)

    st.header("Rozłożenie danych")
    st.subheader("Boxplot wybranych cech")

    num_features = [
        "SalePrice",
        "GrLivArea",
        "TotalBsmtSF",
        "LotArea",
        "OverallQual"
    ]

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=df_house[num_features], orient="h", ax=ax)
    ax.set_yticklabels([
        "Cena sprzedaży",
        "Powierzchnia mieszkalna",
        "Powierzchnia piwnicy",
        "Powierzchnia działki",
        "Ogólna jakość"
    ])
    st.pyplot(fig)

    st.subheader("Scatterplot: powierzchnia vs cena")

    fig, ax = plt.subplots()
    sns.scatterplot(
        data=df_house,
        x="GrLivArea",
        y="SalePrice",
        ax=ax
    )
    ax.set_xlabel("Powierzchnia mieszkalna (m²)")
    ax.set_ylabel("Cena sprzedaży (USD)")
    st.pyplot(fig)