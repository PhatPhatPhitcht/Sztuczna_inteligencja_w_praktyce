import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from utils.file_loader import load_file_to_dataframe

st.set_page_config(page_title="Interaktywne algorytmy uczenia maszynowego", initial_sidebar_state="collapsed")
st.markdown("""
    <style>
        [data-testid="stSidebarNav"] {display: none;}
    </style>
""", unsafe_allow_html=True)

if 'df' not in st.session_state:
    st.error("Brak danych! Najpierw załaduj stronę główną")
    st.stop()

st.page_link("pages/k-means.py", label="⬅️ Powrót do K-means")

st.subheader("Metoda łokcia")
st.markdown("""
**Metoda łokcia (Elbow Method)**

Metoda łokcia służy do określenia **optymalnej liczby klastrów (k)** w algorytmie *k-means*.

Dla kolejnych wartości *k*:
- trenujemy model k-means
- obliczamy **WCSS (Within-Cluster Sum of Squares)** - miarę rozrzutu punktów
  wewnątrz klastrów

Na wykresie:
- **oś X** - liczba klastrów (*k*)
- **oś Y** - WCSS (inertia)

Optymalną wartością *k* jest zwykle punkt,w którym tempo spadku WCSS **gwałtownie maleje** - tworząc charakterystyczny „łokieć”.
"""
)
st.divider()

with st.expander("Wczytaj inne dane"):
    uploaded_file = st.file_uploader(
    "Wybierz plik (CSV, JSON lub XML)", 
    type=['csv', 'json', 'xml']
)
    if uploaded_file is not None:
        load_file_to_dataframe(uploaded_file)

with st.expander("Podgląd danych"):
    st.dataframe(st.session_state.df.head())

data = st.session_state.df

# tylko kolumny numeryczne
numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()

if len(numeric_columns) < 2:
    st.error("Dane muszą zawierać co najmniej 2 kolumny numeryczne.")
    st.stop()

with st.expander("Wybór zmiennych do analizy"):
    st.markdown("**Wybierz zmienne numeryczne używane do obliczenia metody łokcia:**")

    selected_columns = st.multiselect(
        "Dostępne zmienne numeryczne:",
        options=numeric_columns,
        default=numeric_columns[:min(2, len(numeric_columns))],
        help="Wybierz co najmniej 2 zmienne"
    )

    if len(selected_columns) < 2:
        st.warning("Wybierz co najmniej 2 zmienne do analizy.")
        st.stop()

data_selected = data[selected_columns]

scaler = StandardScaler()
X = scaler.fit_transform(data_selected)

st.markdown("**Parametry metody łokcia**")

col1, col2 = st.columns(2)
with col1:
    k_min = st.number_input("Minimalna liczba klastrów (k)", 1, 20, 1)
with col2:
    k_max = st.number_input("Maksymalna liczba klastrów (k)", 2, 30, 10)

if k_min >= k_max:
    st.error("Minimalna liczba klastrów musi być mniejsza niż maksymalna.")
    st.stop()

wcss = []

if st.button("Zobacz optymalną liczbę klastrów:", type="primary"):
    for k in range(k_min, k_max + 1):
        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    # różnice WCSS 
    wcss_diff = np.diff(wcss)

    st.subheader("Wykres metody łokcia")

    fig, ax = plt.subplots()
    ax.plot(range(k_min, k_max + 1), wcss, marker="o")
    ax.set_xlabel("Liczba klastrów (k)")
    ax.set_ylabel("WCSS (Inertia)")
    ax.set_title("Metoda łokcia")
    ax.grid(True)
    st.pyplot(fig)


    st.markdown("""
    **Jak wybrać optymalną wartość k?**     

    - Zwróć uwagę na punkt, w którym **krzywa zaczyna się wypłaszczać**
    - Jest to miejsce, gdzie **dalsze zwiększanie liczby klastrów nie daje dużej poprawy**
    - Najczęściej jest to punkt o **największym „załamaniu” krzywej**
    """
    )

    st.markdown("**Zmiana WCSS pomiędzy kolejnymi wartościami k:**")
    diff_df = pd.DataFrame({
        "k → k+1": [f"{k} → {k+1}" for k in range(k_min, k_max)],
        "Spadek WCSS": wcss_diff
    })

    st.dataframe(diff_df)
