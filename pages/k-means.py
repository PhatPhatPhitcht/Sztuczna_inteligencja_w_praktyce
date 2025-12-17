import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
from matplotlib.animation import FuncAnimation
import time
import streamlit as st

st.set_page_config(page_title="Interaktywne algorytmy uczenia maszynowego", initial_sidebar_state="collapsed")
st.markdown("""
    <style>
        [data-testid="stSidebarNav"] {display: none;}
    </style>
""", unsafe_allow_html=True)

if 'df' not in st.session_state:
    st.error("Brak danych! Najpierw załaduj stronę główną")
    st.stop()


df = st.session_state.df

# kolumny numeryczne i usuń braki
data = df.select_dtypes(include=[np.number]).dropna()

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

pca = PCA(n_components=2)
data_2d = pca.fit_transform(data_scaled)
explained_variance = pca.explained_variance_ratio_

class KMeansIterative:
    def __init__(self, n_clusters=3, max_iter=100, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.history = []
        
    def fit(self, X):
        np.random.seed(self.random_state)
        
        # Inicjalizacja
        idx = np.random.choice(len(X), self.n_clusters, replace=False)
        centroids = X[idx]
        
        # Stan początkowy
        labels = self._assign_clusters(X, centroids)
        self.history.append({
            'centroids': centroids.copy(),
            'labels': labels.copy(),
            'iteration': 0
        })
        #--------------------Główna pętla---------------------
        for i in range(self.max_iter):
            # Przypisanie punktów do najbliższych centroidów
            labels = self._assign_clusters(X, centroids)
            
            # Obliczenie nowych centroidów
            new_centroids = np.array([
                X[labels == k].mean(axis=0) 
                for k in range(self.n_clusters)
            ])
            
            # Zapisz stan po iteracji
            self.history.append({
                'centroids': new_centroids.copy(),
                'labels': labels.copy(),
                'iteration': i + 1
            })
            
            # Sprawdź konwergencję
            if np.allclose(centroids, new_centroids, rtol=1e-4):
                #print(f"Konwergencja osiągnięta po {i+1} iteracjach")
                break
                
            centroids = new_centroids
        #-----------------------------------------------------
        #Wyniki
        self.centroids = centroids
        self.labels = labels
        return self
    
    def _assign_clusters(self, X, centroids):
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)

#----------------------------Początek streamlita--------------------------------

st.page_link("main.py", label="⬅️ Powrót do strony głównej")

st.subheader("Algorytm K-means")
st.markdown("""
Algorytm K-Means grupuje dane, próbując podzielić próbki na n grup o równej wariancji, minimalizując kryterium znane jako inercja (ang. inertia) lub suma kwadratów wewnątrz klastrów (patrz poniżej). Algorytm wymaga podania liczby klastrów. Dobrze skalowalny do dużej liczby próbek, jest szeroko stosowany w wielu dziedzinach.
            
Algorytm K-Means dzieli zbiór próbek na rozłączne klastry, z których każdy opisany jest średnią próbek w tym klastrze. Te średnie są nazywane centroidami; zazwyczaj nie są to rzeczywiste punkty ze zbioru, chociaż leżą w tej samej przestrzeni.

Celem algorytmu K-Means jest wybór centroidów minimalizujących inercję, czyli sumę kwadratów odchyleń wewnątrz klastrów.

Inercję można traktować jako miarę spójności wewnątrz klastrów. Ma jednak kilka wad:
-Zakłada, że klastry są wypukłe i izotropowe, co nie zawsze jest prawdą. Źle radzi sobie z klastrami wydłużonymi lub o nieregularnych kształtach.
-Nie jest znormalizowana: wiadomo tylko, że mniejsze wartości są lepsze, a 0 jest idealne. W bardzo wysokich wymiarach odległości euklidesowe rosną (tzw. klątwa wymiarowości). Redukcja wymiarów, np. PCA, przed klasteryzacją k-means, może złagodzić ten problem i przyspieszyć obliczenia.

Algorytm k-means jest często nazywany algorytmem Lloyda. Składa się z trzech kroków. Pierwszy wybiera centroidy początkowe — najprostsza metoda to losowy wybór próbek z zestawu danych. Następnie algorytm wykonuje dwie fazy w pętli:
- Przypisanie każdej próbki do najbliższego centroidu.
- Wyznaczenie nowych centroidów jako średnich próbek przypisanych do poprzednich centroidów.

Różnica między starymi a nowymi centroidami jest obliczana i algorytm powtarza te kroki, aż do osiągnięcia progu — innymi słowy, gdy centroidy przestają się istotnie zmieniać.

Iteracje zwykle kończą się, gdy względny spadek funkcji celu jest mniejszy niż tolerancja; w tej implementacji — gdy centroidy przesuwają się o mniej niż wartość tolerancji.

[Dowiedz się więcej:](https://scikit-learn.org/stable/modules/clustering.html#k-means)            

Poniżej możesz zobaczyć iteracyjne zmiany przy obliczaniu klastrów na wybranym zbiorze danych. Wybierz liczbę klastrów, którą chcesz obliczyć, oraz liczbę iteracji którą chcesz zobaczyć.

*zmiany pomiędzy pojedyńczymi iteracjami mogą być minimalne, dlatego wykresy wyświetlane będą co kilka z nich.*            
""")

st.divider()

col1, col2 = st.columns(2)              

max_iter = 100

with col1:
    n_clusters = st.slider("Liczba klastrów", 2, 10, 3)
with col2:
    max_plots = st.slider("Liczba wykresów", 3, 10, 5)

# Uruchom K-means
if st.button("Uruchom K-means", type="primary"):
    kmeans = KMeansIterative(n_clusters=n_clusters, max_iter=max_iter)
    kmeans.fit(data_2d)

    # Wszystkie iteracje
    st.subheader(f"Historia K-means ({len(kmeans.history)} iteracji)")

    #----------------------Wybieranie 5 iteracji do pokazania---------------------
    total_iterations = len(kmeans.history)

    if total_iterations <= max_plots:
        frames_to_show = list(range(total_iterations))
    else:
        step = total_iterations // (max_plots - 1)
        frames_to_show = list(range(0, total_iterations, step))
        if frames_to_show[-1] != total_iterations - 1:
            frames_to_show.append(total_iterations - 1)

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', 
          '#F38181', '#AA96DA', '#FCBAD3', '#A8E6CF', '#FFD3B6']

    markers = ['o', 's', '^', 'D', 'P', 'X', 'v', '<', '>', '*']

    #---------------------Wykresy----------------------
    for frame in frames_to_show:
        state = kmeans.history[frame]
        centroids = state['centroids']
        labels = state['labels']
        iteration = state['iteration']

        # Nowy wykres dla każdej iteracji
        fig, ax = plt.subplots(figsize=(10, 8))

        # Punkty
        for k in range(n_clusters):
            cluster_points = data_2d[labels == k]
            ax.scatter(
                            cluster_points[:, 0],
                            cluster_points[:, 1],
                            c=colors[k],
                            marker=markers[k % len(markers)],
                            label=f'Klaster {k+1}',
                            alpha=0.6,
                            s=60,
                            edgecolors='black',
                            linewidths=0.5
                        )

        # Centroidy
        ax.scatter(centroids[:, 0], centroids[:, 1], 
                  c='black', marker='X', s=300, 
                  edgecolors='white', linewidths=2,
                  label='Centroidy', zorder=5)

        # Numery do centroidów
        for k, centroid in enumerate(centroids):
            ax.text(centroid[0], centroid[1], str(k+1), 
                   color='white', fontsize=12, fontweight='bold',
                   ha='center', va='center', zorder=6)

        ax.set_xlabel(f'PC1 ({explained_variance[0]*100:.1f}% wariancji)')
        ax.set_ylabel(f'PC2 ({explained_variance[1]*100:.1f}% wariancji)')

        ax.set_title(f'K-means Clustering - Iteracja {iteration+1}\nDataset: Irys ({len(data_2d)} punktów, {n_clusters} klastry)', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

    st.markdown("""
    **Procent wariacji** oznacza, że ten wymiar zachowuje taką ilość informacji o zróżnicowaniu danych. 
                PC1 + PC2 nie muszą dawać 100%, ale im bliżej 100%, tym lepiej zachowana struktura danych po redukcji wymiarów.
    """)
    #---------------------Koniec Wykresów----------------------
