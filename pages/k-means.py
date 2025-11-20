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

df = st.session_state.df # Pamiętaj żebu tu wrócić!!!--------------
#from sklearn.datasets import load_iris
#iris = load_iris()
#df = pd.DataFrame(iris.data, columns=iris.feature_names)
#-------------------------------------------------------------------

# kolumny numeryczne i usuń braki
data = df.select_dtypes(include=[np.number]).dropna()

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

pca = PCA(n_components=2)
data_2d = pca.fit_transform(data_scaled)


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

st.header("K-means")
st.markdown("""
Przed omówieniem samego algorytmu należy wspomnieć o przygotowywaniu danych, czyli standaryzacji danych i technice PCA  
         
**Standaryzacja (StandardScaler)**    
Standaryzacja to proces przekształcania danych tak, aby każda cecha miała średnią równą 0 i odchylenie standardowe równe 1.  
  
**Dlaczego jest kluczowa dla K-means?**
- K-means używa odległości euklidesowej do przypisywania punktów do klastrów
- Cechy o większych wartościach (np. pensja: 20000-80000) dominowałyby cechy o mniejszych (np. wiek: 20-60)
- Bez standaryzacji algorytm skupiałby się głównie na cechach o największym zakresie wartości
- Po standaryzacji wszystkie cechy mają równy wpływ na wynik klasteryzacji  

**PCA (Principal Component Analysis)**
PCA to technika redukcji wymiarowości, która przekształca dane do nowego układu współrzędnych, gdzie nowe osie (składowe główne) wyjaśniają maksymalną wariancję danych.
PCA NIE jest wymagane dla K-means. K-means działa bezpośrednio na oryginalnych danych. Nie ma potrzeby na PCA dla samego algorytmu klasteryzacji.

**Kiedy jest używane?**
- Gdy masz więcej niż 3 cechy i chcesz wizualizować dane w 2D lub 3D
- Gdy masz bardzo wiele cech (np. 50+) i chcesz przyspieszyć obliczenia
- Gdy cechy są skorelowane i można je zredukować bez utraty wielu informacji
            
*PCA zawsze wiąże się z ryzykiem utraty informmacji*          
            """)
st.subheader("Algorytm K-means")
st.markdown("""
**Co to jest K-means?**
            
K-means to iteracyjny algorytm klasteryzacji należący do kategorii uczenia nienadzorowanego. Jego celem jest partycjonowanie zbioru danych na K rozłącznych klastrów poprzez minimalizację wewnątrzklasterowej sumy kwadratów odległości.

**Inicjalizacja**

Algorytm rozpoczyna się od losowego wyboru K punktów ze zbioru danych jako początkowych centroidów. Wybór początkowych centroidów ma znaczący wpływ na zbieżność algorytmu i jakość końcowego rozwiązania, ponieważ funkcja celu J jest nie-wypukła i może zawierać wiele lokalnych minimów.

**Przypisanie punktów**
            
Każdy punkt jest przypisywany do najbliższego centroidu zgodnie z metryką odległości. Najczęściej używaną metryką jest odległość euklidesowa.
            
**Aktualizacja punktów**

Po przypisaniu wszystkich punktów do klastrów, centroidy są aktualizowane jako środki masy (centroidy geometryczne) punktów należących do każdego klastra. Ta operacja przesuwa centroid w kierunku "centrum" punktów w klastrze, minimalizując tym samym funkcję celu J dla bieżących przypisań.
   
**Algorytm kończy działanie gdy spełniony jest jeden z warunków:**

- Brak zmian w przypisaniach: żaden punkt nie zmienił klastra między iteracjami
- Stabilność centroidów: zmieniają się mniej niż o ustalony, bardzo mały próg.
- Osiągnięcie maksymalnej liczby iteracji: zabezpieczenie przed nieskończonym działaniem
            
Poniżej możesz zobaczyć iteracyjne zmiany przy obliczaniu klastrów na wybranym zbiorze danych. Wybierz liczbę klastrów, którą chcesz obliczyć, oraz liczbę iteracji którą chcesz zobaczyć.

*zmiany pomiędzy pojedyńczymi iteracjami mogą być minimalne, dlatego wykresy wyświetlane będą co kilka z nich.*
            """)
col1, col2 = st.columns(2)              

max_iter = 100

with col1:
    n_clusters = st.slider("Liczba klastrów", 2, 10, 3)
with col2:
    max_plots = st.slider("Liczba wykresów", 3, 10, 5)

# Uruchom K-means
if st.button("Uruchom K-means"):
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
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                      c=colors[k], label=f'Klaster {k+1}', 
                      alpha=0.6, s=50)

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

        #ax.set_xlabel(f'PCA Składowa 1 ({pca.explained_variance_ratio_[0]:.1%} wariancji)', fontsize=12)
        #ax.set_ylabel(f'PCA Składowa 2 ({pca.explained_variance_ratio_[1]:.1%} wariancji)', fontsize=12)
        ax.set_title(f'K-means Clustering - Iteracja {iteration+1}\nDataset: Irys ({len(data_2d)} punktów, {n_clusters} klastry)', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)
    #---------------------Koniec Wykresów----------------------
