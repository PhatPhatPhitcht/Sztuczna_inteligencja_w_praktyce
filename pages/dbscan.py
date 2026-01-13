import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import streamlit as st
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

if 'df' in st.session_state:
    df = st.session_state.df

data = df.select_dtypes(include=[np.number]).dropna()

class DBSCANIterative:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.history = []
        
    def fit(self, X):
        n = len(X)
        labels = np.full(n, -2)  
        # -2 = nieodwiedzony, -1 = szum, >=0 = klaster
        core_points = np.zeros(n, dtype=bool)
        cluster_id = 0
        
        # Stan początkowy
        self.history.append({
            'labels': labels.copy(),
            'core_points': core_points.copy(),
            'iteration': 0,
            'description': 'Stan początkowy - wszystkie punkty nieprzetworzone',
            'current_point': None,
            'cluster_id': -1
        })
        
        #-------------------Główna pętla DBSCAN---------------------
        for point_idx in range(n):
            if labels[point_idx] != -2:
                continue
            
            neighbors = self._find_neighbors(X, point_idx)
            
            if len(neighbors) < self.min_samples:
                labels[point_idx] = -1
            else:
                core_points[point_idx] = True
                # Stan przed
                self.history.append({
                    'labels': labels.copy(),
                    'core_points': core_points.copy(),
                    'iteration': len(self.history),
                    'description': f'Znaleziono punkt rdzeniowy #{point_idx} - {len(neighbors)} sąsiadów',
                    'current_point': point_idx,
                    'cluster_id': cluster_id
                })
                
                labels = self._expand_cluster(X, labels, core_points, point_idx, neighbors, cluster_id)
                # Stan po 
                self.history.append({
                    'labels': labels.copy(),
                    'core_points': core_points.copy(),
                    'iteration': len(self.history),
                    'description': f'Rozszerzono klaster {cluster_id} - {np.sum(labels == cluster_id)} punktów',
                    'current_point': None,
                    'cluster_id': cluster_id
                })
                
                cluster_id += 1
        #---------------------------------------------------------------------
        # Wyniki
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(labels == -1)
        self.history.append({
            'labels': labels.copy(),
            'core_points': core_points.copy(),
            'iteration': len(self.history),
            'description': f'Wynik końcowy: {n_clusters} klastrów, {n_noise} punktów szumu',
            'current_point': None,
            'cluster_id': -1
        })
        
        self.labels = labels
        self.core_points = core_points
        return self
    
    def _find_neighbors(self, X, point_idx):
        distances = np.sqrt(((X - X[point_idx])**2).sum(axis=1))
        neighbors = np.where((distances <= self.eps) & (np.arange(len(X)) != point_idx))[0]
        return neighbors.tolist()
    
    def _expand_cluster(self, X, labels, core_points, point_idx, neighbors, cluster_id):
        labels[point_idx] = cluster_id
        
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            
            if labels[neighbor_idx] == -2: 
                labels[neighbor_idx] = cluster_id
                neighbor_neighbors = self._find_neighbors(X, neighbor_idx)
                
                if len(neighbor_neighbors) >= self.min_samples:
                    core_points[neighbor_idx] = True
                    neighbors.extend(neighbor_neighbors)
            
            elif labels[neighbor_idx] == -1:
                labels[neighbor_idx] = cluster_id
            
            i += 1
        
        return labels

#--------------------------------Streamlit------------------------------------

st.page_link("main.py", label="⬅️ Powrót do strony głównej")

st.subheader("Algorytm DBSCAN")
st.markdown("""
**Algorytm DBSCAN** traktuje klastry jako obszary o dużej gęstości oddzielone obszarami o małej gęstości. Dzięki temu ogólnemu 
podejściu klastry wykrywane przez DBSCAN mogą mieć dowolny kształt, w przeciwieństwie do algorytmu k-means, który zakłada, 
że klastry mają kształt wypukły. Centralnym pojęciem DBSCAN jest **próbka rdzeniowa** (ang. *core sample*), czyli próbka 
znajdująca się w obszarze o dużej gęstości. Klaster stanowi zbiór próbek rdzeniowych leżących blisko siebie oraz zbiór próbek 
nierdzeniowych, które są sąsiadami próbek rdzeniowych, lecz same nimi nie są.  

Algorytm posiada dwa parametry: **min_samples** oraz **eps**, które formalnie definiują pojęcie „gęstości”. 
Wyższa wartość **min_samples** lub niższa wartość **eps** oznaczają większą wymaganą gęstość do utworzenia klastra.

Dokładniej, **próbka rdzeniowa** to taka próbka, dla której istnieje co najmniej **min_samples** innych próbek 
w odległości **eps**, uznawanych za jej sąsiadów. Oznacza to, że próbka ta znajduje się w gęstym obszarze przestrzeni. 
Klaster jest zbiorem próbek rdzeniowych, które można uzyskać rekurencyjnie, rozpoczynając od dowolnej próbki rdzeniowej, 
wyszukując wszystkich jej sąsiadów będących próbkami rdzeniowymi, a następnie powtarzając ten proces dla kolejnych 
próbek. Klaster zawiera również próbki nierdzeniowe — są to próbki będące sąsiadami próbek rdzeniowych, lecz same 
niebędące próbkami rdzeniowymi. Intuicyjnie leżą one na obrzeżach klastra.

Każda próbka rdzeniowa należy do klastra. Próbka, która nie jest próbką rdzeniową i znajduje się w odległości większej 
niż **eps** od każdej próbki rdzeniowej, jest uznawana za **punkt odstający (outlier)**.

Parametr **min_samples** wpływa głównie na odporność algorytmu na szum — w przypadku dużych i zaszumionych zbiorów danych 
często warto go zwiększyć. Natomiast parametr **eps** jest kluczowy i zwykle nie powinien pozostawać na wartości domyślnej. 
Określa on lokalne sąsiedztwo punktów: zbyt mała wartość **eps** sprawia, że większość danych pozostaje niesklasteryzowana 
(oznaczona jako -1, czyli „szum”), a zbyt duża prowadzi do łączenia pobliskich klastrów w jeden, a w skrajnym przypadku 
może spowodować, że cały zbiór zostanie zwrócony jako pojedynczy klaster.

[Dowiedz się więcej](https://scikit-learn.org/stable/modules/clustering.html#dbscan)

Poniżej możesz zobaczyć, jak zmieniają się wyniki klasteryzacji przy różnych wartościach parametrów **eps** i 
**min_samples**. Zwróć uwagę, jak bardzo wyniki mogą się różnić nawet przy niewielkich zmianach parametrów.

*Wykres pokazuje wyłącznie moment znalezienia nowego klastra.*
""")

st.divider()

with st.expander("Wczytaj inne dane"):
    uploaded_file = st.file_uploader(
        "Wybierz plik (CSV, JSON lub XML)", 
        type=['csv', 'json', 'xml']
    )
    
    if uploaded_file is not None:
        if 'uploaded_filename' not in st.session_state or \
           st.session_state['uploaded_filename'] != uploaded_file.name:
            
            load_file_to_dataframe(uploaded_file)
            st.session_state['uploaded_filename'] = uploaded_file.name
            st.rerun()

with st.expander("Podgląd danych"):
    st.dataframe(st.session_state.df.head())

# Expander do wyboru zmiennych
with st.expander("Wybór zmiennych do analizy"):
    st.markdown("**Wybierz zmienne, które chcesz użyć do klasteryzacji:**")
    st.info("Jeśli wybierzesz więcej niż 2 zmienne, automatycznie zostanie zastosowane PCA do redukcji wymiarów.")
    
    available_columns = data.columns.tolist()
    selected_columns = st.multiselect(
        "Dostępne zmienne numeryczne:",
        options=available_columns,
        default=available_columns[:min(2, len(available_columns))],
        help="Wybierz co najmniej 2 zmienne"
    )
    
    if len(selected_columns) < 2:
        st.warning("Wybierz co najmniej 2 zmienne do analizy!")
        st.stop()

data_selected = data[selected_columns]
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_selected)

if len(selected_columns) > 2:
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(data_scaled)
    explained_variance = pca.explained_variance_ratio_
    use_pca = True
    st.info(f"Zastosowano PCA: z {len(selected_columns)} zmiennych do 2 głównych składowych")
elif len(selected_columns) == 2:
    data_2d = data_scaled
    explained_variance = [1.0, 1.0]
    use_pca = False
else:
    st.error("Wybierz co najmniej 2 zmienne!")
    st.stop()

col1, col2, col3 = st.columns(3)
with col1:
    epsilon = st.slider("Epsilon (ε) - promień sąsiedztwa", min_value=0.1, max_value=2.0, value=0.5, step=0.05)
with col2:
    min_pts = st.slider("MinPts - minimalna liczba punktów", min_value=2, max_value=20, value=5, step=1)
with col3:
    max_plots = st.slider("Maksymalna liczba wykresów do pokazania", min_value=3, max_value=15, value=7, step=1)

if st.button("Uruchom DBSCAN", type="primary"):
    dbscan = DBSCANIterative(eps=epsilon, min_samples=min_pts)
    dbscan.fit(data_2d)
    
    # Statystyki
    final_labels = dbscan.labels
    n_clusters = len(set(final_labels)) - (1 if -1 in final_labels else 0)
    n_noise = np.sum(final_labels == -1)
    n_core = np.sum(dbscan.core_points)
    
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.metric("Liczba klastrów", n_clusters)
    with d2:
        st.metric("Punkty szumu", n_noise)
    with d3:
        st.metric("Punkty rdzeniowe", n_core)
    with d4:
        st.metric("Punkty brzegowe", len(data_2d) - n_core - n_noise)
    
    st.markdown("---")
    st.subheader(f"Historia DBSCAN ({len(dbscan.history)} kroków)")
    
    total_steps = len(dbscan.history)
    
    core_point_steps = [0]  # Stan początkowy
    for i, state in enumerate(dbscan.history):
        if state['current_point'] is not None:
            core_point_steps.append(i)
    core_point_steps.append(total_steps - 1)  # Koniec
    
    if len(core_point_steps) > max_plots:
        step = len(core_point_steps) // (max_plots - 1)
        frames_to_show = [core_point_steps[i] for i in range(0, len(core_point_steps), step)]
        if frames_to_show[-1] != core_point_steps[-1]:
            frames_to_show.append(core_point_steps[-1])
    else:
        frames_to_show = core_point_steps
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', 
              '#F38181', '#AA96DA', '#FCBAD3', '#A8E6CF', '#FFD3B6']
    
    #-------------------------Wykresy------------------------
    for frame_idx in frames_to_show:
        state = dbscan.history[frame_idx]
        labels = state['labels']
        core_points = state['core_points']
        iteration = state['iteration']
        description = state['description']
        current_point = state['current_point']
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        unique_labels = set(labels)
        
        for label in unique_labels:
            if label == -2:
                mask = labels == label
                ax.scatter(data_2d[mask, 0], data_2d[mask, 1], 
                          c='lightgray', label='Nieprzetworzone', 
                          alpha=0.5, s=50, edgecolors='gray', linewidths=0.5)
            elif label == -1:
                mask = labels == label
                ax.scatter(data_2d[mask, 0], data_2d[mask, 1], 
                          c='black', marker='x', label='Szum', 
                          alpha=0.7, s=80, linewidths=2)
            else: 
                mask = labels == label
                ax.scatter(data_2d[mask, 0], data_2d[mask, 1], 
                          c=colors[label % len(colors)], 
                          label=f'Klaster {label}', 
                          alpha=0.6, s=70, edgecolors='black', linewidths=0.5)
        
        core_mask = core_points & (labels >= 0)
        if np.any(core_mask):
            ax.scatter(data_2d[core_mask, 0], data_2d[core_mask, 1], 
                      c='gold', marker='o', s=10, 
                      edgecolors='black', linewidths=1,
                      label='Punkty rdzeniowe', zorder=10)

        if current_point is not None:
            ax.scatter(data_2d[current_point, 0], data_2d[current_point, 1], 
                      c='red', marker='o', s=50, 
                      edgecolors='darkred', linewidths=3,
                      label='Aktualny punkt', zorder=15, alpha=0.8)

            circle = plt.Circle((data_2d[current_point, 0], data_2d[current_point, 1]), 
                               epsilon, fill=False, color='red', 
                               linestyle='--', linewidth=2, alpha=0.6)
            ax.add_patch(circle)
        
        # Etykiety osi zależnie od tego czy użyto PCA
        if use_pca:
            ax.set_xlabel(f'PC1 ({explained_variance[0]*100:.1f}% wariancji)')
            ax.set_ylabel(f'PC2 ({explained_variance[1]*100:.1f}% wariancji)')
        else:
            ax.set_xlabel(selected_columns[0])
            ax.set_ylabel(selected_columns[1])
        
        variables_text = f"Zmienne: {', '.join(selected_columns)}"
        if use_pca:
            variables_text += f" (PCA)"
        
        ax.set_title(f'DBSCAN - Krok {iteration+1}\n{description}\n{variables_text}', 
                    fontsize=14, fontweight='bold', pad=15)
        
        # Legenda
        handles, labels_legend = ax.get_legend_handles_labels()
        by_label = dict(zip(labels_legend, handles))
        ax.legend(by_label.values(), by_label.keys(), 
                 loc='upper right', fontsize=9, framealpha=0.9)
        
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    if use_pca:
        st.markdown(f"""
        **Procent wariancji** oznacza, że ten wymiar zachowuje taką ilość informacji o zróżnicowaniu danych. 
        PC1 + PC2 razem zachowują {(explained_variance[0] + explained_variance[1])*100:.1f}% całkowitej wariancji z oryginalnych {len(selected_columns)} zmiennych.
        Im bliżej 100%, tym lepiej zachowana struktura danych po redukcji wymiarów.
        """)