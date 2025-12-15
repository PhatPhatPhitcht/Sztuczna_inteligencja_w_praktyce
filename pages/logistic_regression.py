import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import streamlit as st

st.set_page_config(page_title="Interaktywne algorytmy uczenia maszynowego", initial_sidebar_state="collapsed")
st.markdown("""
    <style>
        [data-testid="stSidebarNav"] {display: none;}
    </style>
""", unsafe_allow_html=True)

if 'df' not in st.session_state:
    st.error("Brak danych! Wróć do strony głównej")
    st.stop()

df = st.session_state.df
#df = df_wine = sns.load_dataset('penguins').dropna()

class LogisticRegressionIterative:
    def __init__(self, C=1.0, penalty='l2', max_iter=1000):
        self.C = C
        self.penalty = penalty
        self.max_iter = max_iter
        self.history = []
        
    def fit(self, X, y):
        n_samples = X.shape[0]
        
        model = LogisticRegression(
            C=self.C, 
            penalty=self.penalty, 
            max_iter=5,
            warm_start=True,
            solver='saga' if self.penalty == 'l1' else 'lbfgs',
            random_state=42,
            tol=1e-4
        )
        
        self.history.append({
            'iteration': 0,
            'coef': None,
            'intercept': None,
            'description': 'Stan początkowy - model niewyuczony'
        })
        
        # Trenowanie
        for i in range(1, self.max_iter + 1):
            model.fit(X, y)
            
            self.history.append({
                'iteration': i,
                'coef': model.coef_.copy(),
                'intercept': model.intercept_.copy(),
                'score': model.score(X, y),
                'description': f'Iteracja {i} - dokładność: {model.score(X, y):.3f}'
            })
            
            # Zakończenie działania
            if hasattr(model, 'n_iter_') and model.n_iter_[0] < i:
                break
        
        self.model = model
        self.coef_ = model.coef_
        self.intercept_ = model.intercept_
        
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def score(self, X, y):
        return self.model.score(X, y)
#-----------------------------------------Rysowanie wykresów-------------------------------------------------
def plot_decision_boundary(X_train, y_train, X_test, y_test, state, ax, title, model=None):
    
    X_all = np.vstack([X_train, X_test])

    x_min, x_max = X_all[:, 0].min() - 1, X_all[:, 0].max() + 1
    y_min, y_max = X_all[:, 1].min() - 1, X_all[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    # Szukanie klas
    unique_classes = np.unique(np.concatenate([y_train, y_test]))
    n_classes = len(unique_classes)
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', 
              '#F38181', '#AA96DA', '#FCBAD3', '#A8E6CF', '#FFD3B6']
    
    # Jeśli model istnieje
    if state['coef'] is not None and model is not None:
        temp_model = LogisticRegression()
        temp_model.coef_ = state['coef']
        temp_model.intercept_ = state['intercept']
        temp_model.classes_ = model.classes_
        
        # Predykcje dla siatki
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        Z = temp_model.predict(grid_points)
        Z = Z.reshape(xx.shape)
        # Dla 2 klas
        if n_classes == 2:
            Z_score = state['coef'][0][0] * xx + state['coef'][0][1] * yy + state['intercept'][0]
            
            ax.contourf(xx, yy, Z_score, levels=[-100, 0, 100], 
                       colors=[colors[0], colors[1]], alpha=0.2)

            ax.contour(xx, yy, Z_score, levels=[0], colors='black', 
                      linewidths=2, linestyles='--', label='Granica decyzyjna')
        else:
            # Dla 3+ klas
            class_to_color_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
            Z_colors = np.array([class_to_color_idx[cls] for cls in Z.ravel()]).reshape(Z.shape)
            
            from matplotlib.colors import ListedColormap
            n_colors_needed = len(unique_classes)
            cmap = ListedColormap([colors[i % len(colors)] for i in range(n_colors_needed)])
            
            ax.contourf(xx, yy, Z_colors, levels=np.arange(n_colors_needed + 1) - 0.5,
                       cmap=cmap, alpha=0.3)
            
            ax.contour(xx, yy, Z_colors, levels=np.arange(n_colors_needed + 1) - 0.5,
                      colors='black', linewidths=1.5, linestyles='--', alpha=0.6)
    
    # punkty TRENINGOWE
    for idx, cls in enumerate(unique_classes):
        mask = y_train == cls
        if np.any(mask):
            ax.scatter(X_train[mask, 0], X_train[mask, 1], 
                      c=colors[idx % len(colors)], 
                      alpha=0.8, s=80, edgecolors='black', linewidths=1.0, zorder=5)
    
    # punkty TESTOWE
    for idx, cls in enumerate(unique_classes):
        mask = y_test == cls
        if np.any(mask):
            ax.scatter(X_test[mask, 0], X_test[mask, 1], 
                      c=colors[idx % len(colors)], 
                      label=f'Klasa {cls}',
                      alpha=0.8, s=80, edgecolors='black', linewidths=1.0, zorder=4)
            ax.scatter(X_test[mask, 0], X_test[mask, 1],
                      facecolors='none', edgecolors='black', 
                      s=10, linewidths=1.0, zorder=6)
    ax.scatter([], [], facecolors='none', edgecolors='black', 
              s=10, linewidths=1.0, zorder=6, label='Punktu testowy')
    
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9, ncol=1)
    ax.grid(True, alpha=0.3, linestyle='--')

#--------------------------------Streamlit------------------------------------

st.page_link("main.py", label="⬅️ Powrót do strony głównej")

st.subheader("Algorytm Regresji Logistycznej")
st.markdown("""
Regresja logistyczna to metoda klasyfikacji, która uczy się, jakie relacje zachodzą między 
zestawem cech opisujących obiekty a przynależnością tych obiektów do jednej z klas. Model 
buduje wewnętrzną funkcję liniową, która na podstawie wartości cech przypisuje każdej 
obserwacji pewną wagę. Taki surowy wynik przechodzi przez funkcję sigmoid, zmieniającą jego 
zakres od [0, 1]. Wynikiem tego procesu jest reprezentacja prawdopodobieństwa należenia do 
danej klasy. Proces uczenia polega na dopasowaniu parametrów modelu tak, by przewidywane 
prawdopodobieństwa dobrze odpowiadały rzeczywistym etykietom w zbiorze treningowym. 
W odróżnieniu od prostych reguł typu „jeśli cecha większa niż wartość, to klasa A”, regresja 
logistyczna łączy sygnały z wielu cech jednocześnie i zwraca gładką ocenę prawdopodobieństwa, 
co czyni ją elastyczną, łatwą do interpretacji i szeroko stosowaną przy problemach 
klasyfikacji.
            
**Parametry:**
            
Parametr oznaczany jako **C** w implementacjach regresji logistycznej pełni rolę kontrolera siły 
regularyzacji modelu. Mniejsza wartość **C** oznacza silniejszą kontrolę nad wielkością wag, a 
większa wartość **C** pozwala wagom swobodniej przyjmować duże wartości. W praktyce regularyzacja 
jest mechanizmem zapobiegającym nadmiernemu dopasowaniu modelu do szumu w danych — zmniejsza skłonność 
algorytmu do przeuczenia, czyli dopasowywania bardzo specyficznych wzorców, które nie generalizują 
poza zbiór treningowy. Kiedy **C** jest bardzo małe, model staje się „prostszym” opisem relacji: jego 
współczynniki są mocniej tłumione, co zwykle redukuje wariancję kosztem pewnego wzrostu błędu na 
zbiorze treningowym i prowadzi do niedouczenia.

**Parametr penalty** (Regularyzacja) jest sposobem kontrolowania złożoności modelu poprzez dodanie 
kary za zbyt duże wartości wag. Dzięki temu model nie dopasowuje się zbyt mocno do danych 
treningowych i lepiej generalizuje na nowe dane.
    - Regularyzacja typu **l2** sprawia, że wszystkie wagi są „przyciągane” w kierunku mniejszych 
        wartości, ale w sposób równomierny. Nie eliminuje ona całkowicie żadnej cechy: zamiast tego 
        stara się rozłożyć wagę pomiędzy wiele cech, sprawiając, że każda z nich ma niewielki wpływ.
    - Regularyzacja typu **l1** działa bardziej „agresywnie” — nie tylko ogranicza wielkość wag, 
        ale też sprzyja temu, aby część z nich spadła dokładnie do zera. W praktyce oznacza to, że model 
        samodzielnie wybiera, które cechy są istotne, a które mogą zostać całkowicie pominięte.
            
Istotne w praktycznym stosowaniu tych parametrów jest rozumienie ich wzajemnych powiązań oraz wpływu na jakość i stabilność modelu. Regularyzacja i jej siła (penalty oraz C) decydują o złożoności modelu i jego odporności na szum, a max_iter zapewnia, że proces optymalizacji ma szansę dojść do końca.            

[Dowiedz się więcej](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)

Poniżej możesz obserwować, jak granica decyzyjna zmienia się w trakcie treningu modelu.
""")

st.divider()

with st.expander("Podgląd danych"):
    st.dataframe(st.session_state.df.head())

# Wybór kolumny docelowej
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
all_cols = df.columns.tolist()

st.markdown("### Konfiguracja danych")
target_col = st.selectbox("Wybierz kolumnę docelową (etykiety)", all_cols)

if target_col:
    unique_classes = df[target_col].nunique()
    
    if unique_classes < 2:
        st.error("Kolumna docelowa musi mieć co najmniej 2 klasy!")
        st.stop()
    elif unique_classes > 10:
        st.warning(f"Kolumna ma {unique_classes} klas. Dla lepszej wizualizacji zaleca się mniej klas.")
    
    # Przygotowanie danych
    feature_cols = [col for col in numeric_cols if col != target_col]
    
    if len(feature_cols) < 2:
        st.error("Potrzebne są co najmniej 2 kolumny numeryczne jako cechy!")
        st.stop()
    
    X = df[feature_cols].dropna()
    y = df.loc[X.index, target_col]
  
    valid_mask = ~y.isna()
    X = X[valid_mask]
    y = y[valid_mask]
    
    # Informacje o danych
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Liczba próbek", len(X))
    with col2:
        st.metric("Liczba cech", len(feature_cols))
    with col3:
        st.metric("Liczba klas", y.nunique())
    
    st.divider()
    st.markdown("### Parametry modelu")
    
    # Parametry modelu
    col1, col2, col3= st.columns(3)
    with col1:
        C_value = st.slider("C (odwrotność regularyzacji)", 
                           min_value=0.01, max_value=10.0, value=1.0, step=0.1)
    with col2:
        penalty_type = st.selectbox("Penalty (regularyzacja)", ['l2', 'l1'])
    with col3:
        #max_iter = st.slider("Maksymalna liczba iteracji", 
                           #min_value=10, max_value=200, value=100, step=10)
        n_plots = st.slider("Liczba wykresów", 
                          min_value=3, max_value=15, value=7, step=1)                   
    #with col4:
    
    if st.button("Trenuj model", type="primary"):
        with st.spinner("Przetwarzanie danych..."):
            # Standaryzacja i PCA
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_pca, y, test_size=20, random_state=42, stratify=y
            )

            st.session_state['pca_variance'] = pca.explained_variance_ratio_.sum()
        
        with st.spinner("Trenowanie modelu..."):
            # Trenowanie modelu
            lr = LogisticRegressionIterative(C=C_value, penalty=penalty_type, max_iter=1000)
            lr.fit(X_train, y_train)
            
            st.session_state['trained_model'] = lr.model
            st.session_state['scaler'] = scaler
            st.session_state['pca'] = pca
            st.session_state['X_train'] = X_train
            st.session_state['X_test'] = X_test
            st.session_state['y_train'] = y_train
            st.session_state['y_test'] = y_test
            st.session_state['feature_cols'] = feature_cols
            st.session_state['target_col'] = target_col
            st.session_state['lr_history'] = lr.history
            st.session_state['train_score'] = lr.score(X_train, y_train)
            st.session_state['test_score'] = lr.score(X_test, y_test)
            st.session_state['C_value'] = C_value
            st.session_state['n_plots'] = n_plots
            st.session_state['test_size'] = 20
        
        st.rerun()
    
    if 'trained_model' in st.session_state and st.session_state.get('target_col') == target_col:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Dokładność (accuracy)", f"{st.session_state['test_score']:.3f}")
        with col2:
            st.metric("Liczba iteracji", len(st.session_state['lr_history']) - 1)
        with col3:
            st.metric("Współczynnik C", st.session_state['C_value'])
        
        st.subheader(f"Ewolucja granicy decyzyjnej ({len(st.session_state['lr_history'])-1} kroków)")
        
        # Wybieranie kroków
        lr_history = st.session_state['lr_history']
        X_train = st.session_state['X_train']
        X_test = st.session_state['X_test']
        y_train = st.session_state['y_train']
        y_test = st.session_state['y_test']
        n_plots = st.session_state['n_plots']
        
        total_steps = len(lr_history)
        if total_steps <= n_plots:
            frames_to_show = list(range(total_steps))
        else:
            step = total_steps // (n_plots - 1)
            frames_to_show = [i * step for i in range(n_plots - 1)]
            frames_to_show.append(total_steps - 1) 
        
        #---------------------------------------Wykresy-----------------------------------------
        for frame_idx in frames_to_show:
            state = lr_history[frame_idx]
            
            fig, ax = plt.subplots(figsize=(10, 7))
            
            title = f"Krok {state['iteration']}\n{state['description']}"
            plot_decision_boundary(X_train, y_train, X_test, y_test, state, ax, title, 
                                 st.session_state['trained_model'])
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        
        st.markdown("---")
        st.markdown("### Testowanie modelu")
        st.markdown("""
        Model został zapisany w pamięci sesji. Możesz teraz przetestować go na nowych danych 
        wprowadzając wartości poniżej.
        """)
        
        model = st.session_state['trained_model']
        scaler = st.session_state['scaler']
        pca = st.session_state['pca']
        feature_cols = st.session_state['feature_cols']
        X = df[feature_cols].dropna()
        
        st.markdown("#### Wprowadź wartości cech:")
        
        test_values = {}
        cols_per_row = 3
        
        for i in range(0, len(feature_cols), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col_name in enumerate(feature_cols[i:i+cols_per_row]):
                with cols[j]:
                    min_val = float(X[col_name].min())
                    max_val = float(X[col_name].max())
                    mean_val = float(X[col_name].mean())
                    test_values[col_name] = st.number_input(
                        col_name, 
                        min_value=min_val,
                        max_value=max_val,
                        value=mean_val,
                        step=(max_val - min_val) / 100,
                        key=f"test_{col_name}"
                    )
        
        if st.button("Klasyfikuj próbkę", type="secondary"):
            test_df = pd.DataFrame([test_values])
            test_scaled = scaler.transform(test_df)
            test_pca = pca.transform(test_scaled)
            
            prediction = model.predict(test_pca)[0]
            probabilities = model.predict_proba(test_pca)[0]
            
            st.markdown("""
            **Wynik predykcji:**
            """)
            st.success(f"{prediction}")
            
            st.markdown("##### Prawdopodobieństwa dla każdej klasy:")
            prob_df = pd.DataFrame({
                'Klasa': model.classes_,
                'Prawdopodobieństwo': probabilities
            }).sort_values('Prawdopodobieństwo', ascending=False)
            
            st.dataframe(prob_df, hide_index=True, use_container_width=True)
            
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.barh(prob_df['Klasa'].astype(str), prob_df['Prawdopodobieństwo'], 
                   color='#4ECDC4', edgecolor='black')
            ax.set_xlabel('Prawdopodobieństwo', fontsize=10)
            ax.set_ylabel('Klasa', fontsize=10)
            ax.set_title('Rozkład prawdopodobieństw predykcji', fontsize=12, fontweight='bold')
            ax.set_xlim(0, 1)
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)