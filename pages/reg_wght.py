import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from scipy import stats
from utils.file_loader import load_file_to_dataframe
from sklearn.datasets import fetch_openml

st.set_page_config(page_title="Interaktywne algorytmy uczenia maszynowego", initial_sidebar_state="collapsed")
st.markdown("""
    <style>
        [data-testid="stSidebarNav"] {display: none;}
    </style>
""", unsafe_allow_html=True)
st.page_link("main.py", label="⬅️ Powrót do strony głównej")

st.subheader("Algorytm Regresji Ważonej")

if 'df' not in st.session_state:
    st.error("Brak danych w session_state.df")
    st.stop()

df = st.session_state.df.copy()

st.markdown("""
**Regresja ważona** to rozszerzenie regresji liniowej, które pozwala nadawać różne znaczenie poszczególnym 
obserwacjom w danych. Niektóre punkty mogą być bardziej wiarygodne (np. precyzyjniejsze pomiary) lub mniej wiarygodne 
(np. zawierające większy szum), a regresja ważona to uwzględnia. Wagi mogą być nadane manualnie, lub metodą IRLS. 
W tym przypadku skupimy się na tym drugim.

Model stara się lepiej dopasować linię regresji do bardziej wiarygodnych punktów. Punkty z małą wagą mają mniejszy 
wpływ na wynik i model może je w pewnym stopniu ignorować.

**IRLS (Iteratively Reweighted Least Squares)** to algorytm, który automatycznie oblicza wagi, gdy nie znamy ich z góry. Działa w iteracjach:
- W każdej iteracji model analizuje błędy przewidywań (reszty)
- Na podstawie tych błędów szacuje, które obszary danych są bardziej/mniej stabilne
- Przydziela większe wagi punktom w stabilnych obszarach, mniejsze w niestabilnych
- Powtarza proces, aż wagi przestają się zmieniać (zbieżność)

**Siła ważenia** kontroluje jak mocno algorytm stosuje wagi:
- 0% = ignoruje wagi całkowicie (zwykła regresja liniowa)
- 100% = pełne ważenie (maksymalny efekt IRLS)
- Wartości pośrednie (np. 50%) to kompromis - łagodniejsze ważenie, które może działać lepiej gdy dane są trudne

**Liczba iteracji IRLS** określa ile razy algorytm przeliczy wagi:
- 1 iteracja = szybkie, ale może nie wystarczyć do pełnej korekcji
- 3-5 iteracji = typowo wystarczające (algorytm zbiega się)
- 10+ iteracji = rzadko potrzebne, chyba że dane są bardzo trudne

Więcej iteracji nie zawsze oznacza lepszy model - jeśli algorytm zbiegł się wcześniej, dodatkowe iteracje nic nie zmienią.

**Metryki jakości modelu:**

- **R² (Coefficient of Determination)** - Współczynnik determinacji, Pokazuje, jaki procent zmienności zmiennej zależnej jest wyjaśniony przez model.

- **MSE (Mean Squared Error)** - Średni błąd kwadratowy. Średnia kwadratów różnic między wartościami rzeczywistymi a przewidywanymi.

[Regresja liniowa](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
            
[Modele liniowe](https://scikit-learn.org/stable/modules/linear_model.html)                       
""")

st.divider()

with st.expander("Wczytaj inne dane"):
    st.markdown("""
    Do regresji zalecam użycie zbioru *House Sales*
        """)
    if st.button("Wczytaj *House sales*", type="secondary"):
        st.session_state.df = pd.read_csv("house_data.csv")
        df = pd.read_csv("house_data.csv")

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

# Inicjalizacja session_state z prefiksem dla regresji ważonej
for key, default in {
    "wr_model": None,
    "wr_features": None,
    "wr_target": None,
    "wr_is_trained": False,
    "wr_label_encoders": {},
    "wr_use_pca": False,
    "wr_pca": None,
    "wr_scaler": None
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if len(numeric_cols) < 1:
    st.error("Wymagana jest co najmniej 1 kolumna numeryczna jako zmienna docelowa.")
    st.stop()

# Expander do wyboru zmiennych
with st.expander("Wybór zmiennych", expanded=False):
    st.markdown("""
    **Wybierz zmienne , które będą użyte do predykcji.**
    
    - Jeśli wybierzesz **więcej niż 2 zmienne niezależne**, zastosowana zostanie **PCA (Principal Component Analysis)** do redukcji wymiarowości dla wizualizacji.
    - Zmienne tekstowe (kategoryczne) nie są obsługiwane.
    """)
    
    y_col = st.selectbox(
    "Wybierz zmienną docelową (Y) - wartość do przewidzenia",
    numeric_cols,
    index=numeric_cols.index(st.session_state.wr_target)
    if st.session_state.wr_target in numeric_cols else 0
    )

    available_features = [c for c in df.columns if c != y_col]
    
    selected_features = st.multiselect(
        "Wybierz zmienne niezależne (X)",
        available_features,
        default=available_features if len(available_features) <= 5 else available_features[:3]
    )
    
    if len(selected_features) == 0:
        st.warning("Proszę wybrać co najmniej jedną zmienną niezależną.")

# Macierz korelacji
if len(selected_features) > 0:
    with st.expander("Macierz korelacji"):
        try:
            corr_df = df[selected_features + [y_col]].copy()
            
            temp_encoders = {}
            for col in corr_df.columns:
                if corr_df[col].dtype == 'object':
                    le = LabelEncoder()
                    corr_df[col] = le.fit_transform(corr_df[col].astype(str))
                    temp_encoders[col] = le
            
            correlation_matrix = corr_df.corr()
            
            fig_corr = go.Figure(data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=correlation_matrix.values,
                texttemplate='%{text:.2f}',
                textfont={"size": 10},
                colorbar=dict(title="Korelacja")
            ))
            
            fig_corr.update_layout(
                title='Macierz korelacji między zmiennymi',
                xaxis_title="Zmienne",
                yaxis_title="Zmienne",
                height=500,
                width=700
            )
            
            st.plotly_chart(fig_corr, use_container_width=True)
            
            st.markdown("""
            **Interpretacja:**
            - Wartości bliskie **1** (czerwone) - silna korelacja dodatnia
            - Wartości bliskie **-1** (niebieskie) - silna korelacja ujemna
            - Wartości bliskie **0** (białe) - brak korelacji liniowej
            """)
        except Exception as e:
            st.error(f"Nie udało się wygenerować macierzy korelacji: {str(e)}")

if len(selected_features) == 0:
    st.stop()

# Przygotowanie danych
try:
    X = df[selected_features].copy()
    y = df[y_col].copy()
    
    if not pd.api.types.is_numeric_dtype(y):
        st.error(f"Zmienna docelowa '{y_col}' musi być numeryczna dla regresji!")
        st.stop()
    
    if y.isna().any():
        st.warning(f"Usunięto {y.isna().sum()} wierszy z brakującymi wartościami w zmiennej docelowej.")
        valid_indices = ~y.isna()
        X = X[valid_indices]
        y = y[valid_indices]
    
    label_encoders = {}
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
            st.info(f"Zakodowano zmienną kategoryczną: {col}")
    
    if X.isna().any().any():
        st.warning(f"Uzupełniono brakujące wartości w zmiennych niezależnych medianą.")
        X = X.fillna(X.median())
    
    X_array = X.values
    y_array = y.values
    
    if len(X_array) < 10:
        st.error("Za mało danych po usunięciu wartości NaN (minimum 10 próbek wymagane)")
        st.stop()
    
    # Skalowanie i PCA
    use_pca = len(selected_features) > 1
    
    if use_pca:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_array)
        
        pca = PCA(n_components=1)
        X_for_model = pca.fit_transform(X_scaled)
        
        st.info(f"Zastosowano PCA: {len(selected_features)} zmiennych → 1 komponent główny (wyjaśnia {pca.explained_variance_ratio_[0]:.1%} wariancji)")
    else:
        X_for_model = X_array
        X_scaled = X_array
        scaler = None
        pca = None
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Liczba próbek", len(X_for_model))
    with col2:
        st.metric("Liczba oryginalnych cech", len(selected_features))
    
    st.divider()
    
except Exception as e:
    st.error(f"Błąd podczas przygotowania danych: {str(e)}")
    st.stop()

#--------------------------Test heteroskedastyczności------------------------------
st.markdown("### Test Heteroskedastyczności")
st.markdown("""
**Heteroskedastyczność** oznacza, że błędy przewidywań modelu (reszty) mają różną wariancję w różnych obszarach danych. 
Model popełnia większe błędy w niektórych regionach, a mniejsze w innych.

**Test Breuscha-Pagana** służy do statystycznej weryfikacji występowania heteroskedastyczności. Polega on na sprawdzeniu, 
czy wariancja składnika losowego (reszt z modelu regresji liniowej) zależy od wartości zmiennych objaśniających (X).

**Statystyka BP** to liczba, która mierzy siłę zależności między wartościami X a wielkością błędów. 
Im wyższa wartość, tym silniejsza heteroskedastyczność.

**Wartość p** to prawdopodobieństwo, że zaobserwowane dane mogły powstać przypadkowo, gdyby nie było heteroskedastyczności:
- p < 0.05: Heteroskedastyczność jest obecna (regresja ważona ma sens!)
- p ≥ 0.05: Dane są homoskedastyczne (zwykła regresja wystarczy)

[Dowiedz się więcej](https://homepage.univie.ac.at/robert.kunst/emwipres121.pdf)
""")

model_initial = LinearRegression()
model_initial.fit(X_for_model, y_array)
y_pred_initial = model_initial.predict(X_for_model)
residuals_initial = y_array - y_pred_initial

# Test Breusch-Pagan
def breusch_pagan_test(X, residuals):
    """Uproszczony test Breusch-Pagan"""
    residuals_squared = residuals ** 2
    model_bp = LinearRegression()
    model_bp.fit(X, residuals_squared)
    r2 = model_bp.score(X, residuals_squared)
    n = len(residuals)
    bp_stat = n * r2
    df_bp = X.shape[1]
    p_value = 1 - stats.chi2.cdf(bp_stat, df_bp)
    return bp_stat, p_value

bp_stat, bp_pvalue = breusch_pagan_test(X_for_model, residuals_initial)
mse_initial = np.mean(residuals_initial ** 2)
r2_initial = model_initial.score(X_for_model, y_array)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Statystyka BP", f"{bp_stat:.4f}")
with col2:
    st.metric("Wartość p", f"{bp_pvalue:.4f}")
with col3:
    st.metric("MSE", f"{mse_initial:.4f}")
with col4:
    st.metric("R²", f"{r2_initial:.4f}")

if bp_pvalue >= 0.05:
    st.warning("Dane są homoskedastyczne (p ≥ 0.05). Regresja ważona może nie być potrzebna.")
    heteroskedastic = False
else:
    st.success("Wykryto heteroskedastyczność (p < 0.05). Regresja ważona jest zalecana.")
    heteroskedastic = True

st.divider()

#---------------------------------Wykres reszt---------------------------------
st.markdown("### Wykres Reszt")
st.markdown("""
**Wykres reszt** (residual plot) pokazuje błędy modelu w funkcji wartości przewidywanych.
""")

fig_residuals = go.Figure()

fig_residuals.add_trace(go.Scatter(
    x=y_pred_initial,
    y=residuals_initial,
    mode='markers',
    marker=dict(size=6, opacity=0.6, color='steelblue'),
    name='Reszty',
    hovertemplate='<b>Przewidywane:</b> %{x:.2f}<br><b>Reszta:</b> %{y:.2f}<extra></extra>'
))

fig_residuals.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Brak błędu")

fig_residuals.update_layout(
    title="Reszty vs Wartości Przewidywane (Zwykła Regresja)",
    xaxis_title="Wartości przewidywane",
    yaxis_title="Reszty",
    height=400
)
st.plotly_chart(fig_residuals, use_container_width=True)

st.divider()

#------------------------------Zwykła regresja liniowa---------------------------
st.markdown("### Zwykła Regresja Liniowa")

col1, col2 = st.columns(2)
with col1:
    st.metric("R²", f"{r2_initial:.4f}")
with col2:
    st.metric("MSE", f"{mse_initial:.4f}")

fig_linear = go.Figure()

sort_idx = np.argsort(X_for_model.flatten())
X_sorted = X_for_model[sort_idx]
y_sorted = y_array[sort_idx]
y_pred_sorted = y_pred_initial[sort_idx]

fig_linear.add_trace(go.Scatter(
    x=X_sorted.flatten(),
    y=y_sorted,
    mode='markers',
    name='Dane',
    marker=dict(size=8, opacity=0.6, color='steelblue')
))

fig_linear.add_trace(go.Scatter(
    x=X_sorted.flatten(),
    y=y_pred_sorted,
    mode='lines',
    name='Regresja liniowa',
    line=dict(color='green', width=2)
))

x_label = f"PC1 ({pca.explained_variance_ratio_[0]:.1%} wariancji)" if use_pca else selected_features[0]
fig_linear.update_layout(
    title="Dopasowanie Zwykłej Regresji Liniowej",
    xaxis_title=x_label,
    yaxis_title=y_col,
    height=400
)
st.plotly_chart(fig_linear, use_container_width=True)

st.divider()

#---------------------------------PARAMETRY IRLS-----------------------------
st.markdown("### Parametry Regresji Ważonej (IRLS)")

col1, col2 = st.columns(2)

with col1:
    weight_multiplier = st.slider(
        "Siła ważenia (%)",
        min_value=0,
        max_value=100,
        value=100,
        step=5,
        help="0% = zwykła regresja, 100% = pełne ważenie"
    ) / 100.0

with col2:
    n_iterations = st.slider(
        "Liczba iteracji IRLS",
        min_value=1,
        max_value=10,
        value=3,
        help="Liczba iteracji do obliczania wag"
    )

with st.expander("⚙️ Zaawansowane opcje"):
    variance_model_type = st.selectbox(
        "Model wariancji",
        [
            "Liniowy: σ = a + b·ŷ",
            "Kwadratowy: σ = a + b·ŷ + c·ŷ²",
            "Potęgowy: σ = a·ŷ^b",
            "Logarytmiczny: σ = a + b·log(ŷ)"
        ],
        help="Jak modelować zależność wariancji od wartości przewidywanych"
    )
    
    st.info("""
    **Kiedy którego użyć?**
    - **Liniowy**: Wariancja rośnie proporcjonalnie (domyślny, najczęstszy)
    - **Kwadratowy**: Wariancja rośnie szybciej dla dużych wartości
    - **Potęgowy**: Wariancja rośnie wykładniczo (silna heteroskedastyczność)
    - **Logarytmiczny**: Wariancja rośnie wolno, stabilizuje się
    """)

train_button = st.button("Trenuj model regresji ważonej", type="primary")

#------------------------------Główny algorytm------------------------------
if train_button:
    try:
        iterations_results = []
        weights = np.ones(len(y_array))
        current_model = model_initial
        current_residuals = residuals_initial
        
        weight_changes = []
        mse_history = []
        r2_history = []
        variance_stability = []
        
        with st.spinner('Trenuję model regresji ważonej...'):
            for iteration in range(n_iterations):
                old_weights = weights.copy() if iteration > 0 else None
                
                abs_residuals = np.abs(current_residuals)
                # POPRAWKA: Używamy wygładzonej wersji reszt do obliczania wag
                # aby uniknąć nadmiernego reagowania na pojedyncze outliers
                abs_residuals_smoothed = np.maximum(abs_residuals, np.percentile(abs_residuals, 10))
                abs_residuals = np.maximum(abs_residuals_smoothed, 0.01)
                
                # POPRAWKA: Używamy oryginalnych skalowanych danych do modelowania wariancji
                # zamiast PCA, żeby uniknąć nadmiernego uproszczenia
                # ALE dla wielu cech, używamy wartości przewidywanych zamiast wszystkich X
                if use_pca and len(selected_features) > 2:
                    # Dla wielu cech, używamy wartości przewidywanych do modelowania wariancji
                    # To stabilniejsze niż próba modelowania na wielu wymiarach
                    y_pred_for_variance = current_model.predict(X_for_model)
                    X_for_variance = y_pred_for_variance.reshape(-1, 1)
                elif use_pca:
                    # Dla 2 cech, możemy użyć oryginalnych skalowanych danych
                    X_for_variance = X_scaled
                else:
                    # Bez PCA, używamy oryginalnych danych
                    X_for_variance = X_for_model
                
                # Modelowanie wariancji - używamy prostszych modeli dla stabilności
                if "Liniowy" in variance_model_type:
                    # Dla modelu liniowego - używamy tylko 1 wymiar
                    if X_for_variance.shape[1] > 1:
                        X_var = np.mean(X_for_variance, axis=1).reshape(-1, 1)
                    else:
                        X_var = X_for_variance
                    variance_model = LinearRegression()
                    variance_model.fit(X_var, abs_residuals)
                    predicted_std = variance_model.predict(X_var).flatten()
                    
                elif "Kwadratowy" in variance_model_type:
                    # Dla modelu kwadratowego - używamy średnią z cech
                    if X_for_variance.shape[1] > 1:
                        X_mean = np.mean(X_for_variance, axis=1).reshape(-1, 1)
                    else:
                        X_mean = X_for_variance
                    X_var_quad = np.column_stack([X_mean, X_mean ** 2])
                    variance_model = LinearRegression()
                    variance_model.fit(X_var_quad, abs_residuals)
                    predicted_std = variance_model.predict(X_var_quad).flatten()
                    
                elif "Potęgowy" in variance_model_type:
                    # Dla modelu potęgowego bierzemy średnią z cech
                    if X_for_variance.shape[1] > 1:
                        X_mean = np.mean(X_for_variance, axis=1).reshape(-1, 1)
                    else:
                        X_mean = X_for_variance
                    log_xmean = np.log(np.maximum(np.abs(X_mean), 0.01))
                    log_std = np.log(np.maximum(abs_residuals, 0.01))
                    
                    variance_model = LinearRegression()
                    variance_model.fit(log_xmean, log_std)
                    predicted_std = np.exp(variance_model.predict(log_xmean)).flatten()
                    
                else:  # Logarytmiczny
                    if X_for_variance.shape[1] > 1:
                        X_mean = np.mean(X_for_variance, axis=1).reshape(-1, 1)
                    else:
                        X_mean = X_for_variance
                    X_var_log = np.log(np.maximum(np.abs(X_mean), 0.01))
                    variance_model = LinearRegression()
                    variance_model.fit(X_var_log, abs_residuals)
                    predicted_std = variance_model.predict(X_var_log).flatten()
                
                predicted_std = np.maximum(predicted_std, 0.01)
                
                # Obliczanie wag z zabezpieczeniem przed ekstremalnymi wartościami
                weights_full = 1 / (predicted_std ** 2)
                
                # POPRAWKA: Łagodniejsza normalizacja wag dla lepszej stabilności
                # Używamy percentyli zamiast minimum/maksimum
                p25, p75 = np.percentile(weights_full, [25, 75])
                weights_full = np.clip(weights_full, p25 * 0.5, p75 * 2.0)
                
                weights_full = weights_full / np.median(weights_full)
                
                weights_uniform = np.ones_like(weights_full)
                weights = weight_multiplier * weights_full + (1 - weight_multiplier) * weights_uniform
                weights = weights / np.mean(weights)
                
                # Finalna kontrola - zapobiegamy ekstremalnym wagom
                weights = np.clip(weights, 0.3, 3.0)
                
                if old_weights is not None:
                    weight_change = np.mean(np.abs(weights - old_weights))
                    weight_changes.append(weight_change)
                
                # Trenowanie modelu ważonego - używamy PCA dla modelu, ale oryginalne dane dla wag
                weighted_model = LinearRegression()
                weighted_model.fit(X_for_model, y_array, sample_weight=weights)
                y_pred_weighted = weighted_model.predict(X_for_model)
                residuals_weighted = y_array - y_pred_weighted
                
                scaled_residuals = residuals_weighted / np.sqrt(weights)
                variance_of_scaled = np.var(scaled_residuals)
                variance_stability.append(variance_of_scaled)
                
                # Metryki
                # R² ważone - pokazuje jak dobrze model pasuje do ważonych danych
                r2_weighted = weighted_model.score(X_for_model, y_array, sample_weight=weights)
                
                # MSE nieważone - dla porównania z modelem baseline
                mse_unweighted = np.mean(residuals_weighted ** 2)
                
                # MSE ważone - pokazuje błąd w kontekście wag
                mse_weighted_metric = np.average(residuals_weighted ** 2, weights=weights)
                
                # RMSE dla punktów o wysokich wagach (ważniejszych)
                high_weight_mask = weights > np.median(weights)
                mse_high_weight = np.mean(residuals_weighted[high_weight_mask] ** 2) if high_weight_mask.any() else mse_unweighted
                
                mse_history.append(mse_unweighted)
                r2_history.append(r2_weighted)
                
                iterations_results.append({
                    'iteration': iteration + 1,
                    'model': weighted_model,
                    'weights': weights.copy(),
                    'y_pred': y_pred_weighted,
                    'residuals': residuals_weighted,
                    'scaled_residuals': scaled_residuals,
                    'r2': r2_weighted,
                    'mse': mse_unweighted,
                    'mse_weighted': mse_weighted_metric,
                    'mse_high_weight': mse_high_weight,
                    'variance_stability': variance_of_scaled,
                    'weight_stats': {
                        'min': np.min(weights),
                        'max': np.max(weights),
                        'mean': np.mean(weights),
                        'median': np.median(weights),
                        'std': np.std(weights)
                    }
                })
                
                current_model = weighted_model
                current_residuals = residuals_weighted
        
        st.session_state.wr_iterations_results = iterations_results
        st.session_state.wr_model_initial = model_initial
        st.session_state.wr_y_pred_initial = y_pred_initial
        st.session_state.wr_mse_initial = mse_initial
        st.session_state.wr_r2_initial = r2_initial
        st.session_state.wr_bp_pvalue = bp_pvalue
        st.session_state.wr_X_for_model = X_for_model
        st.session_state.wr_X_scaled = X_scaled
        st.session_state.wr_y_array = y_array
        st.session_state.wr_features = selected_features
        st.session_state.wr_target = y_col
        st.session_state.wr_use_pca = use_pca
        st.session_state.wr_pca = pca
        st.session_state.wr_scaler = scaler
        st.session_state.wr_label_encoders = label_encoders
        st.session_state.wr_weight_changes = weight_changes
        st.session_state.wr_mse_history = mse_history
        st.session_state.wr_r2_history = r2_history
        st.session_state.wr_variance_stability = variance_stability
        st.session_state.wr_variance_model_type = variance_model_type
        st.session_state.wr_is_trained = True
        
        st.success("Model regresji ważonej został wytrenowany pomyślnie!")
        
    except Exception as e:
        st.error(f"Wystąpił błąd podczas trenowania modelu: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        st.session_state.wr_is_trained = False

if st.session_state.wr_is_trained:
    st.divider()
    
    iterations_results = st.session_state.wr_iterations_results
    
    #-----------------------------Analiza zbieżności IRLS-----------------------------
    st.markdown("### Analiza Zbieżności IRLS")
    
    if len(st.session_state.wr_weight_changes) > 0:
        fig_conv = go.Figure()
        fig_conv.add_trace(go.Scatter(
            x=list(range(1, len(st.session_state.wr_weight_changes) + 1)),
            y=st.session_state.wr_weight_changes,
            mode='lines+markers',
            name='Zmiana wag',
            line=dict(width=3, color='purple'),
            marker=dict(size=10)
        ))
        fig_conv.update_layout(
            title="Zbieżność algorytmu: Zmiana wag między iteracjami",
            xaxis_title="Iteracja",
            yaxis_title="Średnia zmiana wag",
            height=400
        )
        st.plotly_chart(fig_conv, use_container_width=True)
        
        final_change = st.session_state.wr_weight_changes[-1]
        if final_change < 0.01:
            st.success(f"Algorytm zbiegł się (zmiana wag: {final_change:.4f})")
        else:
            st.warning(f"Wagi nadal się zmieniają (zmiana: {final_change:.4f}) - rozważ więcej iteracji")
    
    st.divider()
    
    #--------------------------Wyświetl wyniki każdej iteracji--------------------------
    st.markdown("### Wyniki Iteracji IRLS")
    
    for idx, result in enumerate(iterations_results):
        with st.expander(f"Iteracja {result['iteration']}", expanded=(idx == len(iterations_results) - 1)):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("R² (ważone)", f"{result['r2']:.4f}")
            with col2:
                st.metric("MSE (ogólne)", f"{result['mse']:.2f}")
            with col3:
                st.metric("MSE (ważone)", f"{result['mse_weighted']:.2f}")
            
            col4, col5, col6 = st.columns(3)
            with col4:
                st.metric("MSE (ważne punkty)", f"{result['mse_high_weight']:.2f}")
            with col5:
                st.metric("Mediana wag", f"{result['weight_stats']['median']:.2f}")
            with col6:
                weight_range = result['weight_stats']['max'] - result['weight_stats']['min']
                st.metric("Zakres wag", f"{weight_range:.2f}")
            
            # Wykres regresji
            fig_weighted = go.Figure()
            
            X_model = st.session_state.wr_X_for_model
            y_arr = st.session_state.wr_y_array
            
            sort_idx = np.argsort(X_model.flatten())
            X_sorted = X_model[sort_idx]
            y_sorted = y_arr[sort_idx]
            y_pred_sorted = result['y_pred'][sort_idx]
            weights_sorted = result['weights'][sort_idx]
            
            fig_weighted.add_trace(go.Scatter(
                x=X_sorted.flatten(),
                y=y_sorted,
                mode='markers',
                name='Dane',
                marker=dict(
                    size=weights_sorted * 10,
                    opacity=0.6,
                    color=weights_sorted,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Waga")
                ),
                hovertemplate='<b>X:</b> %{x:.2f}<br><b>Y:</b> %{y:.2f}<br><b>Waga:</b> %{marker.color:.2f}<extra></extra>'
            ))
            
            fig_weighted.add_trace(go.Scatter(
                x=X_sorted.flatten(),
                y=y_pred_sorted,
                mode='lines',
                name='Regresja ważona',
                line=dict(color='green', width=2)
            ))
            
            if idx == 0:
                fig_weighted.add_trace(go.Scatter(
                    x=X_sorted.flatten(),
                    y=st.session_state.wr_y_pred_initial[sort_idx],
                    mode='lines',
                    name='Zwykła regresja (baseline)',
                    line=dict(color='orange', width=2, dash='dash')
                ))
            
            x_label = f"PC1 ({st.session_state.wr_pca.explained_variance_ratio_[0]:.1%} wariancji)" if st.session_state.wr_use_pca else st.session_state.wr_features[0]
            fig_weighted.update_layout(
                title=f"Regresja Ważona - Iteracja {result['iteration']}",
                xaxis_title=x_label,
                yaxis_title=st.session_state.wr_target,
                height=400
            )
            st.plotly_chart(fig_weighted, use_container_width=True)
            
            st.markdown("""
            **Interpretacja:**
            - Rozmiar i kolor punktów reprezentuje **wagę** - większe/jaśniejsze punkty mają większy wpływ na model
            - Model stara się lepiej dopasować do punktów z większymi wagami
            - Zielona linia pokazuje aktualną regresję ważoną
            """)
    
    st.divider()
    
    #-----------------------------------Porównanie--------------------------------
    st.markdown("### Porównanie Wyników")
    
    weight_changes_padded = [np.nan] + st.session_state.wr_weight_changes
    while len(weight_changes_padded) < len(iterations_results) + 1:
        weight_changes_padded.insert(1, np.nan)
    
    comparison_df = pd.DataFrame({
        'Iteracja': ['Początkowa (zwykła regresja)'] + [f"IRLS {r['iteration']}" for r in iterations_results],
        'R²': [st.session_state.wr_r2_initial] + [r['r2'] for r in iterations_results],
        'MSE ogólne': [st.session_state.wr_mse_initial] + [r['mse'] for r in iterations_results],
        'MSE ważone': [st.session_state.wr_mse_initial] + [r['mse_weighted'] for r in iterations_results],
        'MSE ważne punkty': [st.session_state.wr_mse_initial] + [r['mse_high_weight'] for r in iterations_results],
        'Zmiana wag': weight_changes_padded
    })
    
    st.dataframe(comparison_df, use_container_width=True)
    
    st.markdown("""
    **Wyjaśnienie metryk:**
    - **MSE ogólne**: Zwykły błąd średniokwadratowy - porównanie z baseline
    - **MSE ważone**: Błąd z uwzględnieniem wag - to jest optymalizowane przez IRLS
    - **MSE ważne punkty**: Błąd tylko dla punktów o wysokich wagach (najbardziej wiarygodnych)
    """)
    
    # Wykres porównawczy - wszystkie metryki MSE
    fig_comparison = go.Figure()
    
    fig_comparison.add_trace(go.Scatter(
        x=list(range(len(comparison_df))),
        y=comparison_df['MSE ogólne'],
        mode='lines+markers',
        name='MSE ogólne',
        line=dict(width=3, color='blue'),
        marker=dict(size=10)
    ))
    
    fig_comparison.add_trace(go.Scatter(
        x=list(range(len(comparison_df))),
        y=comparison_df['MSE ważone'],
        mode='lines+markers',
        name='MSE ważone (cel IRLS)',
        line=dict(width=3, color='green'),
        marker=dict(size=10)
    ))
    
    fig_comparison.add_trace(go.Scatter(
        x=list(range(len(comparison_df))),
        y=comparison_df['MSE ważne punkty'],
        mode='lines+markers',
        name='MSE ważne punkty',
        line=dict(width=3, color='orange'),
        marker=dict(size=10, symbol='diamond')
    ))
    
    fig_comparison.update_layout(
        title="Porównanie różnych miar błędu",
        xaxis_title="Iteracja",
        yaxis_title="MSE",
        xaxis=dict(
            ticktext=comparison_df['Iteracja'], 
            tickvals=list(range(len(comparison_df)))
        ),
        height=400,
        hovermode='x unified'
    )
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    st.divider()
    
    # Interpretacja wyników
    initial_mse = comparison_df['MSE ogólne'].iloc[0]
    final_mse = comparison_df['MSE ogólne'].iloc[-1]
    improvement_general = (initial_mse - final_mse) / initial_mse * 100
    
    initial_mse_weighted = comparison_df['MSE ważone'].iloc[0]
    final_mse_weighted = comparison_df['MSE ważone'].iloc[-1]
    improvement_weighted = (initial_mse_weighted - final_mse_weighted) / initial_mse_weighted * 100
    
    initial_mse_high = comparison_df['MSE ważne punkty'].iloc[0]
    final_mse_high = comparison_df['MSE ważne punkty'].iloc[-1]
    improvement_high = (initial_mse_high - final_mse_high) / initial_mse_high * 100
    
    st.markdown("### Podsumowanie")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**MSE Ważone**")
        if improvement_weighted > 5:
            st.success(f"Poprawa o **{improvement_weighted:.1f}%**\n\nIRLS skutecznie zoptymalizował błąd ważony")
        elif improvement_weighted > 0:
            st.info(f"Poprawa o **{improvement_weighted:.1f}%**")
        else:
            st.warning(f"Pogorszenie o **{-improvement_weighted:.1f}%**")
    
    with col2:
        st.markdown("**MSE Ważne Punkty**")
        if improvement_high > 5:
            st.success(f"Poprawa o **{improvement_high:.1f}%**\n\nModel lepiej przewiduje dla wiarygodnych danych")
        elif improvement_high > 0:
            st.info(f"Poprawa o **{improvement_high:.1f}%**")
        else:
            st.warning(f"Pogorszenie o **{-improvement_high:.1f}%**")
    
    with col3:
        st.markdown("**MSE Ogólne**")
        if improvement_general > 5:
            st.success(f"Poprawa o **{improvement_general:.1f}%**")
        elif improvement_general > 0:
            st.info(f"Poprawa o **{improvement_general:.1f}%**")
        elif improvement_general > -10:
            st.info(f"Zmiana o **{improvement_general:.1f}%**\n\n Nieważone MSE może się pogorszyć")
        else:
            st.warning(f"Pogorszenie o **{-improvement_general:.1f}%**\n\n**Możliwe przyczyny:**\n- Dane homoskedastyczne (p={st.session_state.wr_bp_pvalue:.3f})\n- Model wariancji nieodpowiedni\n- Za mało iteracji")
    
    st.divider()
    
    #-----------------------------------Testowanie modelu--------------------------------
    st.markdown("### Testowanie Modelu")
    st.markdown("Wprowadź wartości dla nowej próbki:")
    
    final_model = iterations_results[-1]['model']
    
    input_data = {}
    num_features = len(st.session_state.wr_features)
    cols_per_row = min(4, num_features)
    
    for idx, feature in enumerate(st.session_state.wr_features):
        if idx % cols_per_row == 0:
            cols = st.columns(cols_per_row)
        
        with cols[idx % cols_per_row]:
            if feature in st.session_state.wr_label_encoders:
                le = st.session_state.wr_label_encoders[feature]
                options = list(le.classes_)
                input_data[feature] = st.selectbox(
                    feature,
                    options,
                    key=f"wr_input_{feature}"
                )
            else:
                min_val = float(df[feature].min())
                max_val = float(df[feature].max())
                mean_val = float(df[feature].mean())
                step = (max_val - min_val) / 100 if max_val > min_val else 1.0
                
                input_data[feature] = st.number_input(
                    feature,
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    step=step,
                    key=f"wr_input_{feature}"
                )
    
    if st.button("Przewiduj wartość", type="primary"):
        try:
            input_df = pd.DataFrame([input_data])
            
            for col in input_df.columns:
                if col in st.session_state.wr_label_encoders:
                    le = st.session_state.wr_label_encoders[col]
                    input_df[col] = le.transform(input_df[col].astype(str))
            
            X_input = input_df.values
            
            if st.session_state.wr_use_pca:
                X_input_scaled = st.session_state.wr_scaler.transform(X_input)
                X_input_transformed = st.session_state.wr_pca.transform(X_input_scaled)
                prediction = final_model.predict(X_input_transformed)[0]
            else:
                prediction = final_model.predict(X_input)[0]
            
            st.success(f"Przewidywana wartość dla **{st.session_state.wr_target}**: **{prediction:.4f}**")
            
            # Porównanie z modelem baseline
            if st.session_state.wr_use_pca:
                prediction_baseline = st.session_state.wr_model_initial.predict(X_input_transformed)[0]
            else:
                prediction_baseline = st.session_state.wr_model_initial.predict(X_input)[0]
            
            st.info(f"Przewidywanie modelu bazowego (bez ważenia): **{prediction_baseline:.4f}**")
            
            diff = abs(prediction - prediction_baseline)
            if diff > 0.01:
                st.write(f"Różnica między modelami: **{diff:.4f}**")
            
        except Exception as e:
            st.error(f"Błąd podczas predykcji: {str(e)}")
            import traceback
            st.code(traceback.format_exc())