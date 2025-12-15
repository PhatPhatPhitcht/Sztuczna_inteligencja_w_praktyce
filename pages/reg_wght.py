import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from scipy import stats

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

df = st.session_state.df

st.markdown("""
**Regresja ważona** to rozszerzenie regresji liniowej, które pozwala nadawać różne znaczenie poszczególnym 
            obserwacjom w danych. Niektóre punkty mogą być bardziej wiarygodne (np. precyzyjniejsze pomiary) lub mniej wiarygodne 
            (np. zawierające większy szum), a regresja ważona to uwzględnia. Wagi mogą być nadane manualnie, lub metodą IRLS. 
            W tym przypadku skupimy się na tym drugim.

**Jak to działa?**
Model stara się lepiej do nich dopasować linię regresji do bardziej wiarygodnych punktów. Punkty z małą wagą mają mniejszy 
            wpływ na wynik i model może je w pewnym stopniu ignorować.

**IRLS (Iteratively Reweighted Least Squares)** to algorytm, który automatycznie oblicza wagi, gdy nie znamy ich z góry. Działa w iteracjach:
- W każdej iteracji model analizuje błędy przewidywań (reszty)
- Na podstawie tych błędów szacuje, które obszary danych są bardziej/mniej stabilne
- Przydziela większe wagi punktom w stabilnych obszarach, mniejsze w niestabilnych
- Powtarza proces, aż wagi przestają się zmieniać (zbieżność)

**Parametry:**

**Siła ważenia** kontroluje jak mocno algorytm stosuje wagi:
- 0% = ignoruje wagi całkowicie (zwykła regresja liniowa)
- 100% = pełne ważenie (maksymalny efekt IRLS)
- Wartości pośrednie (np. 50%) to kompromis – łagodniejsze ważenie, które może działać lepiej gdy dane są trudne

**Liczba iteracji IRLS** określa ile razy algorytm przeliczy wagi:
- 1 iteracja = szybkie, ale może nie wystarczyć do pełnej korekcji
- 3-5 iteracji = typowo wystarczające (algorytm zbiega się)
- 10+ iteracji = rzadko potrzebne, chyba że dane są bardzo trudne

Więcej iteracji nie zawsze oznacza lepszy model – jeśli algorytm zbiegł się wcześniej, dodatkowe iteracje nic nie zmienią.

[Regresja liniowa](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
            
[Modele liniowe](https://scikit-learn.org/stable/modules/linear_model.html)                       
""")

st.divider()

with st.expander("Podgląd danych"):
    st.dataframe(st.session_state.df.head())

st.markdown("### Wybór zmiennych")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if len(numeric_cols) < 2:
    st.error("Potrzeba co najmniej 2 kolumn numerycznych")
    st.stop()

y_col = st.selectbox("Wybierz kolumnę docelową (wartość do przewidzenia):", numeric_cols)

x_cols = [col for col in numeric_cols if col != y_col]

if not x_cols:
    st.error("Brak zmiennych niezależnych - potrzeba więcej niż jedna kolumna numeryczna")
    st.stop()
# Przygotowywanie danych
X = df[x_cols].values
y = df[y_col].values

mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
X = X[mask]
y = y[mask]

if len(X) < 10:
    st.error("Za mało danych po usunięciu wartości NaN")
    st.stop()

# Standaryzacja i PCA
if X.shape[1] > 1:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=1)
    X_pca = pca.fit_transform(X_scaled)
    
    X_for_model = X_pca
else:
    X_for_model = X
    scaler = None
    pca = None

# Podział na zbiory train/test
test_size = min(int(len(X_for_model) * 0.2), 20)
if test_size < 5:
    test_size = max(5, len(X_for_model) // 5)

X_train, X_test, y_train, y_test = train_test_split(
    X_for_model, y, test_size=test_size, random_state=42
)

col1, col2 = st.columns(2)
with col1:
    st.metric("Liczba próbek", len(X))
with col2:
    st.metric("Liczba cech", len(x_cols))

st.divider()

#--------------------------Test heteroskedastyczności------------------------------
st.markdown("### Test Heteroskedastyczności")
st.markdown("""
**Heteroskedastyczność** oznacza, że błędy przewidywań modelu (reszty) mają różną wariancję w różnych obszarach danych. 
            Model popełnia większe błędy w niektórych regionach, a mniejsze w innych.

**Dlaczego to problem?**
Zwykła regresja liniowa zakłada, że błędy mają stałą wariancję (homoskedastyczność). Gdy to założenie jest naruszone:
- Przedziały ufności są niepoprawne
- Testy statystyczne dają błędne wyniki
- Model jest mniej efektywny

**Test Breuscha-Pagana** służy do statystycznej weryfikacji występowania heteroskedastyczności. Polega on na sprawdzeniu, 
            czy wariancja składnika losowego (reszt z modelu regresji liniowej) zależy od wartości zmiennych objaśniających (X).

**Interpretacja wyników:**

**Statystyka BP** to liczba, która mierzy siłę zależności między wartościami X a wielkością błędów. 
            Im wyższa wartość, tym silniejsza heteroskedastyczność.

**Wartość p** to prawdopodobieństwo, że zaobserwowane dane mogły powstać przypadkowo, gdyby nie było heteroskedastyczności:
- p < 0.05: Heteroskedastyczność jest obecna (regresja ważona ma sens!)
- p > 0.05: Dane są homoskedastyczne (zwykła regresja wystarczy)

[Dowiedz się więcej](https://homepage.univie.ac.at/robert.kunst/emwipres121.pdf)
""")

model_initial = LinearRegression()
model_initial.fit(X_train, y_train)
y_pred_train = model_initial.predict(X_train)
residuals_initial = y_train - y_pred_train

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

bp_stat, bp_pvalue = breusch_pagan_test(X_train, residuals_initial)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Statystyka BP", f"{bp_stat:.4f}")
with col2:
    st.metric("Wartość p", f"{bp_pvalue:.4f}")
with col3:
    y_pred_test = model_initial.predict(X_test)
    test_mse = np.mean((y_test - y_pred_test) ** 2)
    st.metric("MSE (test)", f"{test_mse:.4f}")

if bp_pvalue > 0.05:
    st.warning("Dane się homoskedastyczne (p > 0.05). Regresja ważona może nie być potrzebna.")
    heteroskedastic = False
else:
    st.success("Wykryto heteroskedastyczność (p < 0.05). Regresja ważona jest zalecana.")
    heteroskedastic = True

st.divider()

#---------------------------------Wykres reszt---------------------------------
st.markdown("### Wykres Reszt")
st.markdown("""
**Wykres reszt** (residual plot) pokazuje błędy modelu w funkcji wartości przewidywanych.
Reszta to różnica między rzeczywistą wartością a przewidywaną: reszta = y_rzeczywiste - y_przewidywane
- Dodatnia reszta = model przewidział za mało
- Ujemna reszta = model przewidział za dużo
- Reszta bliska zero = model trafił

**Jak czytać wykres?**

**Oś X (wartości przewidywane):** Pokazuje co model przewidział dla danego punktu
**Oś Y (reszty):** Pokazuje o ile model się pomylił
**Linia y=0:** Idealna sytuacja – brak błędu

**Punkty niebieskie (Train):** Błędy na zbiorze treningowym
**Punkty czerwone (Test):** Błędy na zbiorze testowym (ważniejsze dla oceny generalizacji!)

**W kontekście regresji ważonej:**
Przed zastosowaniem wag często widać "lejek" – reszty rosną w jednym kierunku, co świadczy o heteroskedastyczności danych.
""")

fig_residuals = go.Figure()

fig_residuals.add_trace(go.Scatter(
    x=y_pred_train,
    y=residuals_initial,
    mode='markers',
    marker=dict(size=6, opacity=0.6, color='blue'),
    name='Train'
))

y_pred_test_initial = model_initial.predict(X_test)
residuals_test = y_test - y_pred_test_initial
fig_residuals.add_trace(go.Scatter(
    x=y_pred_test_initial,
    y=residuals_test,
    mode='markers',
    marker=dict(size=6, opacity=0.6, color='red'),
    name='Test'
))

fig_residuals.add_hline(y=0, line_dash="dash", line_color="gray")
fig_residuals.update_layout(
    title="Reszty vs Wartości Przewidywane",
    xaxis_title="Wartości przewidywane",
    yaxis_title="Reszty",
    height=400
)
st.plotly_chart(fig_residuals, use_container_width=True)
st.divider()
#------------------------------Zwykła regresja liniowa---------------------------
st.markdown("### Zwykła Regresja Liniowa")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("R² (train)", f"{model_initial.score(X_train, y_train):.4f}")
with col2:
    mse_initial = np.mean(residuals_initial ** 2)
    st.metric("MSE (test)", f"{mse_initial:.4f}")
with col3:
    r2_test = model_initial.score(X_test, y_test)
    st.metric("R² (test)", f"{r2_test:.4f}")

fig_linear = go.Figure()

sort_idx_train = np.argsort(X_train.flatten())
X_train_sorted = X_train[sort_idx_train]
y_train_sorted = y_train[sort_idx_train]
y_pred_train_sorted = y_pred_train[sort_idx_train]

fig_linear.add_trace(go.Scatter(
    x=X_train_sorted.flatten(),
    y=y_train_sorted,
    mode='markers',
    name='Train',
    marker=dict(size=8, opacity=0.6, color='blue')
))

fig_linear.add_trace(go.Scatter(
    x=X_test.flatten(),
    y=y_test,
    mode='markers',
    name='Test',
    marker=dict(size=8, opacity=0.6, color='red')
))

fig_linear.add_trace(go.Scatter(
    x=X_train_sorted.flatten(),
    y=y_pred_train_sorted,
    mode='lines',
    name='Regresja liniowa',
    line=dict(color='green', width=2)
))

x_label = "PC1" if X.shape[1] > 1 else x_cols[0]
fig_linear.update_layout(
    title="Dopasowanie Regresji Liniowej",
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

train_button = st.button("Trenuj", type="primary")

#------------------------------Główny algorytm------------------------------
if train_button or 'iterations_results' in st.session_state:
    if train_button:
        iterations_results = []
        weights = np.ones(len(y_train))
        current_model = model_initial
        current_residuals = residuals_initial
        
        weight_changes = []
        mse_train_history = []
        mse_test_history = []
        variance_stability = []
        
        with st.spinner('Trenuję model...'):
            for iteration in range(n_iterations):
                old_weights = weights.copy() if iteration > 0 else None
                
                abs_residuals = np.abs(current_residuals)
                abs_residuals = np.maximum(abs_residuals, 0.01)
                
                y_pred_for_variance = current_model.predict(X_train)
                
                if "Liniowy" in variance_model_type:
                    X_var = y_pred_for_variance.reshape(-1, 1)
                    variance_model = LinearRegression()
                    variance_model.fit(X_var, abs_residuals)
                    predicted_std = variance_model.predict(X_var)
                    
                elif "Kwadratowy" in variance_model_type:
                    X_var = np.column_stack([
                        y_pred_for_variance,
                        y_pred_for_variance ** 2
                    ])
                    variance_model = LinearRegression()
                    variance_model.fit(X_var, abs_residuals)
                    predicted_std = variance_model.predict(X_var)
                    
                elif "Potęgowy" in variance_model_type:
                    log_ypred = np.log(np.maximum(y_pred_for_variance, 0.01))
                    log_std = np.log(np.maximum(abs_residuals, 0.01))
                    
                    variance_model = LinearRegression()
                    variance_model.fit(log_ypred.reshape(-1, 1), log_std)
                    
                    predicted_std = np.exp(variance_model.predict(log_ypred.reshape(-1, 1)))
                    
                else:  # Logarytmiczny
                    X_var = np.log(np.maximum(y_pred_for_variance, 0.01)).reshape(-1, 1)
                    variance_model = LinearRegression()
                    variance_model.fit(X_var, abs_residuals)
                    predicted_std = variance_model.predict(X_var)
                
                predicted_std = np.maximum(predicted_std, 0.01)
                
                weights_full = 1 / (predicted_std ** 2)
                weights_uniform = np.ones_like(weights_full)
                weights = weight_multiplier * weights_full + (1 - weight_multiplier) * weights_uniform
                weights = weights / np.mean(weights)
                
                if old_weights is not None:
                    weight_change = np.mean(np.abs(weights - old_weights))
                    weight_changes.append(weight_change)
                
                weighted_model = LinearRegression()
                weighted_model.fit(X_train, y_train, sample_weight=weights)
                y_pred_weighted_train = weighted_model.predict(X_train)
                residuals_weighted = y_train - y_pred_weighted_train
                
                scaled_residuals = residuals_weighted / np.sqrt(weights)
                variance_of_scaled = np.var(scaled_residuals)
                variance_stability.append(variance_of_scaled)
                
                # Ewaluacja modelu
                y_pred_weighted_test = weighted_model.predict(X_test)
                test_mse_weighted = np.mean((y_test - y_pred_weighted_test) ** 2)
                test_r2_weighted = weighted_model.score(X_test, y_test)
                
                r2_weighted = weighted_model.score(X_train, y_train, sample_weight=weights)
                mse_weighted = np.average(residuals_weighted ** 2, weights=weights)
                
                mse_train_history.append(mse_weighted)
                mse_test_history.append(test_mse_weighted)
                
                iterations_results.append({
                    'iteration': iteration + 1,
                    'model': weighted_model,
                    'weights': weights.copy(),
                    'y_pred_train': y_pred_weighted_train,
                    'y_pred_test': y_pred_weighted_test,
                    'residuals': residuals_weighted,
                    'scaled_residuals': scaled_residuals,
                    'r2_train': r2_weighted,
                    'r2_test': test_r2_weighted,
                    'mse_train': mse_weighted,
                    'mse_test': test_mse_weighted,
                    'variance_stability': variance_of_scaled
                })
                
                current_model = weighted_model
                current_residuals = residuals_weighted
        
        st.session_state.iterations_results = iterations_results
        st.session_state.model_initial = model_initial
        st.session_state.y_pred_train = y_pred_train
        st.session_state.y_pred_test = model_initial.predict(X_test)
        st.session_state.mse_initial = mse_initial
        st.session_state.test_mse = test_mse
        st.session_state.bp_pvalue = bp_pvalue
        st.session_state.r2_test = r2_test
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        st.session_state.x_cols = x_cols
        st.session_state.y_col = y_col
        st.session_state.X_orig_shape = X.shape[1]
        st.session_state.weight_changes = weight_changes
        st.session_state.mse_train_history = mse_train_history
        st.session_state.mse_test_history = mse_test_history
        st.session_state.variance_stability = variance_stability
        st.session_state.variance_model_type = variance_model_type
    
    iterations_results = st.session_state.iterations_results
    
    #-----------------------------Analiza zbieżności IRLS-----------------------------
    st.markdown("### Analiza Zbieżności IRLS")
    
    # Wykres: Zmiana wag między iteracjami
    if len(st.session_state.weight_changes) > 0:
        fig_conv = go.Figure()
        fig_conv.add_trace(go.Scatter(
            x=list(range(1, len(st.session_state.weight_changes) + 1)),
            y=st.session_state.weight_changes,
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
        
        final_change = st.session_state.weight_changes[-1]
        if final_change < 0.01:
            st.success(f"Algorytm zbiegł się (zmiana wag: {final_change:.4f})")
        else:
            st.warning(f"Wagi nadal się zmieniają (zmiana: {final_change:.4f}) - rozważ więcej iteracji")
    
    #--------------------------Wyświetl wyniki każdej iteracji--------------------------
    st.markdown("### Wyniki Iteracji IRLS")
    
    for idx, result in enumerate(iterations_results):
        with st.expander(f" Iteracja {result['iteration']}", expanded=(idx == len(iterations_results) - 1)):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("R² (train)", f"{result['r2_train']:.4f}")
            with col2:
                st.metric("MSE (train)", f"{result['mse_train']:.4f}")
            with col3:
                st.metric("R² (test)", f"{result['r2_test']:.4f}")
            with col4:
                st.metric("MSE (test)", f"{result['mse_test']:.4f}")
            
            # Wykres regresji
            fig_weighted = go.Figure()
            
            X_train_ss = st.session_state.X_train
            y_train_ss = st.session_state.y_train
            X_test_ss = st.session_state.X_test
            y_test_ss = st.session_state.y_test
            
            sort_idx = np.argsort(X_train_ss.flatten())
            X_train_sorted = X_train_ss[sort_idx]
            y_train_sorted = y_train_ss[sort_idx]
            y_pred_sorted = result['y_pred_train'][sort_idx]
            weights_sorted = result['weights'][sort_idx]
            
            fig_weighted.add_trace(go.Scatter(
                x=X_train_sorted.flatten(),
                y=y_train_sorted,
                mode='markers',
                name='Train',
                marker=dict(
                    size=weights_sorted * 10,
                    opacity=0.6,
                    color=weights_sorted,
                    colorscale='Viridis',
                    showscale=False,
                    colorbar=dict(title="Waga")
                )
            ))
            
            fig_weighted.add_trace(go.Scatter(
                x=X_test_ss.flatten(),
                y=y_test_ss,
                mode='markers',
                name='Test',
                marker=dict(size=8, opacity=0.6, color='red')
            ))
            
            fig_weighted.add_trace(go.Scatter(
                x=X_train_sorted.flatten(),
                y=y_pred_sorted,
                mode='lines',
                name='Regresja ważona',
                line=dict(color='green', width=2)
            ))
            
            if idx == 0:
                fig_weighted.add_trace(go.Scatter(
                    x=X_train_sorted.flatten(),
                    y=st.session_state.y_pred_train[sort_idx],
                    mode='lines',
                    name='Zwykła regresja',
                    line=dict(color='orange', width=2, dash='dash')
                ))
            
            x_label = "PC1" if st.session_state.X_orig_shape > 1 else st.session_state.x_cols[0]
            fig_weighted.update_layout(
                title=f"Regresja Ważona - Iteracja {result['iteration']}",
                xaxis_title=x_label,
                yaxis_title=st.session_state.y_col,
                height=400
            )
            st.plotly_chart(fig_weighted, use_container_width=True)
    
    #-----------------------------------Porównanie--------------------------------
    st.markdown("### Porównanie Wyników")
    
    # lista zmian wag 
    weight_changes_padded = [np.nan] + st.session_state.weight_changes
    while len(weight_changes_padded) < len(iterations_results) + 1:
        weight_changes_padded.insert(1, np.nan)
    
    comparison_df = pd.DataFrame({
        'Iteracja': ['Początkowa'] + [f"IRLS {r['iteration']}" for r in iterations_results],
        'R² (train)': [st.session_state.model_initial.score(st.session_state.X_train, st.session_state.y_train)] + [r['r2_train'] for r in iterations_results],
        'R² (test)': [st.session_state.r2_test] + [r['r2_test'] for r in iterations_results],
        'MSE (train)': [st.session_state.mse_initial] + [r['mse_train'] for r in iterations_results],
        'MSE (test)': [st.session_state.test_mse] + [r['mse_test'] for r in iterations_results],
        'Zmiana wag': weight_changes_padded
    })
    
    st.dataframe(comparison_df, use_container_width=True)
#    
#    # Wykres porównawczy - MSE
#    fig_comparison = go.Figure()
#    fig_comparison.add_trace(go.Scatter(
#        x=list(range(len(comparison_df))),
#        y=comparison_df['MSE (test)'],
#        mode='lines+markers',
#        name='MSE Test',
#        line=dict(width=3, color='red')
#    ))
#    fig_comparison.add_trace(go.Scatter(
#        x=list(range(len(comparison_df))),
#        y=comparison_df['MSE (train)'],
#        mode='lines+markers',
#        name='MSE Train',
#        line=dict(width=3, color='blue')
#    ))
#    fig_comparison.update_layout(
#        title="Poprawa modelu: czy IRLS zmniejsza błąd?",
#        xaxis_title="Iteracja",
#        yaxis_title="MSE",
#        xaxis=dict(ticktext=comparison_df['Iteracja'], 
#                   tickvals=list(range(len(comparison_df)))),
#        height=400
#    )
#    st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Interpretacja wyników
    initial_mse_test = comparison_df['MSE (test)'].iloc[0]
    final_mse_test = comparison_df['MSE (test)'].iloc[-1]
    improvement = (initial_mse_test - final_mse_test) / initial_mse_test * 100
    
    st.markdown("### Podsumowanie")
    
    col1, col2 = st.columns(2)
    with col1:
        if improvement > 5:
            st.success(f" **IRLS działa!**\n\nMSE na zbiorze testowym poprawiło się o **{improvement:.1f}%**")
        elif improvement > 0:
            st.info(f" **Niewielka poprawa**\n\nMSE na zbiorze testowym poprawiło się o **{improvement:.1f}%**")
        else:
            st.warning(f" **IRLS nie pomogło**\n\nMSE pogorszyło się o **{-improvement:.1f}%**\n\nMożliwe przyczyny:\n- Model wariancji za prosty\n- Heteroskedastyczność grupowa (nie wykryta przez PCA)\n- Dane homoskedastyczne (p={st.session_state.bp_pvalue:.3f})")
    
    with col2:
        if len(st.session_state.weight_changes) > 0:
            final_change = st.session_state.weight_changes[-1]
            
            if final_change < 0.01:
                st.success(f" **Zbieżność osiągnięta**\n\nWagi stabilne (zmiana: {final_change:.4f})")
            else:
                st.warning(f" **Brak pełnej zbieżności**\n\nWagi nadal się zmieniają (zmiana: {final_change:.4f})\n\nRozważ więcej iteracji.")