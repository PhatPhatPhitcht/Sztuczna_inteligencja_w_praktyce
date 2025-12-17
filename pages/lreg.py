import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.set_page_config(page_title="Interaktywne algorytmy uczenia maszynowego", initial_sidebar_state="collapsed")
st.markdown("""
    <style>
        [data-testid="stSidebarNav"] {display: none;}
    </style>
""", unsafe_allow_html=True)

st.page_link("main.py", label="⬅️ Powrót do strony głównej")

st.subheader("Regresja liniowa i wieloraka")

st.markdown("""
**Regresja liniowa wieloraka** (ang. *multiple linear regression*) jest metodą uczenia maszynowego 
            służącą do modelowania zależności pomiędzy jedną zmienną zależną (y) a wieloma zmiennymi niezależnymi (X₁, X₂, …, Xₙ).

Model zakłada, że wpływ każdej zmiennej niezależnej na wynik jest liniowy, a końcowa predykcja jest sumą ważoną tych wpływów 
            oraz wyrazu wolnego. Każdy współczynnik regresji informuje, o ile średnio zmieni się wartość zmiennej zależnej, 
            gdy dana cecha wzrośnie o jedną jednostkę, przy założeniu, że pozostałe cechy pozostają bez zmian.

**Metryki**
            
- **R²** pokazuje ogólną jakość dopasowania modelu
- **RMSE** mówi, jak duży jest typowy błąd predykcji
- **MAE** daje intuicyjną informację o przeciętnym odchyleniu
- **MSE** pomaga wykrywać duże błędy i problemy z outlierami
            
R² powinno dążyć do 1, a reszta do 0
            
[Dowiedz się więcej](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
            
*Poniżej możesz wytrenować model dla twoich danych i zobaczyć jak dobrze regresja liniowa się sprawdza do problemu!*
""")

for key, default in {
    "model": None,
    "scaler": None,
    "features": None,
    "target": None,
    "is_trained": False
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


if "df" not in st.session_state:
    st.error("Brak danych w st.session_state.df")
    st.stop()

df = st.session_state.df.copy()

st.divider()

with st.expander("Podgląd danych"):
    st.dataframe(st.session_state.df.head())

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if len(numeric_cols) < 2:
    st.error("Wymagane są co najmniej 2 kolumny numeryczne.")
    st.stop()

target = st.selectbox(
    "Wybierz kolumnę docelową",
    numeric_cols,
    index=numeric_cols.index(st.session_state.target)
    if st.session_state.target in numeric_cols else 0
)

if st.button("Trenuj model", type="primary"):
    features = [c for c in numeric_cols if c != target]

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    y_pred_test = model.predict(X_test_scaled)
    y_pred_train = model.predict(X_train_scaled)

    st.session_state.model = model
    st.session_state.scaler = scaler
    st.session_state.features = features
    st.session_state.target = target
    st.session_state.is_trained = True

    st.session_state.metrics_test = {
        "R2": r2_score(y_test, y_pred_test),
        "MSE": mean_squared_error(y_test, y_pred_test),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_test)),
        "MAE": mean_absolute_error(y_test, y_pred_test)
    }
    
    st.session_state.metrics_train = {
        "R2": r2_score(y_train, y_pred_train),
        "MSE": mean_squared_error(y_train, y_pred_train),
        "RMSE": np.sqrt(mean_squared_error(y_train, y_pred_train)),
        "MAE": mean_absolute_error(y_train, y_pred_train)
    }

    st.session_state.residuals = y_test - y_pred_test
    st.session_state.y_pred = y_pred_test
    st.session_state.y_test = y_test
    st.session_state.X_test_scaled = X_test_scaled


if st.session_state.is_trained:
    st.markdown("**Metryki modelu**")
    

    st.markdown("*Zbiór treningowy*")
    m_train = st.session_state.metrics_train
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R²", f"{m_train['R2']:.4f}")
    c2.metric("MSE", f"{m_train['MSE']:.4f}")
    c3.metric("RMSE", f"{m_train['RMSE']:.4f}")
    c4.metric("MAE", f"{m_train['MAE']:.4f}")
    
    st.markdown("*Zbiór testowy*")
    m_test = st.session_state.metrics_test
    c1, c2, c3, c4 = st.columns(4)
    
    # Delta pokazuje różnicę (im mniejsza tym lepiej - brak overfittingu)
    r2_delta = m_test['R2'] - m_train['R2']
    mse_delta = m_test['MSE'] - m_train['MSE']
    rmse_delta = m_test['RMSE'] - m_train['RMSE']
    mae_delta = m_test['MAE'] - m_train['MAE']
    
    c1.metric("R²", f"{m_test['R2']:.4f}", f"{r2_delta:+.4f}")
    c2.metric("MSE", f"{m_test['MSE']:.4f}", f"{mse_delta:+.4f}", delta_color="inverse")
    c3.metric("RMSE", f"{m_test['RMSE']:.4f}", f"{rmse_delta:+.4f}", delta_color="inverse")
    c4.metric("MAE", f"{m_test['MAE']:.4f}", f"{mae_delta:+.4f}", delta_color="inverse")
    
    # Pierwszy argument - label, Drugi argument - główna wartość do wyświetlenia, Trzeci argument - różnica, która automatycznie tworzy strzałkę

    st.markdown("""
        *Zbyt duże różnice między train / test mogą świadczyć o przeuczenieuczeniu.*
    """)

    st.markdown("**Wykresy diagnostyczne**")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Wykres : Wartości rzeczywiste vs. przewidywane (linia regresji)
    ax1.scatter(st.session_state.y_test, st.session_state.y_pred, 
                alpha=0.6, edgecolors='k', linewidth=0.5, s=50)
    
    min_val = min(st.session_state.y_test.min(), st.session_state.y_pred.min())
    max_val = max(st.session_state.y_test.max(), st.session_state.y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Idealna predykcja')
    
    ax1.set_xlabel('Wartości rzeczywiste', fontsize=11)
    ax1.set_ylabel('Wartości przewidywane', fontsize=11)
    ax1.set_title('Wartości rzeczywiste vs. Przewidywane', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Wykres : Wykres reszt
    ax2.scatter(st.session_state.y_pred, st.session_state.residuals, 
                alpha=0.6, edgecolors='k', linewidth=0.5, s=50)
    ax2.axhline(y=0, color='r', linestyle='--', lw=2, label='Reszty = 0')
    ax2.set_xlabel('Wartości przewidywane', fontsize=11)
    ax2.set_ylabel('Reszty (błędy)', fontsize=11)
    ax2.set_title('Wykres reszt', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)

    # Wykres PCA (jeśli więcej niż 2 cechy)
    if len(st.session_state.features) > 2:
        st.markdown("**Wizualizacja PCA z rozkładem reszt**")
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(st.session_state.X_test_scaled)

        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=st.session_state.residuals,
            cmap='RdYlGn_r',
            alpha=0.6,
            edgecolors='k',
            linewidth=0.5,
            s=50
        )
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} wariancji)', fontsize=11)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} wariancji)', fontsize=11)
        ax.set_title('Rozkład reszt w przestrzeni PCA', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Reszty', rotation=270, labelpad=20)
        
        st.pyplot(fig)

    st.markdown("""
        **Zielone** = małe błędy (model dobrze przewidział)
                
        **Żółte** = średnie błędy
                
        **Czerwone** = duże błędy (model się pomylił)
                
        Jeśli błędy są rozłożone losowo model jest OK, jeśli czerwone punkty tworzą klaster model ma problem z pewnym typem danych,
                a jeżeli błędy tworzą wzór model nie uchwycił jakiejś zależności.

    """)

    st.markdown("**Testowanie modelu**")

    input_data = {}
    num_features = len(st.session_state.features)
    cols_per_row = min(4, num_features)
    
    for i, feature in enumerate(st.session_state.features):
        # Co n-tą cechę twórz nowy wiersz
        if i % cols_per_row == 0:
            cols = st.columns(cols_per_row)
        
        with cols[i % cols_per_row]:
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
                key=f"input_{feature}"
            )

    if st.button("Klasyfikuj próbkę", type="primary"):
        input_df = pd.DataFrame([input_data])
        input_scaled = st.session_state.scaler.transform(input_df)
        prediction = st.session_state.model.predict(input_scaled)[0]

        st.success(f"Przewidywana wartość: **{prediction:.4f}**")