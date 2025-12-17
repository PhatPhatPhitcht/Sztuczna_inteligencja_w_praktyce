import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
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
    
    r2_delta = m_test['R2'] - m_train['R2']
    mse_delta = m_test['MSE'] - m_train['MSE']
    rmse_delta = m_test['RMSE'] - m_train['RMSE']
    mae_delta = m_test['MAE'] - m_train['MAE']
    
    c1.metric("R²", f"{m_test['R2']:.4f}", f"{r2_delta:+.4f}")
    c2.metric("MSE", f"{m_test['MSE']:.4f}", f"{mse_delta:+.4f}", delta_color="inverse")
    c3.metric("RMSE", f"{m_test['RMSE']:.4f}", f"{rmse_delta:+.4f}", delta_color="inverse")
    c4.metric("MAE", f"{m_test['MAE']:.4f}", f"{mae_delta:+.4f}", delta_color="inverse")

    st.markdown("""
        *Zbyt duże różnice między train / test mogą świadczyć o przeuczenieuczeniu.*
    """)

    st.markdown("**Wykresy diagnostyczne**")
    
    #col1, col2 = st.columns([5, 5])
    
    #with col1:
        # Wykres 1: Wartości rzeczywiste vs. przewidywane
    fig1 = go.Figure()
    
    fig1.add_trace(
        go.Scatter(
            x=st.session_state.y_test,
            y=st.session_state.y_pred,
            mode='markers',
            name='Predykcje',
            marker=dict(
                size=8,
                color='steelblue',
                opacity=0.6,
                line=dict(width=0.5, color='darkblue')
            ),
            hovertemplate='<b>Rzeczywiste:</b> %{x:.2f}<br><b>Przewidywane:</b> %{y:.2f}<extra></extra>'
        )
    )
    
    min_val = min(st.session_state.y_test.min(), st.session_state.y_pred.min())
    max_val = max(st.session_state.y_test.max(), st.session_state.y_pred.max())
    
    fig1.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Idealna predykcja',
            line=dict(color='red', width=2, dash='dash'),
            hovertemplate='Idealna predykcja<extra></extra>'
        )
    )
    
    fig1.update_layout(
        title='Wartości rzeczywiste vs. Przewidywane',
        xaxis_title="Wartości rzeczywiste",
        yaxis_title="Wartości przewidywane",
        height=500,
        showlegend=True,
        hovermode='closest',
        plot_bgcolor='white',
        xaxis=dict(showgrid=True, gridcolor='lightgray'),
        yaxis=dict(showgrid=True, gridcolor='lightgray'),
        font=dict(size=11)
    )
    
    st.plotly_chart(fig1, use_container_width=True)
    
    #with col2:
        # Wykres 2: Wykres reszt
    fig2 = go.Figure()
    
    fig2.add_trace(
        go.Scatter(
            x=st.session_state.y_pred,
            y=st.session_state.residuals,
            mode='markers',
            name='Reszty',
            marker=dict(
                size=8,
                color='darkorange',
                opacity=0.6,
                line=dict(width=0.5, color='darkred')
            ),
            hovertemplate='<b>Przewidywane:</b> %{x:.2f}<br><b>Reszta:</b> %{y:.2f}<extra></extra>'
        )
    )
    
    fig2.add_trace(
        go.Scatter(
            x=[st.session_state.y_pred.min(), st.session_state.y_pred.max()],
            y=[0, 0],
            mode='lines',
            name='Reszty = 0',
            line=dict(color='red', width=2, dash='dash'),
            hovertemplate='Reszty = 0<extra></extra>'
        )
    )
    
    fig2.update_layout(
        title='Wykres reszt',
        xaxis_title="Wartości przewidywane",
        yaxis_title="Reszty (błędy)",
        height=500,
        showlegend=True,
        hovermode='closest',
        plot_bgcolor='white',
        xaxis=dict(showgrid=True, gridcolor='lightgray'),
        yaxis=dict(showgrid=True, gridcolor='lightgray'),
        font=dict(size=11)
    )
    
    st.plotly_chart(fig2, use_container_width=True)

    # Wykres PCA
    if len(st.session_state.features) > 2:
        st.markdown("**Wizualizacja PCA z rozkładem reszt**")
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(st.session_state.X_test_scaled)

        fig_pca = go.Figure()
        
        fig_pca.add_trace(
            go.Scatter(
                x=X_pca[:, 0],
                y=X_pca[:, 1],
                mode='markers',
                marker=dict(
                    size=10,
                    color=st.session_state.residuals,
                    colorscale='RdYlGn_r',
                    showscale=True,
                    opacity=0.7,
                    line=dict(width=0.5, color='black'),
                    colorbar=dict(
                        title="Reszty",
                        thickness=15,
                        len=0.7
                    )
                ),
                hovertemplate='<b>PC1:</b> %{x:.2f}<br><b>PC2:</b> %{y:.2f}<br><b>Reszta:</b> %{marker.color:.2f}<extra></extra>'
            )
        )
        
        fig_pca.update_layout(
            title='Rozkład reszt w przestrzeni PCA',
            xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.1%} wariancji)',
            yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.1%} wariancji)',
            height=600,
            hovermode='closest',
            plot_bgcolor='white',
            xaxis=dict(showgrid=True, gridcolor='lightgray'),
            yaxis=dict(showgrid=True, gridcolor='lightgray'),
            font=dict(size=11)
        )
        
        st.plotly_chart(fig_pca, use_container_width=True)

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