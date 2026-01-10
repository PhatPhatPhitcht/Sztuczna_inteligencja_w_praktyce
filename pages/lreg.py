import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from utils.file_loader import load_file_to_dataframe
from sklearn.datasets import fetch_openml

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

**Równanie modelu:**

$$y = \\beta_0 + \\beta_1 x_1 + \\beta_2 x_2 + ... + \\beta_n x_n + \\epsilon$$

gdzie:
- $y$ - zmienna zależna (target)
- $x_1, x_2, ..., x_n$ - zmienne niezależne (features)
- $\\beta_0$ - wyraz wolny (intercept)
- $\\beta_1, ..., \\beta_n$ - współczynniki regresji
- $\\epsilon$ - błąd losowy

**Metryki jakości modelu:**

- **R² (Coefficient of Determination)** - Współczynnik determinacji
  
  $$R^2 = 1 - \\frac{\\sum_{i=1}^{n}(y_i - \\hat{y}_i)^2}{\\sum_{i=1}^{n}(y_i - \\bar{y})^2}$$
  
  Pokazuje, jaki procent zmienności zmiennej zależnej jest wyjaśniony przez model. Wartości bliskie 1 oznaczają dobre dopasowanie.

- **MSE (Mean Squared Error)** - Średni błąd kwadratowy
  
  $$MSE = \\frac{1}{n}\\sum_{i=1}^{n}(y_i - \\hat{y}_i)^2$$
  
  Średnia kwadratów różnic między wartościami rzeczywistymi a przewidywanymi. Penalizuje duże błędy.

- **RMSE (Root Mean Squared Error)** - Pierwiastek ze średniego błędu kwadratowego
  
  $$RMSE = \\sqrt{MSE} = \\sqrt{\\frac{1}{n}\\sum_{i=1}^{n}(y_i - \\hat{y}_i)^2}$$
  
  Jest w tych samych jednostkach co zmienna zależna, co ułatwia interpretację.

- **MAE (Mean Absolute Error)** - Średni błąd bezwzględny
  
  $$MAE = \\frac{1}{n}\\sum_{i=1}^{n}|y_i - \\hat{y}_i|$$
  
  Średnia wartość bezwzględna różnic między wartościami rzeczywistymi a przewidywanymi.

**R²** powinno dążyć do 1, pozostałe metryki do 0.
            
[Dowiedz się więcej](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
            
*Poniżej możesz wytrenować model dla twoich danych i zobaczyć jak dobrze regresja liniowa się sprawdza do problemu!*
""")

for key, default in {
    "lr_model": None,
    "lr_scaler": None,
    "lr_features": None,
    "lr_target": None,
    "lr_is_trained": False,
    "lr_use_pca": False,
    "lr_pca": None,
    "lr_label_encoders": {}
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


if "df" not in st.session_state:
    st.error("Brak danych w st.session_state.df")
    st.stop()

df = st.session_state.df.copy()
#-----------------------------------------------------Początek-----------------------------------------------------------
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
        load_file_to_dataframe(uploaded_file)

with st.expander("Podgląd danych"):
    st.dataframe(df.head())

# Wybór zmiennej docelowej
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if len(numeric_cols) < 1:
    st.error("Wymagana jest co najmniej 1 kolumna numeryczna jako zmienna docelowa.")
    st.stop()



# Expander do wyboru zmiennych niezależnych
with st.expander("Wybór zmiennych", expanded=False):
    st.markdown("""
    **Wybierz zmienne , które będą użyte do predykcji.**
    
    - Jeśli wybierzesz **więcej niż 2 zmienne niezależne**, zastosowana zostanie **PCA (Principal Component Analysis)** do redukcji wymiarowości dla wizualizacji.
    - Zmienne tekstowe (kategoryczne) zostaną automatycznie zakodowane numerycznie.
    """)

    target = st.selectbox(
    "Wybierz zmienną docelową (Y)",
    numeric_cols,
    index=numeric_cols.index(st.session_state.lr_target)
    if st.session_state.lr_target in numeric_cols else 0
    )
    
    # Wszystkie kolumny oprócz target
    available_features = [c for c in df.columns if c != target]
    
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
            # Przygotowanie danych do korelacji
            corr_df = df[selected_features + [target]].copy()
            
            # Kodowanie zmiennych tekstowych dla korelacji
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

if st.button("Trenuj model", type="primary") and len(selected_features) > 0:
    try:
        # Przygotowanie danych
        X = df[selected_features].copy()
        y = df[target].copy()
        
        # Sprawdzenie czy target jest numeryczny
        if not pd.api.types.is_numeric_dtype(y):
            st.error(f"Zmienna docelowa '{target}' musi być numeryczna dla regresji liniowej.")
            st.stop()
        
        # Obsługa brakujących wartości w target
        if y.isna().any():
            st.warning(f"Usunięto {y.isna().sum()} wierszy z brakującymi wartościami w zmiennej docelowej.")
            valid_indices = ~y.isna()
            X = X[valid_indices]
            y = y[valid_indices]
        
        # Kodowanie zmiennych tekstowych
        label_encoders = {}
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                label_encoders[col] = le
        
        # Obsługa brakujących wartości w features
        if X.isna().any().any():
            st.warning(f"Uzupełniono brakujące wartości w zmiennych niezależnych medianą.")
            X = X.fillna(X.median())
        
        # Skalowanie
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # PCA jeśli więcej niż 2 zmienne
        use_pca = len(selected_features) > 2
        pca = None
        
        if use_pca:
            pca = PCA(n_components=2)
            X_for_viz = pca.fit_transform(X_scaled)
        else:
            X_for_viz = X_scaled
        
        # Trenowanie modelu
        model = LinearRegression()
        model.fit(X_scaled, y)
        
        # Predykcje
        y_pred = model.predict(X_scaled)
        residuals = y - y_pred
        
        # Zapisanie w session_state
        st.session_state.lr_model = model
        st.session_state.lr_scaler = scaler
        st.session_state.lr_features = selected_features
        st.session_state.lr_target = target
        st.session_state.lr_is_trained = True
        st.session_state.lr_use_pca = use_pca
        st.session_state.lr_pca = pca
        st.session_state.lr_label_encoders = label_encoders
        
        # Metryki
        st.session_state.lr_metrics = {
            "R2": r2_score(y, y_pred),
            "MSE": mean_squared_error(y, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y, y_pred)),
            "MAE": mean_absolute_error(y, y_pred)
        }
        
        st.session_state.lr_residuals = residuals
        st.session_state.lr_y_pred = y_pred
        st.session_state.lr_y = y
        st.session_state.lr_X_for_viz = X_for_viz
        
        
    except Exception as e:
        st.error(f"Wystąpił błąd podczas trenowania modelu: {str(e)}")
        st.session_state.lr_is_trained = False


if st.session_state.lr_is_trained:
    st.markdown("**Metryki modelu**")
    
    m = st.session_state.lr_metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R²", f"{m['R2']:.4f}")
    c2.metric("MSE", f"{m['MSE']:.4f}")
    c3.metric("RMSE", f"{m['RMSE']:.4f}")
    c4.metric("MAE", f"{m['MAE']:.4f}")

    st.markdown("**Wykresy diagnostyczne**")
    
    # Wykres 1: Wartości rzeczywiste vs. przewidywane
    fig1 = go.Figure()
    
    fig1.add_trace(
        go.Scatter(
            x=st.session_state.lr_y,
            y=st.session_state.lr_y_pred,
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
    
    min_val = min(st.session_state.lr_y.min(), st.session_state.lr_y_pred.min())
    max_val = max(st.session_state.lr_y.max(), st.session_state.lr_y_pred.max())
    
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
    
    # Wykres 2: Wykres reszt
    fig2 = go.Figure()
    
    fig2.add_trace(
        go.Scatter(
            x=st.session_state.lr_y_pred,
            y=st.session_state.lr_residuals,
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
            x=[st.session_state.lr_y_pred.min(), st.session_state.lr_y_pred.max()],
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

    # Wizualizacja w przestrzeni zmiennych/PCA
    st.markdown("**Wizualizacja przestrzeni zmiennych z rozkładem reszt**")
    
    X_viz = st.session_state.lr_X_for_viz
    
    if st.session_state.lr_use_pca:
        # Użyto PCA
        pca = st.session_state.lr_pca
        var_ratio = pca.explained_variance_ratio_
        
        xlabel = f'PC1 ({var_ratio[0]:.1%} wariancji)'
        ylabel = f'PC2 ({var_ratio[1]:.1%} wariancji)'
        title = 'Rozkład reszt w przestrzeni PCA'
        
        explanation = f"""
        **Zastosowano PCA (Principal Component Analysis)**
        
        Ponieważ wybrano {len(st.session_state.lr_features)} zmiennych niezależnych, użyto PCA do redukcji wymiarowości 
        dla celów wizualizacji. Dane zostały zredukowane do 2 komponentów głównych:
        
        - **PC1** wyjaśnia {var_ratio[0]:.1%} całkowitej wariancji danych
        - **PC2** wyjaśnia {var_ratio[1]:.1%} całkowitej wariancji danych
        - Łącznie: {sum(var_ratio):.1%} wariancji
        
        PCA nie wpływa na model regresji – służy tylko do wizualizacji przestrzeni wielowymiarowej na płaszczyźnie 2D.
        """
    else:
        # Bez PCA
        if len(st.session_state.lr_features) == 1:
            xlabel = st.session_state.lr_features[0]
            ylabel = 'Index'
            title = f'Rozkład reszt względem {st.session_state.lr_features[0]}'
            explanation = f"""
            **Wizualizacja dla pojedynczej zmiennej niezależnej**
            
            Wykres pokazuje wartości zmiennej **{st.session_state.lr_features[0]}** oraz rozkład reszt (błędów predykcji).
            """
        else:  # 2 zmienne
            xlabel = st.session_state.lr_features[0]
            ylabel = st.session_state.lr_features[1]
            title = f'Rozkład reszt w przestrzeni ({st.session_state.lr_features[0]}, {st.session_state.lr_features[1]})'
            explanation = f"""
            **Wizualizacja dla dwóch zmiennych niezależnych**
            
            Wykres pokazuje przestrzeń dwuwymiarową zdefiniowaną przez:
            - Oś X: **{st.session_state.lr_features[0]}**
            - Oś Y: **{st.session_state.lr_features[1]}**
            """
    
    fig_viz = go.Figure()
    
    if len(st.session_state.lr_features) == 1:
        # Dla 1 zmiennej: wykres punktowy z indeksem
        fig_viz.add_trace(
            go.Scatter(
                x=X_viz[:, 0],
                y=np.arange(len(X_viz)),
                mode='markers',
                marker=dict(
                    size=10,
                    color=st.session_state.lr_residuals,
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
                hovertemplate=f'<b>{xlabel}:</b> %{{x:.2f}}<br><b>Reszta:</b> %{{marker.color:.2f}}<extra></extra>'
            )
        )
    else:
        # Dla 2+ zmiennych
        fig_viz.add_trace(
            go.Scatter(
                x=X_viz[:, 0],
                y=X_viz[:, 1],
                mode='markers',
                marker=dict(
                    size=10,
                    color=st.session_state.lr_residuals,
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
                hovertemplate=f'<b>{xlabel.split("(")[0].strip()}:</b> %{{x:.2f}}<br><b>{ylabel.split("(")[0].strip()}:</b> %{{y:.2f}}<br><b>Reszta:</b> %{{marker.color:.2f}}<extra></extra>'
            )
        )
    
    fig_viz.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        height=600,
        hovermode='closest',
        plot_bgcolor='white',
        xaxis=dict(showgrid=True, gridcolor='lightgray'),
        yaxis=dict(showgrid=True, gridcolor='lightgray'),
        font=dict(size=11)
    )
    
    st.plotly_chart(fig_viz, use_container_width=True)
    
    st.markdown(explanation)
    
    st.markdown("""
    **Interpretacja kolorów:**
    
    - **Zielone** - małe błędy (model dobrze przewidział)
    - **Żółte** - średnie błędy
    - **Czerwone** - duże błędy (model się pomylił)
    
    **Analiza rozkładu:**
    
    - Jeśli błędy są **rozłożone losowo** → model jest poprawny
    - Jeśli **czerwone punkty tworzą klaster** → model ma problem z pewnym typem danych
    - Jeśli błędy **tworzą wzór** → model nie uchwycił jakiejś zależności (możliwa nielinearność)
    """)

    st.markdown("**Testowanie modelu**")

    input_data = {}
    num_features = len(st.session_state.lr_features)
    cols_per_row = min(4, num_features)
    
    for i, feature in enumerate(st.session_state.lr_features):
        if i % cols_per_row == 0:
            cols = st.columns(cols_per_row)
        
        with cols[i % cols_per_row]:
            # Sprawdzenie czy zmienna była kodowana
            if feature in st.session_state.lr_label_encoders:
                le = st.session_state.lr_label_encoders[feature]
                options = list(le.classes_)
                input_data[feature] = st.selectbox(
                    feature,
                    options,
                    key=f"lr_input_{feature}"
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
                    key=f"lr_input_{feature}"
                )

    if st.button("Przewiduj wartość", type="primary"):
        try:
            input_df = pd.DataFrame([input_data])
            
            # Kodowanie zmiennych tekstowych
            for col in input_df.columns:
                if col in st.session_state.lr_label_encoders:
                    le = st.session_state.lr_label_encoders[col]
                    input_df[col] = le.transform(input_df[col].astype(str))
            
            input_scaled = st.session_state.lr_scaler.transform(input_df)
            prediction = st.session_state.lr_model.predict(input_scaled)[0]

            st.success(f"Przewidywana wartość dla **{st.session_state.lr_target}**: **{prediction:.4f}**")
        except Exception as e:
            st.error(f"Błąd podczas predykcji: {str(e)}")