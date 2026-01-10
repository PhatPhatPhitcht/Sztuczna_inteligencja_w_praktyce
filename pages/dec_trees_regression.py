import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import plotly.graph_objects as go
from utils.file_loader import load_file_to_dataframe
from sklearn.datasets import fetch_openml

st.set_page_config(page_title="Interaktywne algorytmy uczenia maszynowego", initial_sidebar_state="collapsed")
st.markdown("""
    <style>
        [data-testid="stSidebarNav"] {display: none;}
    </style>
""", unsafe_allow_html=True)

st.page_link("main.py", label="⬅️ Powrót do strony głównej")

st.subheader("Drzewa Decyzyjne (Regresja)")

if 'df' not in st.session_state:
    st.error("Brak danych! Najpierw załaduj stronę główną")
    st.stop()

df = st.session_state.df.copy()

st.markdown("""
Regresyjne drzewa decyzyjne są atrakcyjnymi modelami, jeśli zależy nam na interpretowalności przewidywań wartości ciągłych. Podobnie jak w klasyfikacji, możemy myśleć o tym modelu jako o metodzie dzielenia danych poprzez podejmowanie decyzji na podstawie zadawania serii pytań, z tą różnicą że końcowym wynikiem jest przewidywana wartość numeryczna, a nie klasa.

**Struktura Drzewa**

- **Korzeń** (Root Node): Najwyższy węzeł w drzewie, reprezentujący całą populację danych. To punkt startowy, od którego rozpoczyna się podział.

- **Węzły wewnętrzne** (Internal Nodes): Punkty decyzyjne w drzewie. Każdy węzeł wewnętrzny zawiera pytanie o konkretną cechę danych i ma co najmniej dwie gałęzie wychodzące, reprezentujące możliwe odpowiedzi.

- **Liście** (Leaf Nodes): Końcowe węzły drzewa, które nie mają dalszych podziałów. Każdy liść reprezentuje przewidywaną wartość numeryczną, będącą zazwyczaj średnią wartości ze zbioru treningowego, które trafiły do tego liścia.

- **Gałęzie** (Branches): Połączenia między węzłami, reprezentujące możliwe wartości cechy lub wynik testu warunkowego.

**Działanie algorytmu**

Na podstawie cech z naszego zbioru treningowego model drzewa decyzyjnego uczy się serii pytań, aby przewidzieć wartości ciągłe. Zamiast maksymalizować przyrost informacji (jak w klasyfikacji), drzewo regresyjne minimalizuje błąd predykcji, najczęściej mierzony jako MSE (Mean Squared Error).

W każdym węźle algorytm wybiera taką cechę i wartość progową, która najlepiej dzieli dane, minimalizując wariancję wartości docelowych w powstałych podzbiorach. Proces ten jest powtarzany rekurencyjnie, aż zostaną spełnione warunki zatrzymania (maksymalna głębokość, minimalna liczba próbek w liściu, itp.).

**Kluczowe Parametry**

**max_depth**
- Ten parametr określa maksymalną dozwoloną głębokość drzewa. Gdy drzewo osiągnie tę głębokość, zatrzymuje dalsze dzielenie węzłów, nawet jeśli teoretycznie mogłoby kontynuować podział.

**min_samples_split**
- Określa minimalną liczbę próbek, które muszą znajdować się w węźle, aby algorytm w ogóle rozważył ich podział na mniejsze węzły.

**min_samples_leaf**
- Określa minimalną liczbę próbek, które muszą znajdować się w każdym końcowym węźle (liściu) po dokonaniu podziału.

**Przeuczenie**

Drzewo "zapamiętuje" dane treningowe zbyt dokładnie, włączając w to szum i przypadkowe fluktuacje. Doskonale przewiduje wartości dla danych treningowych, ale słabo radzi sobie z nowymi danymi, gdyż tworzy zbyt szczegółowe reguły nieogólne na nowe przypadki.

**Metody zapobiegania**

- Przycinanie wstępne (pre-pruning): Ograniczanie wartości parametrów max_depth, min_samples_split i min_samples_leaf.
- Przycinanie końcowe (post-pruning): Pozwalamy drzewu urosnąć do pełnego rozmiaru, a następnie usuwamy części, które nie poprawiają wydajności na zbiorze walidacyjnym.

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

[Dowiedz się więcej](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)

[Wikipedia](https://en.wikipedia.org/wiki/Decision_tree_learning)

Poniżej znajdziesz możliwość przetestowania algorytmu. Wygeneruj swoje własne drzewo, zobacz jak zmienia się wraz z hiperparametrami oraz przeprowadź analizę nowej próbki.
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
        load_file_to_dataframe(uploaded_file)

with st.expander("Podgląd danych"):
    st.dataframe(st.session_state.df.head())

# Inicjalizacja session_state z prefiksem dla drzewa decyzyjnego regresji
for key, default in {
    "dtr_model": None,
    "dtr_features": None,
    "dtr_target": None,
    "dtr_is_trained": False,
    "dtr_label_encoders": {},
    "dtr_y_pred": None,
    "dtr_y": None
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Wybór zmiennej docelowej
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if len(numeric_cols) < 1:
    st.error("Wymagana jest co najmniej 1 kolumna numeryczna jako zmienna docelowa.")
    st.stop()


# Expander do wyboru zmiennych niezależnych
with st.expander("Wybór zmiennych", expanded=False):
    st.markdown("""
    **Wybierz zmienne zależne (etykiety) i niezależne (cechy), które będą użyte do budowy drzewa decyzyjnego.**
    
    - Zmienne tekstowe (kategoryczne) zostaną automatycznie zakodowane numerycznie.
    - Wybierz cechy, które Twoim zdaniem mogą wpływać na wartość docelową.
    """)
    
    target_column = st.selectbox(
    "Wybierz zmienną docelową (Y) - wartość do przewidzenia",
    numeric_cols,
    index=numeric_cols.index(st.session_state.dtr_target)
    if st.session_state.dtr_target in numeric_cols else 0
    )

    available_features = [c for c in df.columns if c != target_column]
    
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
            corr_df = df[selected_features + [target_column]].copy()
            
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
            
            Uwaga: Drzewa decyzyjne mogą uchwycić nieliniowe zależności, których nie widać w macierzy korelacji.
            """)
        except Exception as e:
            st.error(f"Nie udało się wygenerować macierzy korelacji: {str(e)}")

st.markdown("**Parametry hiperparametrów drzewa**")

col1, col2, col3 = st.columns(3)

with col1:
    max_depth = st.slider(
        "max_depth",
        min_value=1,
        max_value=20,
        value=5,
    )

with col2:
    min_samples_leaf = st.slider(
        "min_samples_leaf",
        min_value=1,
        max_value=20,
        value=1,
    )

with col3:
    min_samples_split = st.slider(
        "min_samples_split",
        min_value=2,
        max_value=20,
        value=2,
    )

def create_interactive_tree(model, feature_names):
    """Tworzy interaktywną wizualizację drzewa decyzyjnego przy użyciu Plotly"""
    tree = model.tree_
    
    def get_node_info(node_id):
        """Zbiera informacje o węźle"""
        if tree.feature[node_id] != -2:  # Nie jest liściem
            feature = feature_names[tree.feature[node_id]]
            threshold = tree.threshold[node_id]
            decision = f"{feature} ≤ {threshold:.3f}"
            node_type = "Węzeł decyzyjny"
        else:
            decision = "LIŚĆ"
            node_type = "Węzeł końcowy"
        
        mse = tree.impurity[node_id]
        samples = tree.n_node_samples[node_id]
        value = tree.value[node_id][0][0]
        
        return {
            'decision': decision,
            'node_type': node_type,
            'mse': mse,
            'samples': samples,
            'value': value
        }
    
    def build_tree_layout(node_id=0, x=0.5, y=1.0, level=0, x_offset=0.25):
        """Buduje układ drzewa z pozycjami węzłów"""
        if node_id == -1:
            return [], [], []
        
        node_info = get_node_info(node_id)
        
        nodes = [{
            'id': node_id,
            'x': x,
            'y': y,
            'info': node_info
        }]
        
        edges_x = []
        edges_y = []
        
        left_child = tree.children_left[node_id]
        if left_child != -1:
            left_x = x - x_offset / (2 ** level)
            left_y = y - 0.15
            
            edges_x.extend([x, left_x, None])
            edges_y.extend([y, left_y, None])
            
            left_nodes, left_edges_x, left_edges_y = build_tree_layout(
                left_child, left_x, left_y, level + 1, x_offset
            )
            nodes.extend(left_nodes)
            edges_x.extend(left_edges_x)
            edges_y.extend(left_edges_y)
        
        right_child = tree.children_right[node_id]
        if right_child != -1:
            right_x = x + x_offset / (2 ** level)
            right_y = y - 0.15
            
            edges_x.extend([x, right_x, None])
            edges_y.extend([y, right_y, None])
            
            right_nodes, right_edges_x, right_edges_y = build_tree_layout(
                right_child, right_x, right_y, level + 1, x_offset
            )
            nodes.extend(right_nodes)
            edges_x.extend(right_edges_x)
            edges_y.extend(right_edges_y)
        
        return nodes, edges_x, edges_y
    
    nodes, edges_x, edges_y = build_tree_layout()
    
    values = [n['info']['value'] for n in nodes]
    min_val, max_val = min(values), max(values)
    value_range = max_val - min_val if max_val > min_val else 1
    
    node_x = [n['x'] for n in nodes]
    node_y = [n['y'] for n in nodes]
    node_colors = [(n['info']['value'] - min_val) / value_range for n in nodes]
    
    hover_texts = []
    for n in nodes:
        info = n['info']
        hover_text = (
            f"<b>{info['node_type']}</b><br>"
            f"<b>Decyzja:</b> {info['decision']}<br>"
            f"<b>MSE:</b> {info['mse']:.4f}<br>"
            f"<b>Próbki:</b> {info['samples']}<br>"
            f"<b>Wartość:</b> {info['value']:.4f}"
        )
        hover_texts.append(hover_text)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=edges_x,
        y=edges_y,
        mode='lines',
        line=dict(color='#888', width=2),
        hoverinfo='skip',
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers',
        marker=dict(
            size=30,
            color=node_colors,
            colorscale='RdYlBu_r',
            showscale=True,
            colorbar=dict(
                title="Wartość<br>predykcji",
                thickness=15,
                len=0.7
            ),
            line=dict(color='white', width=2)
        ),
        text=hover_texts,
        hoverinfo='text',
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        ),
        showlegend=False
    ))
    
    fig.update_layout(
        title={
            'text': "Interaktywna wizualizacja drzewa decyzyjnego<br><sub>Kliknij i przeciągnij aby przesunąć | Przewiń aby zoomować | Najedź na węzeł aby zobaczyć szczegóły</sub>",
            'x': 0.5,
            'xanchor': 'center'
        },
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=80),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white',
        height=max(600, len(nodes) * 15)
    )
    
    return fig

if st.button("Trenuj model", type="primary") and len(selected_features) > 0:
    try:
        X = df[selected_features].copy()
        y = df[target_column].copy()
        
        if not pd.api.types.is_numeric_dtype(y):
            st.error(f"Zmienna docelowa '{target_column}' musi być numeryczna dla regresji!")
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
        
        if X.isna().any().any():
            st.warning(f"Uzupełniono brakujące wartości w zmiennych niezależnych medianą.")
            X = X.fillna(X.median())
        
        model = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            random_state=42
        )

        with st.spinner("Zaczekaj aż model się wytrenuje..."):
            model.fit(X, y)

        y_pred = model.predict(X)
        
        st.session_state.dtr_model = model
        st.session_state.dtr_features = selected_features
        st.session_state.dtr_target = target_column
        st.session_state.dtr_is_trained = True
        st.session_state.dtr_label_encoders = label_encoders
        st.session_state.dtr_y_pred = y_pred
        st.session_state.dtr_y = y
        
        st.session_state.dtr_metrics = {
            'mse': mean_squared_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred)
        }
        

    except Exception as e:
        st.error(f"Wystąpił błąd podczas trenowania modelu: {str(e)}")
        st.session_state.dtr_is_trained = False

if st.session_state.dtr_is_trained:
    st.write("---")

    model = st.session_state.dtr_model
    features = st.session_state.dtr_features
    target = st.session_state.dtr_target

    st.markdown("**Interaktywne drzewo decyzyjne**")
    
    num_nodes = model.tree_.node_count
    if num_nodes > 100:
        st.warning(f"Drzewo ma {num_nodes} węzłów. Wizualizacja może być mniej czytelna dla bardzo dużych drzew. Rozważ zmniejszenie max_depth lub zwiększenie min_samples_leaf/min_samples_split.")
    
    fig = create_interactive_tree(model, features)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Wyjaśnienie metryk w węzłach:**

    **MSE** (Mean Squared Error) to średni kwadrat błędu w węźle, który pokazuje jak bardzo wartości w tym węźle różnią się od ich średniej. Wartość 0 oznacza węzeł idealny (wszystkie próbki mają tę samą wartość), a wyższe wartości oznaczają większą zmienność przewidywań.

    **Próbki** (Samples) to liczba próbek treningowych, które trafiły do tego węzła, przydatne do wyobrażenia sobie wpływu parametrów *min_samples_split/leaf*.

    **Wartość** (Value) to przewidywana wartość w tym węźle, obliczana jako średnia wartości docelowych wszystkich próbek, które trafiły do tego węzła. W liściach jest to finalna predykcja modelu.

    Zmieniające się kolory węzłów reprezentują przewidywaną wartość - różne odcienie na skali kolorów pokazują jak zmienia się predykcja w różnych częściach drzewa. Wraz ze zbliżaniem się do liści, kolory stają się bardziej jednolite, gdyż węzły zawierają bardziej jednorodne wartości.
    """)

    st.markdown("**Metryki modelu**")
    metrics = st.session_state.dtr_metrics

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("R²", f"{metrics['r2']:.4f}")
    with col2:
        st.metric("RMSE", f"{metrics['rmse']:.4f}")
    with col3:
        st.metric("MAE", f"{metrics['mae']:.4f}")
    with col4:
        st.metric("Głębokość drzewa", model.get_depth())

    # Wykres rzeczywiste vs przewidywane
    st.markdown("**Wykres: Wartości rzeczywiste vs przewidywane**")
    
    y = st.session_state.dtr_y
    y_pred = st.session_state.dtr_y_pred
    
    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x=y,
        y=y_pred,
        mode='markers',
        name='Predykcje',
        marker=dict(color='steelblue', size=8, opacity=0.6),
        hovertemplate='<b>Rzeczywiste:</b> %{x:.2f}<br><b>Przewidywane:</b> %{y:.2f}<extra></extra>'
    ))

    min_val = min(y.min(), y_pred.min())
    max_val = max(y.max(), y_pred.max())
    fig2.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Idealna predykcja',
        line=dict(color='red', width=2, dash='dash')
    ))

    fig2.update_layout(
        title='Porównanie wartości rzeczywistych i przewidywanych',
        xaxis_title='Wartości rzeczywiste',
        yaxis_title='Wartości przewidywane',
        hovermode='closest',
        showlegend=True,
        height=600
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("""
    Wykres pokazuje jak dobrze model przewiduje wartości w porównaniu z rzeczywistymi danymi:
    
    - **Czerwona przerywana linia** to linia idealna (y = x), gdzie wartości przewidywane są dokładnie równe rzeczywistym
    - **Niebieskie punkty** to poszczególne próbki ze zbioru danych
    - **Im bliżej czerwonej linii**, tym lepsze przewidywania modelu
    - **Punkty rozrzucone daleko od linii** oznaczają większe błędy predykcji
    
    **Przy małym max_depth (np. 1-2):**
    - Zobaczysz, że punkty układają się w poziome lub pionowe linie/grupy
    - Model ma bardzo ograniczoną zdolność do przewidywania - przewiduje tylko kilka różnych wartości
    - To oznacza **niedouczenie** (underfitting) - model jest zbyt prosty
    
    **Przy optymalnym max_depth:**
    - Punkty są blisko czerwonej linii z naturalnym rozrzutem
    - Model dobrze uchwycił zależności w danych
    
    **Przy zbyt dużym max_depth:**
    - Punkty mogą być bardzo blisko linii (model "zapamiętał" dane)
    - To oznaka **przeuczenia** (overfitting) - model może słabo działać na nowych danych
    """)

    st.subheader("Testowanie modelu")
    st.markdown("Wprowadź wartości dla nowej próbki:")
    
    input_data = {}
    
    num_features = len(features)
    cols_per_row = min(4, num_features)
    
    for idx, feature in enumerate(features):
        if idx % cols_per_row == 0:
            cols = st.columns(cols_per_row)
        
        with cols[idx % cols_per_row]:
            if feature in st.session_state.dtr_label_encoders:
                le = st.session_state.dtr_label_encoders[feature]
                options = list(le.classes_)
                input_data[feature] = st.selectbox(
                    feature,
                    options,
                    key=f"dtr_input_{feature}"
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
                    key=f"dtr_input_{feature}"
                )
    
    if st.button("Przewiduj wartość", type="primary"):
        try:
            input_df = pd.DataFrame([input_data])
            
            for col in input_df.columns:
                if col in st.session_state.dtr_label_encoders:
                    le = st.session_state.dtr_label_encoders[col]
                    input_df[col] = le.transform(input_df[col].astype(str))
            
            prediction = model.predict(input_df)[0]
            
            st.markdown("**Wynik predykcji:**")
            st.success(f"Przewidywana wartość dla **{target}**: **{prediction:.4f}**")
        except Exception as e:
            st.error(f"Błąd podczas predykcji: {str(e)}")