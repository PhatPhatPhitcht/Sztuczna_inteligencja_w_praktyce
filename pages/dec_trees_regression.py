import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

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

df = st.session_state.df

st.markdown("""
Regresyjne drzewa decyzyjne są atrakcyjnymi modelami, jeśli zależy nam na interpretowalności przewidywań wartości ciągłych. Podobnie jak w klasyfikacji, możemy myśleć o tym modelu jako o metodzie dzielenia danych poprzez podejmowanie decyzji na podstawie zadawania serii pytań, z tą różnicą że końcowym wynikiem jest przewidywana wartość numeryczna, a nie klasa.

**Struktura Drzewa**

-**Korzeń** (Root Node): Najwyższy węzeł w drzewie, reprezentujący całą populację danych. To punkt startowy, od którego rozpoczyna się podział.

-**Węzły wewnętrzne** (Internal Nodes): Punkty decyzyjne w drzewie. Każdy węzeł wewnętrzny zawiera pytanie o konkretną cechę danych i ma co najmniej dwie gałęzie wychodzące, reprezentujące możliwe odpowiedzi.

-**Liście** (Leaf Nodes): Końcowe węzły drzewa, które nie mają dalszych podziałów. Każdy liść reprezentuje przewidywaną wartość numeryczną, będącą zazwyczaj średnią wartości ze zbioru treningowego, które trafiły do tego liścia.

-**Gałęzie** (Branches): Połączenia między węzłami, reprezentujące możliwe wartości cechy lub wynik testu warunkowego.

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

[Dowiedz się więcej](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)

[Wikipedia](https://en.wikipedia.org/wiki/Decision_tree_learning)

Poniżej znajdziesz możliwość przetestowania algorytmu. Wygeneruj swoje własne drzewo, zobacz jak zmienia się wraz z hiperparametrami oraz przeprowadź analizę nowej próbki.
""")

st.divider()

with st.expander("Podgląd danych"):
    st.dataframe(st.session_state.df.head())

target_column = st.selectbox("Wybierz kolumnę docelową (wartość do przewidzenia):", options=df.columns.tolist())

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
        
        # Normalizacja koloru na podstawie wartości
        nodes = [{
            'id': node_id,
            'x': x,
            'y': y,
            'info': node_info
        }]
        
        edges_x = []
        edges_y = []
        
        # Lewe dziecko
        left_child = tree.children_left[node_id]
        if left_child != -1:
            left_x = x - x_offset / (2 ** level)
            left_y = y - 0.15
            
            # Dodaj krawędź
            edges_x.extend([x, left_x, None])
            edges_y.extend([y, left_y, None])
            
            # Rekurencyjnie dodaj dzieci
            left_nodes, left_edges_x, left_edges_y = build_tree_layout(
                left_child, left_x, left_y, level + 1, x_offset
            )
            nodes.extend(left_nodes)
            edges_x.extend(left_edges_x)
            edges_y.extend(left_edges_y)
        
        # Prawe dziecko
        right_child = tree.children_right[node_id]
        if right_child != -1:
            right_x = x + x_offset / (2 ** level)
            right_y = y - 0.15
            
            # Dodaj krawędź
            edges_x.extend([x, right_x, None])
            edges_y.extend([y, right_y, None])
            
            # Rekurencyjnie dodaj dzieci
            right_nodes, right_edges_x, right_edges_y = build_tree_layout(
                right_child, right_x, right_y, level + 1, x_offset
            )
            nodes.extend(right_nodes)
            edges_x.extend(right_edges_x)
            edges_y.extend(right_edges_y)
        
        return nodes, edges_x, edges_y
    
    # Zbuduj layout drzewa
    nodes, edges_x, edges_y = build_tree_layout()
    
    # Normalizuj wartości dla kolorowania
    values = [n['info']['value'] for n in nodes]
    min_val, max_val = min(values), max(values)
    value_range = max_val - min_val if max_val > min_val else 1
    
    # Przygotuj dane dla węzłów
    node_x = [n['x'] for n in nodes]
    node_y = [n['y'] for n in nodes]
    node_colors = [(n['info']['value'] - min_val) / value_range for n in nodes]
    
    # Przygotuj teksty dla tooltipów
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
    
    # Utworzenie wykresu
    fig = go.Figure()
    
    # Dodaj krawędzie
    fig.add_trace(go.Scatter(
        x=edges_x,
        y=edges_y,
        mode='lines',
        line=dict(color='#888', width=2),
        hoverinfo='skip',
        showlegend=False
    ))
    
    # Dodaj węzły
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
    
    # Konfiguracja layoutu
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

if target_column:
    # Sprawdź czy kolumna jest numeryczna
    if not pd.api.types.is_numeric_dtype(df[target_column]):
        st.error("Kolumna docelowa musi być numeryczna dla regresji!")
        st.stop()
    
    feature_columns = [col for col in df.columns if col != target_column]
    numeric_features = df[feature_columns].select_dtypes(include=['number']).columns.tolist()

    if len(numeric_features) == 0:
        st.error("Brak kolumn numerycznych do użycia jako cechy! Drzewo decyzyjne wymaga cech numerycznych.")
        st.stop()

    st.markdown("**Parametry**")

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

    if st.button("Trenuj model", type="primary"):
        try:
            X = df[numeric_features]
            y = df[target_column]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )

            model = DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                min_samples_split=min_samples_split,
                random_state=42
            )

            with st.spinner("Zaczekaj aż model się wytrenuje..."):
                model.fit(X_train, y_train)

            # Oblicz metryki od razu po treningu
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            st.session_state.regression_tree_model = model
            st.session_state.regression_model_features = numeric_features
            st.session_state.regression_target_column = target_column
            st.session_state.regression_X_test = X_test
            st.session_state.regression_y_test = y_test
            st.session_state.regression_y_pred = y_pred
            st.session_state.regression_metrics = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            }

        except Exception as e:
            st.error(f"Błąd podczas treningu: {str(e)}")

    if 'regression_tree_model' in st.session_state:
        st.write("---")

        model = st.session_state.regression_tree_model
        features = st.session_state.regression_model_features
        target = st.session_state.regression_target_column

        st.markdown("**Interaktywne drzewo decyzyjne**")
        
        # Sprawdź rozmiar drzewa
        num_nodes = model.tree_.node_count
        if num_nodes > 100:
            st.warning(f"Drzewo ma {num_nodes} węzłów. Wizualizacja może być mniej czytelna dla bardzo dużych drzew. Rozważ zmniejszenie max_depth lub zwiększenie min_samples_leaf/min_samples_split.")
        
        # Utwórz interaktywną wizualizację
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
        y_test = st.session_state.regression_y_test
        y_pred = st.session_state.regression_y_pred
        metrics = st.session_state.regression_metrics

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("R² Score", f"{metrics['r2']:.4f}")
        with col2:
            st.metric("RMSE", f"{metrics['rmse']:.4f}")
        with col3:
            st.metric("MAE", f"{metrics['mae']:.4f}")
        with col4:
            st.metric("Głębokość drzewa", model.get_depth())

        st.markdown("""
        **Wyjaśnienie metryk:**
        - **R² Score**: Współczynnik determinacji (0-1), pokazuje jak dobrze model wyjaśnia wariancję danych. 1.0 = doskonałe dopasowanie.
        - **RMSE**: Root Mean Squared Error - pierwiastek ze średniego kwadratu błędów, w tych samych jednostkach co zmienna docelowa.
        - **MAE**: Mean Absolute Error - średni bezwzględny błąd predykcji.
        """)

        # Wykres rzeczywiste vs przewidywane
        st.markdown("**Wykres: Wartości rzeczywiste vs przewidywane**")
        X_train = df[features].drop(st.session_state.regression_X_test.index)
        y_train = df[target].drop(st.session_state.regression_y_test.index)
        y_train_pred = model.predict(X_train)
        
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.scatter(y_train, y_train_pred, alpha=0.5, label='Zbiór treningowy', color='orange')
        ax2.scatter(y_test, y_pred, alpha=0.5, label='Zbiór testowy', color='blue')
        ax2.plot([min(y_train.min(), y_test.min()), max(y_train.max(), y_test.max())], 
                 [min(y_train.min(), y_test.min()), max(y_train.max(), y_test.max())], 
                 'r--', lw=2, label='Idealna predykcja')
        ax2.set_xlabel('Wartości rzeczywiste')
        ax2.set_ylabel('Wartości przewidywane')
        ax2.set_title('Porównanie wartości rzeczywistych i przewidywanych')
        ax2.legend()
        st.pyplot(fig2)

        st.markdown("""
        Wykres pokazuje jak dobrze model przewiduje wartości w porównaniu z rzeczywistymi danymi ze zbioru testowego:
        
        - **Czerwona przerywana linia** to linia idealna (y = x), gdzie wartości przewidywane są dokładnie równe rzeczywistym
        - **Niebieskie punkty** to poszczególne próbki ze zbioru testowego
        - **Pomorańczowe punkty** to poszczególne próbki ze zbioru treningowego            
        - **Im bliżej czerwonej linii**, tym lepsze przewidywania modelu
        - **Punkty rozrzucone daleko od linii** oznaczają większe błędy predykcji
        
        **Przy małym max_depth (np. 1-2):**
        - Zobaczysz, że punkty układają się w poziome lub pionowe linie/grupy
        - Model ma bardzo ograniczoną zdolność do przewidywania - przewiduje tylko kilka różnych wartości
        - To oznacza **niedouczenie** (underfitting) - model jest zbyt prosty
        
        **Przy optymalnym max_depth:**
        - Punkty są blisko czerwonej linii, ale z naturalnym rozrzutem
        - Model dobrze uchwycił zależności w danych
        
        **Przy zbyt dużym max_depth:**
        - Punkty mogą być bardzo blisko linii dla danych treningowych, ale daleko dla testowych
        - To oznaka **przeuczenia** (overfitting) - model zapamiętał dane treningowe zamiast nauczyć się ogólnych wzorców
        """)

        st.subheader("Testowanie modelu")
        st.markdown("Wprowadź wartości dla nowej próbki:")
        
        input_data = {}
        
        # Inteligentny layout: max 4 kolumny, reszta w wierszach
        num_features = len(features)
        cols_per_row = min(4, num_features)
        
        for idx, feature in enumerate(features):
            # Co n-tą cechę twórz nowy wiersz
            if idx % cols_per_row == 0:
                cols = st.columns(cols_per_row)
            
            with cols[idx % cols_per_row]:
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
                    key=f"test_input_{feature}"
                )
        
        if st.button("Przewiduj wartość"):
            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)[0]
            st.markdown("**Wynik predykcji:**")
            st.success(f"{prediction:.4f}")