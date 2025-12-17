import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="Interaktywne algorytmy uczenia maszynowego", initial_sidebar_state="collapsed")
st.markdown("""
    <style>
        [data-testid="stSidebarNav"] {display: none;}
    </style>
""", unsafe_allow_html=True)

st.page_link("main.py", label="⬅️ Powrót do strony głównej")

st.subheader("Drzewa Decyzyjne (Klasyfikacja)")

if 'df' not in st.session_state:
    st.error("Brak danych! Najpierw załaduj stronę główną")
    st.stop()

df = st.session_state.df

st.markdown("""
Klasyfikatory drzew decyzyjnych są atrakcyjnymi modelami, jeśli zależy nam na interpretowalności. Jak sugeruje nazwa „drzewo decyzyjne", możemy myśleć o tym modelu jako o metodzie dzielenia danych poprzez podejmowanie decyzji na podstawie zadawania serii pytań.          

**Struktura Drzewa**
            

-**Korzeń** (Root Node): Najwyższy węzeł w drzewie, reprezentujący całą populację danych. To punkt startowy, od którego rozpoczyna się podział.                        


-**Węzły wewnętrzne** (Internal Nodes): Punkty decyzyjne w drzewie. Każdy węzeł wewnętrzny zawiera pytanie o konkretną cechę danych i ma co najmniej dwie gałęzie wychodzące, reprezentujące możliwe odpowiedzi.


-**Liście** (Leaf Nodes): Końcowe węzły drzewa, które nie mają dalszych podziałów. Każdy liść reprezentuje konkretną klasę lub kategorię, do której zostały przypisane dane.            


-**Gałęzie** (Branches): Połączenia między węzłami, reprezentujące możliwe wartości cechy lub wynik testu warunkowego.

**Działanie algorytmu**

Na podstawie cech z naszego zbioru treningowego model drzewa decyzyjnego uczy się serii pytań, aby wywnioskować etykiety klas przykładów. 
Korzystając z algorytmu decyzyjnego, zaczynamy od korzenia drzewa i dzielimy dane według tej cechy, która daje największy przyrost informacji. W procesie iteracyjnym możemy następnie powtarzać tę procedurę podziału w każdym węźle potomnym, aż liście staną się czyste. Oznacza to, że przykłady treningowe w każdym węźle należą do tej samej klasy. W praktyce może to prowadzić do powstania bardzo głębokiego drzewa z wieloma węzłami, co może łatwo skutkować przeuczeniem. Dlatego zazwyczaj chcemy przyciąć drzewo, ustawiając ograniczenie na maksymalną głębokość drzewa.                        

**Kluczowe Parametry**            

**max_depth**
- Ten parametr określa maksymalną dozwoloną głębokość drzewa. Gdy drzewo osiągnie tę głębokość, zatrzymuje dalsze dzielenie węzłów, nawet jeśli teoretycznie mogłoby kontynuować podział.

**min_samples_split**
- Określa minimalną liczbę próbek, które muszą znajdować się w węźle, aby algorytm w ogóle rozważył ich podział na mniejsze węzły.

**min_samples_leaf**
- Określa minimalną liczbę próbek, które muszą znajdować się w każdym końcowym węźle (liściu) po dokonaniu podziału.

**Przeuczenie**

Drzewo "zapamiętuje" dane treningowe zbyt dokładnie, włączając w to szum i przypadkowe wzorce. Doskonale klasyfikuje dane treningowe, ale słabo radzi sobie z nowymi danymi.

**Metody zapobiegania**

- Przycinanie wstępne (pre-pruning): Ograniczanie wartości parametrów max_depth, min_samples_split i min_samples_leaf.
- Przycinanie końcowe (post-pruning): Pozwalamy drzewu urosnąć do pełnego rozmiaru, a następnie usuwamy części, które nie poprawiają wydajności na zbiorze walidacyjnym.


[Dowiedz się więcej](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier)

[Wikipedia](https://en.wikipedia.org/wiki/Decision_tree_learning)  

Poniżej znajdziesz możliwość przetestowania algorytmu. Wygeneruj swoje własne drzewo, zobacz jak zmienia się wraz z hiperparametrami oraz przeprowadź analizę nowej próbki.

""")

st.divider()

with st.expander("Podgląd danych"):
    st.dataframe(st.session_state.df.head())

target_column = st.selectbox("Wybierz kolumnę docelową (cel klasyfikacji):", options=df.columns.tolist())
if target_column:
    unique_classes = df[target_column].nunique()
    
    if unique_classes < 2:
        st.error("Kolumna docelowa musi mieć co najmniej 2 klasy!")
        st.stop()
    elif unique_classes > 10:
        st.warning(f"Kolumna ma {unique_classes} klas. Dla lepszej wizualizacji zaleca się mniej klas.")

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

            model = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                min_samples_split=min_samples_split,
                random_state=42
            )

            with st.spinner("Zaczekaj aż model się wytrenuje..."):
                model.fit(X_train, y_train)
                
                cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

            st.session_state.decision_tree_model = model
            st.session_state.model_features = numeric_features
            st.session_state.target_column = target_column
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.X_train = X_train
            st.session_state.y_train = y_train
            st.session_state.cv_scores = cv_scores

            # Metryki
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            st.success(f"Model wytrenowany! Dokładność na zbiorze testowym: {accuracy:.2%}")

        except Exception as e:
            st.error(f"Błąd podczas treningu: {str(e)}")

    if 'decision_tree_model' in st.session_state:
        st.write("---")

        model = st.session_state.decision_tree_model
        features = st.session_state.model_features
        target = st.session_state.target_column
        
        # Drzewa w Plotly
        def create_plotly_tree(model, feature_names, class_names):
            tree = model.tree_
            
            def recurse(node, x, y, dx, parent_x=None, parent_y=None, edges_x=[], edges_y=[], nodes_x=[], nodes_y=[], node_texts=[], node_colors=[], node_ids=[]):
                if node < 0:
                    return
                
                # Informacje o węźle
                n_samples = tree.n_node_samples[node]
                gini = tree.impurity[node]
                value = tree.value[node][0]
                
                # Dominująca klasa
                class_idx = np.argmax(value)
                class_name = class_names[class_idx]
                
                # Kolor węzła
                color_intensity = 1 - gini
                colors_map = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22', '#34495e']
                base_color = colors_map[class_idx % len(colors_map)]
                
                # Tekst węzła
                if tree.feature[node] != -2:  # Nie jest liściem
                    feature = feature_names[tree.feature[node]]
                    threshold = tree.threshold[node]
                    node_text = f"<b>{feature} <= {threshold:.2f}</b><br>gini: {gini:.3f}<br>samples: {n_samples}<br>value: {value.astype(int).tolist()}<br>class: {class_name}"
                else:  # Liść
                    node_text = f"<b>LIŚĆ</b><br>gini: {gini:.3f}<br>samples: {n_samples}<br>value: {value.astype(int).tolist()}<br>class: {class_name}"
                
                nodes_x.append(x)
                nodes_y.append(y)
                node_texts.append(node_text)
                node_colors.append(base_color)
                node_ids.append(node)
                
                # Krawędź od rodzica
                if parent_x is not None:
                    edges_x.extend([parent_x, x, None])
                    edges_y.extend([parent_y, y, None])
                
                # Dzieci
                left_child = tree.children_left[node]
                right_child = tree.children_right[node]
                
                if left_child != -1:
                    recurse(left_child, x - dx, y - 1, dx / 2, x, y, edges_x, edges_y, nodes_x, nodes_y, node_texts, node_colors, node_ids)
                if right_child != -1:
                    recurse(right_child, x + dx, y - 1, dx / 2, x, y, edges_x, edges_y, nodes_x, nodes_y, node_texts, node_colors, node_ids)
                
                return edges_x, edges_y, nodes_x, nodes_y, node_texts, node_colors, node_ids
            
            edges_x, edges_y, nodes_x, nodes_y, node_texts, node_colors, node_ids = recurse(0, 0, 0, 4)
            
            # Krawędzie
            edge_trace = go.Scatter(
                x=edges_x, y=edges_y,
                mode='lines',
                line=dict(color='#7f8c8d', width=2),
                hoverinfo='none',
                showlegend=False
            )
            
            # Węzły
            node_trace = go.Scatter(
                x=nodes_x, y=nodes_y,
                mode='markers+text',
                marker=dict(
                    size=30,
                    color=node_colors,
                    line=dict(color='white', width=2),
                    opacity=0.8
                ),
                text=[f"N{nid}" for nid in node_ids],
                textposition="middle center",
                textfont=dict(color='white', size=10, family='Arial Black'),
                hovertext=node_texts,
                hoverinfo='text',
                showlegend=False
            )
            
            fig = go.Figure(data=[edge_trace, node_trace])
            fig.update_layout(
                title={
            'text': "Interaktywna wizualizacja drzewa decyzyjnego<br><sub>Kliknij i przeciągnij aby przesunąć | Przewiń aby zoomować | Najedź na węzeł aby zobaczyć szczegóły</sub>",
            'x': 0.5,
            'xanchor': 'center'
        },
                showlegend=False,
                hovermode='closest',
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='white',
                height=600,
                margin=dict(l=20, r=20, t=60, b=20)
            )
            
            return fig
        
        st.markdown("**Interaktywne Drzewo Decyzyjne**")
        st.markdown("Przesuń, zoomuj i najedź na węzły, aby zobaczyć szczegóły")
        
        fig_plotly = create_plotly_tree(model, features, [str(c) for c in model.classes_])
        st.plotly_chart(fig_plotly, use_container_width=True)

        st.markdown("""
        Wyjaśnijmy metryki które możesz zobaczyć w węzłach:

        **Gini** to  współczynnik Giniego, czyli miara zanieczyszczenia (impurity) węzła, która pokazuje, czy w danym węźle drzewa znajdują się próbki należące do różnych klas jednocześnie . Dla wartość  0 oznacza węzeł czysty (wszystkie próbki należą do jednej klasy), a Gini maksymalne (np. 0.5 dla 2 klas) oznacza że nie ma jednoznacznej odpowiedzi, którą klasę ten węzeł reprezentuje.

        **Samples** to liczba próbek treningowych, które trafiły do tego węzła, przydatne do wyobrażenia sobie wpływu parametrów *min_samples_split/leaf*.           

        **Value** to tablica pokazująca rozkład klas w węźle, czyli ile próbek z każdej klasy znajduje się w tym węźle.

        Zmieniające się kolory węzłów oznaczają dominującą klasę w tym węźle i jak jest czysta - połączenie gini i value, dlatego wraz ze zbliżaniem się do liścia zauważymy zwiększoną intensywność koloru wraz z zwiększaniem się pewności algorytmu.
        """)
        
        # Metryki modelu
        st.markdown("---")
        st.markdown("**Metryki Modelu**")
        
        y_test = st.session_state.y_test
        y_pred = model.predict(st.session_state.X_test)
        
        # Accuracy i walidacja krzyżowa
        accuracy = accuracy_score(y_test, y_pred)
        cv_scores = st.session_state.cv_scores
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Dokładność (test)", f"{accuracy:.2%}")
        with col2:
            st.metric("Walidacja krzyżowa (średnia)", f"{cv_scores.mean():.2%}")
        with col3:
            st.metric("Odchylenie std. (CV)", f"{cv_scores.std():.3f}")
        
        st.markdown(f"**Wyniki walidacji krzyżowej (5-fold):** {', '.join([f'{score:.2%}' for score in cv_scores])}")
        st.markdown("Walidacja krzyżowa pokazuje, jak stabilny jest model na różnych podzbiorach danych. Niskie odchylenie standardowe oznacza, że model jest stabilny.")
        
        # Macierz pomyłek
        st.markdown("**Macierz Pomyłek**")
        cm = confusion_matrix(y_test, y_pred)
        
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=[str(c) for c in model.classes_],
            y=[str(c) for c in model.classes_],
            colorscale='Blues',
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 16},
            hovertemplate='Prawdziwa: %{y}<br>Predykcja: %{x}<br>Liczba: %{z}<extra></extra>'
        ))
        
        fig_cm.update_layout(
            title="Macierz pomyłek",
            xaxis_title="Predykcja",
            yaxis_title="Rzeczywista wartość",
            height=400
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        
        # Metryki dla każdej klasy
        st.markdown("**Metryki dla każdej klasy**")
        precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred)
        
        metrics_df = pd.DataFrame({
            'Klasa': [str(c) for c in model.classes_],
            'Precision': [f"{p:.2%}" for p in precision],
            'Recall': [f"{r:.2%}" for r in recall],
            'F1-Score': [f"{f:.2%}" for f in f1],
            'Liczba próbek': support
        })
        
        st.dataframe(metrics_df, hide_index=True, use_container_width=True)
        
        st.markdown("""
        **Wyjaśnienie metryk:**
        - **Precision** (precyzja): Jaki odsetek próbek zaklasyfikowanych jako dana klasa faktycznie do niej należy
        - **Recall** (czułość): Jaki odsetek próbek danej klasy został poprawnie zidentyfikowany
        - **F1-Score**: Średnia harmoniczna precision i recall - ogólna miara jakości dla każdej klasy
        """)
#---------------------------------------Testowanie-------------------------------------------
        st.subheader("Testowanie modelu")
        st.markdown("Wprowadź wartości dla nowej próbki:")

        input_data = {}
        num_features = len(features)
        cols_per_row = min(4, num_features)  # Maksymalnie 4 kolumny w rzędzie
        
        for i, feature in enumerate(features):
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
                    key=f"test_input_{feature}"
                )

        if st.button("Klasyfikuj próbkę"):
            input_df = pd.DataFrame([input_data])

            prediction = model.predict(input_df)[0]
            prediction_proba = model.predict_proba(input_df)[0]

            st.markdown("**Wynik predykcji:**")
            st.success(f"Klasa: {prediction}")
            
            # Ścieżka decyzyjna
            st.markdown("**Ścieżka decyzyjna przez drzewo:**")
            
            tree = model.tree_
            node = 0
            path_steps = []
            
            sample = input_df.values[0]
            
            while tree.feature[node] != -2:  # Dopóki nie dojdzie do liścia powtarza
                feature_idx = tree.feature[node]
                feature_name = features[feature_idx]
                threshold = tree.threshold[node]
                feature_value = sample[feature_idx]
                
                if feature_value <= threshold:
                    decision = "TAK"
                    next_node = tree.children_left[node]
                else:
                    decision = "NIE"
                    next_node = tree.children_right[node]
                
                gini = tree.impurity[node]
                n_samples = tree.n_node_samples[node]
                value = tree.value[node][0]
                
                path_steps.append({
                    'Węzeł': f"N{node}",
                    'Pytanie': f"{feature_name} <= {threshold:.2f}?",
                    'Wartość cechy': f"{feature_value:.2f}",
                    'Odpowiedź': decision,
                    'Gini': f"{gini:.3f}",
                    'Próbki': n_samples
                })
                
                node = next_node
            
            # Ostatni węzeł (liść)
            gini = tree.impurity[node]
            n_samples = tree.n_node_samples[node]
            value = tree.value[node][0]
            class_idx = np.argmax(value)
            final_class = model.classes_[class_idx]
            
            path_steps.append({
                'Węzeł': f"N{node} (LIŚĆ)",
                'Pytanie': f"Finalna decyzja",
                'Wartość cechy': "-",
                'Odpowiedź': f"Klasa: {final_class}",
                'Gini': f"{gini:.3f}",
                'Próbki': n_samples
            })
            
            path_df = pd.DataFrame(path_steps)
            st.dataframe(path_df, hide_index=True, use_container_width=True)
            
            st.markdown(f"""
            **Interpretacja ścieżki:**
            Model przeszedł przez {len(path_steps)-1} węzłów decyzyjnych, 
            odpowiadając na pytania o wartości cech, aż dotarł do liścia z predykcją: **{prediction}**.
            
            Wartość Gini w końcowym węźle ({path_steps[-1]['Gini']}) pokazuje pewność decyzji - 
            im bliższa 0, tym bardziej jednoznaczna klasyfikacja.
            """)
            
            st.markdown("**Prawdopodobieństwa dla każdej klasy:**")
            proba_df = pd.DataFrame({
                'Klasa': model.classes_,
                'Prawdopodobieństwo': [f"{p:.2%}" for p in prediction_proba]
            })
            st.dataframe(proba_df, hide_index=True, use_container_width=True)