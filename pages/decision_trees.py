import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

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
Klasyfikatory drzew decyzyjnych są atrakcyjnymi modelami, jeśli zależy nam na interpretowalności. Jak sugeruje nazwa „drzewo decyzyjne”, możemy myśleć o tym modelu jako o metodzie dzielenia danych poprzez podejmowanie decyzji na podstawie zadawania serii pytań.          

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

    #st.write(f"**Cechy numeryczne użyte do treningu:** {', '.join(numeric_features)}")

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

            st.session_state.decision_tree_model = model
            st.session_state.model_features = numeric_features
            st.session_state.target_column = target_column
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test

            # Metryki
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            #st.success(f"Model wytrenowany. Dokładność: {accuracy:.2%}")

        except Exception as e:
            st.error(f"Błąd podczas treningu: {str(e)}")

    if 'decision_tree_model' in st.session_state:
        st.write("---")

        model = st.session_state.decision_tree_model
        features = st.session_state.model_features
        target = st.session_state.target_column

        st.markdown("**Drzewo decyzyjne**")

        fig, ax = plt.subplots(figsize=(20, 10))
        plot_tree(
            model,
            feature_names=features,
            class_names=[str(c) for c in model.classes_],
            filled=True,
            rounded=True,
            ax=ax,
            fontsize=10
        )
        st.pyplot(fig)

        st.markdown("""
        Wyjaśnijmy metryki które możesz zobaczyć w węzłach:

        **Gini** to  współczynnik Giniego, czyli miara zanieczyszczenia (impurity) węzła, która pokazuje, czy w danym węźle drzewa znajdują się próbki należące do różnych klas jednocześnie . Dla wartość  0 oznacza węzeł czysty (wszystkie próbki należą do jednej klasy), a Gini maksymalne (np. 0.5 dla 2 klas) oznacza że nie ma jednoznacznej odpowiedzi, którą klasę ten węzeł reprezentuje.

        **Samples** to liczba próbek treningowych, które trafiły do tego węzła, przydatne do wyobrażenia sobie wpłytu paremetrów *min_samplessplit/leaf*.           

        **Value** to tablica pokazująca rozkład klas w węźle, czyli ile próbek z każdej klasy znajduje się w tym węźle.

        Zmieniające się kolory węzłów oznaczają dominującą klasę w tym węźle i jak jest czysta - połączenie gini i value, dlatego wraz ze zbliżaniem się do liścia zauważymy zwiększoną intensywność koloru wraz z zwiększaniem się pewności algorytmu.
        """)

        ## Metryki
        #st.markdown("**Metryki modelu**")
        #y_test = st.session_state.y_test
        #X_test = st.session_state.X_test
        #y_pred = model.predict(X_test)
        #accuracy = accuracy_score(y_test, y_pred)
        #
        #col1, col2 = st.columns(2)
        #with col1:
        #    st.metric("Dokładność", f"{accuracy:.2%}")
        #with col2:
        #    st.metric("Głębokość drzewa", model.get_depth())

        st.subheader("Testowanie modelu")
        st.markdown("Wprowadź wartości dla nowej próbki:")

        input_data = {}
        cols = st.columns(len(features))

        for idx, feature in enumerate(features):
            with cols[idx]:
                min_val = float(df[feature].min())
                max_val = float(df[feature].max())
                mean_val = float(df[feature].mean())

                input_data[feature] = st.number_input(
                    feature,
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    step=(max_val - min_val) / 100
                )

        if st.button("Klasyfikuj próbkę"):
            input_df = pd.DataFrame([input_data])

            prediction = model.predict(input_df)[0]
            prediction_proba = model.predict_proba(input_df)[0]

            st.markdown("""
            **Wynik predykcji:**
            """)
            st.success(f"{prediction}")

            #st.write("**Prawdopodobieństwa dla każdej klasy:**")
            #proba_df = pd.DataFrame({
            #    'Klasa': model.classes_,
            #    'Prawdopodobieństwo': prediction_proba
            #})
            #st.dataframe(proba_df, hide_index=True)