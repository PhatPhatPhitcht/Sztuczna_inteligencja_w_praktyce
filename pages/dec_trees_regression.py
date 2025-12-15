import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

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

        st.markdown("**Drzewo decyzyjne**")

        fig, ax = plt.subplots(figsize=(20, 10))
        plot_tree(
            model,
            feature_names=features,
            filled=True,
            rounded=True,
            ax=ax,
            fontsize=10
        )
        st.pyplot(fig)

        st.markdown("""
        Wyjaśnijmy metryki które możesz zobaczyć w węzłach:

        **Squared_error** (lub MSE) to średni kwadrat błędu w węźle, który pokazuje jak bardzo wartości w tym węźle różnią się od ich średniej. Wartość 0 oznacza węzeł idealny (wszystkie próbki mają tę samą wartość), a wyższe wartości oznaczają większą zmienność przewidywań.

        **Samples** to liczba próbek treningowych, które trafiły do tego węzła, przydatne do wyobrażenia sobie wpływu parametrów *min_samples_split/leaf*.

        **Value** to przewidywana wartość w tym węźle, obliczana jako średnia wartości docelowych wszystkich próbek, które trafiły do tego węzła. W liściach jest to finalna predykcja modelu.

        Zmieniające się kolory węzłów reprezentują przewidywaną wartość - ciemniejsze kolory mogą oznaczać niższe lub wyższe wartości w zależności od skali kolorów. Wraz ze zbliżaniem się do liści, kolory stają się bardziej jednolite, gdyż węzły zawierają bardziej jednorodne wartości.
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

        if st.button("Przewiduj wartość"):
            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)[0]

            st.markdown("**Wynik predykcji:**")
            st.success(f"{prediction:.4f}")