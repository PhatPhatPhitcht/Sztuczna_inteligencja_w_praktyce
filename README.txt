# Interaktywne Algorytmy Uczenia Maszynowego

Interaktywna aplikacja webowa do wizualizacji i nauki algorytmów uczenia maszynowego, zbudowana przy użyciu Streamlit.

## Spis treści
- [O Projekcie](#o-projekcie)
- [Funkcjonalności](#funkcjonalności)
- [Wymagania](#wymagania)
- [Instalacja](#instalacja)
- [Uruchomienie](#uruchomienie)
- [Struktura Projektu](#struktura-projektu)
- [Dostępne Algorytmy](#dostępne-algorytmy)
- [Używane Zbiory Danych](#używane-zbiory-danych)
- [Użytkowanie](#użytkowanie)
- [Technologie](#technologie)

## O Projekcie

Aplikacja umożliwia interaktywną eksplorację i wizualizację popularnych algorytmów uczenia maszynowego. Użytkownik może eksperymentować z różnymi parametrami, wczytywać własne dane i obserwować działanie algorytmów w czasie rzeczywistym.

**Główne cele projektu:**
- Edukacyjna wizualizacja algorytmów ML
- Interaktywne dostosowywanie parametrów
- Obsługa własnych zbiorów danych
- Wizualizacja procesu uczenia krok po kroku

## Funkcjonalności

### Algorytmy Klasteryzacji
- **K-means** - grupowanie danych z wizualizacją iteracyjną
- **DBSCAN** - klasteryzacja oparta na gęstości

### Algorytmy Klasyfikacji
- **Drzewa Decyzyjne** - klasyfikacja z wizualizacją drzewa
- **Regresja Logistyczna** - binarna i wieloklasowa klasyfikacja

### Algorytmy Regresji
- **Regresja Liniowa** - przewidywanie wartości ciągłych
- **Drzewa Decyzyjne (Regresja)** - nieliniowa regresja
- **Regresja Ważona** - regresja z wagami próbek

### Dodatkowe Funkcjonalności
- **Standaryzacja danych** (StandardScaler)
- **Redukcja wymiarów** (PCA) - automatyczna dla więcej niż 2 cech
- **Metoda łokcia** - pomoc w wyborze optymalnej liczby klastrów
- **Wczytywanie własnych danych** - obsługa CSV, JSON, XML
- **Wizualizacja EDA** - statystyki opisowe, wykresy rozkładów

## Wymagania

### Wymagania Systemowe
- Python 3.8 lub nowszy
- 4 GB RAM (zalecane)
- Przeglądarka internetowa (Chrome, Firefox, Safari, Edge)

### Zależności Python
Wszystkie wymagane biblioteki znajdują się w pliku `requirements.txt`:

```
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
```

## Instalacja

### Krok 1: Sklonuj repozytorium
```bash
git clone https://github.com/PhatPhatPhitcht/Sztuczna_inteligencja_w_praktyce.git
cd Sztuczna_inteligencja_w_praktyce
```

### Krok 2: Utwórz środowisko wirtualne (opcjonalnie, ale zalecane)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Krok 3: Zainstaluj zależności
```bash
pip install -r requirements.txt
```

## Uruchomienie

Po zainstalowaniu zależności, uruchom aplikację komendą:

```bash
streamlit run main.py
```

Aplikacja otworzy się automatycznie w przeglądarce pod adresem `http://localhost:8501`

## Struktura Projektu

```
Sztuczna_inteligencja_w_praktyce/
│
├── main.py                          # Strona główna aplikacji
├── requirements.txt                 # Lista zależności
├── house_data.csv                   # Zbiór danych o domach
├── README.txt                       # Informacje o projekcie
│
├── pages/                           # Podstrony z algorytmami
│   ├── k-means.py                  # Algorytm K-means
│   ├── dbscan.py                   # Algorytm DBSCAN
│   ├── decision_trees.py           # Drzewa decyzyjne (klasyfikacja)
│   ├── logistic_regression.py      # Regresja logistyczna
│   ├── lreg.py                     # Regresja liniowa
│   ├── dec_trees_regression.py     # Drzewa decyzyjne (regresja)
│   ├── reg_wght.py                 # Regresja ważona
│   └── elbow.py                    # Metoda łokcia
│
└── utils/                           # Moduły pomocnicze
    └── file_loader.py              # Wczytywanie plików

```

## Dostępne Algorytmy

### 1. K-means (Klasteryzacja)
**Ścieżka:** `pages/k-means.py`

**Opis:** Algorytm grupuje dane minimalizując sumę kwadratów odległości punktów od centroidów klastrów.

**Parametry:**
- Liczba klastrów (2-10)
- Liczba wizualizowanych iteracji (3-10)

**Funkcjonalności:**
- Wizualizacja iteracyjnych zmian centroidów
- Automatyczne PCA dla więcej niż 2 wymiarów
- Wybór zmiennych do analizy
- Wsparcie metody łokcia do wyboru optymalnej liczby klastrów

**Implementacja:** Własna klasa `KMeansIterative` z zapisem historii iteracji.

### 2. DBSCAN (Klasteryzacja)
**Ścieżka:** `pages/dbscan.py`

**Opis:** Algorytm klasteryzacji oparty na gęstości, wykrywa klastry o dowolnych kształtach i identyfikuje obserwacje odstające.

**Parametry:**
- Epsilon (promień sąsiedztwa)
- MinPts (minimalna liczba punktów w klastrze)

### 3. Drzewa Decyzyjne - Klasyfikacja
**Ścieżka:** `pages/decision_trees.py`

**Opis:** Model klasyfikacyjny budujący drzewo decyzyjne na podstawie atrybutów danych.

**Parametry:**
- Maksymalna głębokość drzewa
- Minimalna liczba próbek w liściu
- Kryterium podziału (gini/entropy)

### 4. Regresja Logistyczna
**Ścieżka:** `pages/logistic_regression.py`

**Opis:** Model klasyfikacji probabilistycznej wykorzystujący funkcję logistyczną.

### 5. Regresja Liniowa
**Ścieżka:** `pages/lreg.py`

**Opis:** Podstawowy model regresji przewidujący wartości ciągłe na podstawie relacji liniowych.

### 6. Drzewa Decyzyjne - Regresja
**Ścieżka:** `pages/dec_trees_regression.py`

**Opis:** Nieliniowy model regresji wykorzystujący strukturę drzewa decyzyjnego.

### 7. Regresja Ważona
**Ścieżka:** `pages/reg_wght.py`

**Opis:** Regresja liniowa z możliwością przypisania różnych wag obserwacjom.

## Używane Zbiory Danych

### 1. Iris Dataset (Domyślny)
**Źródło:** Wbudowany w seaborn

**Opis:** Klasyczny zbiór zawierający pomiary 150 kwiatów irysa z trzech gatunków.

**Cechy:**
- `sepal_length` - długość kielicha (cm)
- `sepal_width` - szerokość kielicha (cm)
- `petal_length` - długość płatka (cm)
- `petal_width` - szerokość płatka (cm)
- `species` - gatunek (setosa, versicolor, virginica)

**Zastosowanie:** Klasteryzacja, klasyfikacja

### 2. House Sales in King County
**Źródło:** `house_data.csv`

**Opis:** Dane o sprzedaży domów w hrabstwie King (Waszyngton, USA) zawierające informacje o cenach i charakterystykach nieruchomości.

**Główne cechy:**
- `price` - cena sprzedaży (USD) - zmienna docelowa
- `bedrooms` - liczba sypialni
- `bathrooms` - liczba łazienek
- `sqft_living` - powierzchnia mieszkalna (stopy kwadratowe)
- `sqft_lot` - powierzchnia działki
- `floors` - liczba pięter
- `waterfront` - widok na wodę (0/1)
- `view` - jakość widoku (0-4)
- `condition` - stan techniczny (1-5)
- `grade` - ocena jakości (1-13)
- `yr_built` - rok budowy
- `lat, long` - współrzędne geograficzne

**Zastosowanie:** Regresja (przewidywanie cen)

### 3. Własne Dane
Aplikacja obsługuje wczytywanie własnych zbiorów danych w formatach:
- **CSV** - wartości rozdzielane przecinkami
- **JSON** - format JavaScript Object Notation
- **XML** - eXtensible Markup Language

**Wymagania:**
- Dane numeryczne (dla algorytmów ML)
- Brak brakujących wartości
- Poprawna struktura (kolumny/wiersze)

## Użytkowanie

### Podstawowy Przepływ Pracy

1. **Uruchom aplikację:**
   ```bash
   streamlit run main.py
   ```

2. **Wybierz zbiór danych:**
   - Użyj domyślnego zbioru Iris
   - Wybierz zbiór House Sales
   - Wczytaj własne dane (CSV/JSON/XML)

3. **Eksploruj dane:**
   - Przeglądaj statystyki opisowe
   - Analizuj wizualizacje (boxplot, scatterplot, histogramy)

4. **Wybierz algorytm:**
   - Kliknij link do wybranego algorytmu z menu głównego

5. **Skonfiguruj parametry:**
   - Dostosuj parametry algorytmu za pomocą suwaków
   - Wybierz zmienne do analizy (jeśli dostępne)

6. **Uruchom algorytm:**
   - Kliknij przycisk "Uruchom [nazwa algorytmu]"
   - Obserwuj wizualizacje i wyniki

### Przykład: Uruchomienie K-means

1. Na stronie głównej wybierz zbiór Iris
2. Kliknij "K-means" w sekcji Klasteryzacja
3. Wybierz zmienne do analizy (np. petal_length, petal_width)
4. Ustaw liczbę klastrów (np. 3)
5. Ustaw liczbę wizualizowanych iteracji (np. 5)
6. Kliknij "Uruchom K-means"
7. Obserwuj iteracyjne zmiany centroidów i przypisań klastrów

### Praca z PCA

Gdy wybierzesz więcej niż 2 zmienne, aplikacja automatycznie zastosuje PCA (Principal Component Analysis) do redukcji wymiarów:

- **PC1 i PC2** - dwie główne składowe
- **Procent wariancji** - informacja o zachowaniu informacji
- Przykład: "PC1 (45.3% wariancji)" oznacza, że pierwsza składowa główna zachowuje 45.3% zróżnicowania danych

### Wczytywanie Własnych Danych

**Format CSV:**
```csv
feature1,feature2,feature3,target
1.2,3.4,5.6,A
2.3,4.5,6.7,B
```

**Format JSON:**
```json
[
  {"feature1": 1.2, "feature2": 3.4, "target": "A"},
  {"feature1": 2.3, "feature2": 4.5, "target": "B"}
]
```

**Format XML:**
```xml
<data>
  <record>
    <feature1>1.2</feature1>
    <feature2>3.4</feature2>
    <target>A</target>
  </record>
</data>
```

## Technologie

### Backend
- **Python 3.8+** - język programowania
- **Streamlit** - framework aplikacji webowej
- **Pandas** - manipulacja danymi
- **NumPy** - operacje numeryczne
- **scikit-learn** - algorytmy uczenia maszynowego

### Wizualizacja
- **Matplotlib** - tworzenie wykresów
- **Seaborn** - zaawansowane wizualizacje statystyczne

### Algorytmy ML
- **StandardScaler** - standaryzacja danych
- **PCA** - redukcja wymiarów
- **KMeans** - klasteryzacja (własna implementacja iteracyjna)
- **DBSCAN** - klasteryzacja oparta na gęstości
- **DecisionTreeClassifier/Regressor** - drzewa decyzyjne
- **LogisticRegression** - klasyfikacja probabilistyczna
- **LinearRegression** - regresja liniowa

## Rozwiązywanie Problemów

### Aplikacja nie uruchamia się
- Sprawdź czy masz zainstalowany Python 3.8+
- Upewnij się, że wszystkie zależności są zainstalowane: `pip install -r requirements.txt`
- Sprawdź czy port 8501 nie jest zajęty

### Błąd przy wczytywaniu danych
- Upewnij się, że plik nie zawiera brakujących wartości
- Sprawdź format pliku (CSV, JSON, XML)
- Dla CSV sprawdź czy separator jest poprawny

### Wykresy nie wyświetlają się
- Odśwież stronę (F5)
- Sprawdź konsolę przeglądarki pod kątem błędów JavaScript
- Spróbuj innej przeglądarki

### PCA zastosowane automatycznie
- Jest to normalne zachowanie gdy wybierzesz więcej niż 2 zmienne
- PCA redukuje wymiary do 2D dla lepszej wizualizacji
- Możesz ograniczyć wybór zmiennych do 2, aby uniknąć PCA

## Rozwój Projektu

### Możliwe Rozszerzenia
- Dodanie więcej algorytmów (SVM, Random Forest, Neural Networks)
- Eksport wyników do pliku
- Porównanie wielu algorytmów jednocześnie
- Interaktywne 3D wizualizacje
- Optymalizacja hiperparametrów
- Metryki ewaluacji modeli
- Historia eksperymentów

## Licencja

Projekt edukacyjny - brak określonej licencji.

## Autor

**PhatPhatPhitcht**

GitHub: [https://github.com/PhatPhatPhitcht](https://github.com/PhatPhatPhitcht)

Repozytorium: [Sztuczna_inteligencja_w_praktyce](https://github.com/PhatPhatPhitcht/Sztuczna_inteligencja_w_praktyce)

---

*Ostatnia aktualizacja dokumentacji: Styczeń 2026*