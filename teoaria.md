Algorytm TF-IDF (Term Frequency-Inverse Document Frequency) jest używany do oceny ważności słowa w zbiorze dokumentów. Jego głównym celem jest identyfikacja istotnych terminów w danym dokumencie w kontekście całego zbioru dokumentów. TF-IDF jest często używany w wyszukiwarkach, systemach rekomendacji oraz w przetwarzaniu języka naturalnego (NLP). Składa się z dwóch głównych komponentów: TF (Term Frequency) i IDF (Inverse Document Frequency).

### Term Frequency (TF)

TF mierzy, jak często dane słowo pojawia się w dokumencie. Jest to surowa liczba wystąpień danego słowa podzielona przez łączną liczbę słów w dokumencie. Może być obliczana według wzoru:

\[ \text{TF}(t, d) = \frac{\text{Liczba wystąpień słowa } t \text{ w dokumencie } d}{\text{Łączna liczba słów w dokumencie } d} \]

### Inverse Document Frequency (IDF)

IDF mierzy, jak rzadkie jest dane słowo w całym zbiorze dokumentów. Słowa, które pojawiają się w wielu dokumentach, mają niską wartość IDF, natomiast rzadkie słowa mają wysoką wartość IDF. Może być obliczana według wzoru:

\[ \text{IDF}(t, D) = \log \left( \frac{\text{Liczba dokumentów w zbiorze } D}{\text{Liczba dokumentów zawierających słowo } t} \right) \]

### TF-IDF

TF-IDF jest iloczynem TF i IDF, co oznacza, że ważność słowa wzrasta, jeśli często występuje w danym dokumencie, ale jednocześnie jest rzadkie w całym zbiorze dokumentów. Wzór na TF-IDF jest:

\[ \text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D) \]

gdzie:
- \( t \) to konkretne słowo,
- \( d \) to konkretny dokument,
- \( D \) to zbiór dokumentów.

### Przykład

Załóżmy, że mamy zbiór trzech dokumentów:
1. Dokument 1: "kot siedzi na macie"
2. Dokument 2: "pies biega po parku"
3. Dokument 3: "kot i pies są przyjaciółmi"

### Krok 1: Obliczanie TF

Najpierw obliczamy TF dla słowa "kot" w Dokumencie 1:

\[ \text{TF}(\text{"kot"}, \text{Dokument 1}) = \frac{1}{5} = 0.2 \]

### Krok 2: Obliczanie IDF

Następnie obliczamy IDF dla słowa "kot":

\[ \text{IDF}(\text{"kot"}, D) = \log \left( \frac{3}{2} \right) \approx 0.176 \]

### Krok 3: Obliczanie TF-IDF

Na końcu obliczamy TF-IDF dla słowa "kot" w Dokumencie 1:

\[ \text{TF-IDF}(\text{"kot"}, \text{Dokument 1}, D) = 0.2 \times 0.176 \approx 0.0352 \]

### Interpretacja

Wartość TF-IDF wskazuje, że słowo "kot" w Dokumencie 1 ma pewną ważność, która jest relatywnie niska, ponieważ słowo to nie jest ani bardzo częste w dokumencie, ani bardzo rzadkie w zbiorze dokumentów. Wartości TF-IDF można następnie wykorzystać do porównania dokumentów, identyfikacji kluczowych słów lub filtrowania mniej ważnych słów w różnych aplikacjach.