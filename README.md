# 1. Opis rzeczywistego problemu

- ### Cel projektu

  Celem projektu jest opracowanie i implementacja systemu, który automatycznie rozpoznaje patogenu roślin na podstawie zdjęć liści lub innych części rośliny.
  Efektem końcowym będzie model, który na podstawie wprowadzonego zdjęcia potrafi z dużą dokładnością zidentyfikować występującą chorobę, szkodnika, czy roślina jest zdrowa.
  Takie rozwiązanie pozwoli na szybszą diagnostykę w terenie, minimalizując potrzebę angażowania ekspertów na każdym etapie oraz umożliwi szybką interwencję, co jest
  kluczowe w minimalizowaniu strat w uprawach.

- ### Motywacja i znaczenie problemu

  Rozpoznawanie chorób roślin na wczesnym etapie jest kluczowe dla rolnictwa i gospodarki żywnościowej, ponieważ szybka
  identyfikacja chorób pozwala ograniczyć straty plonów, zmniejszyć koszty produkcji i obniżyć stosowanie środków chemicznych.
  Obecnie identyfikacja chorób roślin wymaga dużej wiedzy specjalistycznej i jest czasochłonna, ponieważ wymaga oględzin, a czasem
  laboratoryjnych testów. Wprowadzenie systemu opartego na analizie obrazu pozwoliłoby na automatyzację procesu diagnostycznego, co jest
  szczególnie istotne w rolnictwie intensywnym, ekologicznym i na dużą skalę.

- ### Dane wejściowe

  Dane wejściowe będą stanowiły zdjęcia liści, łodyg, kwiatów lub owoców roślin wykonane w różnych warunkach oświetleniowych i klimatycznych.
  Zdjęcia będą mierzone pod kątem występowania cech chorobowych, takich jak zmiany koloru, plamy, przebarwienia, deformacje, **_brazwzrostu_**, chorozy czy nekrozy.
  Wstępna obróbka zdjęć obejmie ich standaryzację pod kątem jakości, rozdzielczości oraz normalizację kolorystyczną, co pozwoli wyeliminować
  błędy wynikające z różnic w sprzęcie fotograficznym lub warunkach pogodowych.

- ### Zagadnienia z zakresu sztucznej inteligencji
  Projekt będzie oparty na metodach rozpoznawania obrazów i klasyfikacji wizyjnej przy użyciu technik głębokiego uczenia (ang. _deep learning_).
  Dokładniej, zastosowanie znajdą tutaj sieci splotowe (CNN - _Convolutional Neural Networks_), które są obecnie jednymi z najskuteczniejszych narzędzi do analizy obrazów.
  W ramach systemu będzie zaimplementowany model uczony na dużych zbiorach zdjęć, który rozpozna różnorodne wzorce charakterystyczne dla chorób roślin.

# 2. State of art (3 pkt)

W sekcji drugiej należy zwięźle opisać znane koncepcje (minimum 3) rozwiązania tego lub podobnego problemu ze wskazaniem ich mocnych i słabych stron. Opisywane rozwiązania muszą być różnorodne, tj. korzystać z różnych metod sztucznej inteligencji.

- ### Pierwsze podejescie: OpenCV oraz Scikit-Learn

  - ### Podejście do problemu

    Rozwiązanie to polega na wykorzystaniu funkcji SIFT z biblioteki OpenCV do wyznaczenia punktów szczególnych obrazu. Mając punkty szczególne, można je pokazać na wykresie typu punktowego. Analizując takie wykresy, spróbujemy dopasować sposób klasyfikacji danych i później na ich podstawie próbować przewidywać dane. Naptkałem problem, iż dane są zbyt rozrzucone przez co ciężko storzyć klasyfikację.

  - ### Wykorzystane biblioteki

    |  Biblioteki   |
    | :-----------: |
    | opencv-python |
    | scikit-learn  |
    |   kagglehub   |

  - ### Tabela zalet i wad

    |                Zalety                 |                      Wady                       |
    | :-----------------------------------: | :---------------------------------------------: |
    | Kontrola nad procesem ekstrakcji cech |    Wymaga więcej pracy i wiedzy dziedzinowej    |
    |    Mniejsze wymagania obliczeniowe    |               Brak skalowalności                |
    |   Przejrzystość i interpretowalność   |        Brak automatycznej optymalizacji         |
    | Skuteczność na małych zbiorach danych | Mniej efektywne w przypadku złożonych problemów |

  - ### przykładowe dane

    - ### Obraz z nałożonymi punktami algorytmu SIFT

    ![Inline image alt text](https://github.com/Kabir132/WdSI_Projekt/blob/main/readme_images/podejscie1_przykladowe_dane.png?raw=true "Optional image title")

    - ### Wykresy punktów SIFT. Każdy wykres ma naniesione na siebie 50 różnych zdjęć

    ![Inline image alt text](https://github.com/Kabir132/WdSI_Projekt/blob/main/readme_images/podejscie1_przykladowy_wykres.png?raw=true "Optional image title")

- ### Drugie podejescie: Tensorflow

  - ### Podejście do problemu

    Tensorflow jest biblioteką która w wygodny dla użytkownika sposób daje możliwość wyuczanego własnej sieci korelacyjnej. Rozwiązanie problemu rozpoznawanią chorób przy pomocy tensorflow jest proste i wygodne. Wykorzystująć odpowiedni model (w moim przypadku jest to gotowy model ResNet50 z dwiema dodatykowymi wartwami Dense) pozwala przy niewielkim wysiłku napisać aplicje która będzie potrafiła rozpoznawac choroby u roślin. Największą wadą tego roziwązania jest to, iż potrzebna jest jednostka obliczenia o dużej wynajości, żeby uczenie zostało zrealizowne szybko i sprawnie.

  - ### Wykorzystane biblioteki

    |  Biblioteki   |
    | :-----------: |
    |   kagglehub   |
    | opencv-python |
    |     keras     |
    |  tensorflow   |

  - ### Tabela zalet i wad

    |                     Zalety                      |              Wady              |
    | :---------------------------------------------: | :----------------------------: |
    | End-to-end learning (automatyczne uczenie cech) | Wysokie wymagania obliczeniowe |
    |                 Wszechstronność                 | Potrzeba dużych zbiorów danych |
    |     Szeroka społeczność i zasoby edukacyjne     |   Mniejsza interpretowalność   |
    |        Automatyczna optymalizacja modeli        |    Problemy z debugowaniem     |

  - przykładowe dane

    - ### Model

    | Layer (type)          | Output Shape | Param #    |
    | --------------------- | ------------ | ---------- |
    | resnet50 (Functional) | (None, 2048) | 23,587,712 |
    | flatten (Flatten)     | (None, 2048) | 0          |
    | dense (Dense)         | (None, 512)  | 1,049,088  |
    | dense_1 (Dense)       | (None, 58)   | 29,754     |

    Total params: 24,666,554 (94.10 MB)

    Trainable params: 1,078,842 (4.12 MB)

    Non-trainable params: 23,587,712 (89.98 MB)

    - ### Wynik naucznia modelu dla epoch = 4

    ![Inline image alt text](https://github.com/Kabir132/WdSI_Projekt/blob/main/readme_images/podejscie2_wynik_nauczania.png?raw=true "Optional image title")

- ### Drugie trzecie: TensorFlow ResNet

## Podejście do Problemu

W tej metodzie celem jest wytrenowanie modelu na podstawie własnych warstw, bez pomocy gotowych architektór.

- Model jest projektowany z myślą o prostocie i efektywności, dostosowując liczbę warstw i parametrów do specyfiki danych.

- Zastosowano techniki regularizacji, takie jak dropout oraz batch normalization, aby zapobiec przeuczeniu modelu.

- Model wykorzystuje klasyczne warstwy konwolucyjne (Conv2D) i maksymalne próbkowanie (MaxPooling), co zapewnia wydajność przy jednoczesnym zachowaniu wysokiej dokładności.

---

## Wykorzystane Biblioteki

W projekcie użyto następujących bibliotek:

|    Biblioteki     |
| :---------------: |
|   **kagglehub**   |
| **opencv-python** |
|     **keras**     |
|  **tensorflow**   |

---

## Zalety i Wady Podejścia z Budową Modelu od Podstaw

|                       **Zalety**                       |                  **Wady**                   |
| :----------------------------------------------------: | :-----------------------------------------: |
|         Pełna kontrola nad architekturą modelu         | Wymaga większego wysiłku przy projektowaniu |
| Możliwość optymalizacji pod kątem specyficznych danych |    Większe ryzyko błędów w implementacji    |
|         Mniejsze zapotrzebowanie na pamięć GPU         |      Może wymagać więcej prób i testów      |
|     Uniknięcie zależności od gotowych architektur      |    Potrzeba większej wiedzy eksperckiej     |

---

## Przykładowe Dane

Do trenowania i testowania systemu wykorzystano zbiór danych obrazów roślin z następującymi klasami:

- Zdrowe rośliny
- Rośliny zaatakowane przez choroby (np. plamistość liści, mączniak)
- Rośliny uszkodzone przez szkodniki (np. mszyce, przędziorki)

### Proces przetwarzania danych

1. Skalowanie obrazów do rozmiaru 128x128 pikseli, co zmniejsza obciążenie obliczeniowe.
2. Normalizacja wartości pikseli w zakresie [0, 1].
3. Augmentacja danych, obejmująca obrót, zmiany jasności i kontrastu, aby zwiększyć różnorodność danych.

---

## Architektura Modelu

Model został zbudowany od podstaw i zawiera następujące elementy:

1. Kilka warstw konwolucyjnych (Conv2D) z różnymi filtrami (32, 64, 128), aby uchwycić zróżnicowane cechy obrazu.
2. Warstwy maksymalnego próbkowania (MaxPooling2D) po każdej warstwie konwolucyjnej, aby zmniejszyć wymiarowość danych.
3. Dropout, aby zapobiec przeuczeniu modelu.
4. Gęsta warstwa (Dense) na końcu z funkcją aktywacji softmax dla klasyfikacji wieloklasowej.

---

## Cel i Korzyści Projektu

Efektem końcowym projektu jest system umożliwiający automatyczną diagnostykę w terenie, eliminując potrzebę stałego udziału ekspertów i umożliwiając:

- Szybkie i precyzyjne wykrywanie chorób oraz szkodników.
- Zwiększenie wydajności upraw poprzez szybką interwencję.
- Minimalizowanie strat wynikających z błędnej diagnostyki lub opóźnionego działania.

System, zbudowany od podstaw w TensorFlow i Keras, oferuje pełną kontrolę nad architekturą i parametrami, co pozwala na lepsze dostosowanie do specyficznych potrzeb użytkownika oraz zwiększenie efektywności w rolnictwie precyzyjnym.

# 3. Opis wybranej koncepcji (5 pkt)

## Wybór metody numer dwa: TensorFlow z modelem ResNet50

### Podejście do problemu

Wykorzystanie gotowego modelu ResNet50 w TensorFlow pozwala na szybkie i efektywne rozwiązanie problemu rozpoznawania chorób roślin. ResNet50 to głęboka sieć neuronowa, która została przeszkolona na dużym zbiorze danych ImageNet, co czyni ją bardzo skuteczną w zadaniach związanych z klasyfikacją obrazów. W naszym projekcie model ResNet50 został dostosowany poprzez dodanie dwóch dodatkowych warstw Dense, co pozwala na lepsze dopasowanie do specyficznych danych dotyczących chorób roślin.

### Zalety metody

1. **Szybkie wdrożenie**: Wykorzystanie gotowego modelu ResNet50 znacznie skraca czas potrzebny na przygotowanie i trenowanie modelu.
2. **Wysoka dokładność**: ResNet50, dzięki swojej głębokiej architekturze, osiąga wysoką dokładność w zadaniach związanych z klasyfikacją obrazów.
3. **Transfer learning**: Możliwość wykorzystania transfer learningu pozwala na adaptację modelu do specyficznych danych przy użyciu mniejszej ilości danych treningowych.
4. **Wsparcie społeczności**: ResNet50 jest dobrze udokumentowany i szeroko stosowany, co ułatwia rozwiązywanie problemów i optymalizację modelu.

### Wady metody

1. **Wysokie wymagania obliczeniowe**: Trenowanie i inferencja modelu ResNet50 wymagają dużej mocy obliczeniowej, co może być problematyczne bez dostępu do odpowiedniego sprzętu.
2. **Mniejsza kontrola nad architekturą**: Korzystanie z gotowego modelu ogranicza możliwość dostosowania architektury do specyficznych potrzeb projektu.
3. **Potrzeba dużych zbiorów danych**: Aby w pełni wykorzystać potencjał modelu ResNet50, konieczne jest posiadanie dużych zbiorów danych treningowych.
4. **Problemy z interpretowalnością**: Modele głębokiego uczenia, takie jak ResNet50, są trudniejsze do interpretacji w porównaniu do tradycyjnych metod uczenia maszynowego.

### Wnioski

Wybór metody opartej na TensorFlow z modelem ResNet50 jest uzasadniony ze względu na wysoką dokładność i efektywność w zadaniach związanych z klasyfikacją obrazów. Pomimo pewnych wad, takich jak wysokie wymagania obliczeniowe i mniejsza kontrola nad architekturą, korzyści płynące z szybkiego wdrożenia i możliwości transfer learningu przeważają. Metoda ta jest dobrym narzędziem do realizacji projektu rozpoznawania chorób roślin, umożliwiając szybkie i precyzyjne wykrywanie chorób.

# 4. Proof of concept (7 pkt)

## Proof of Concept

### Przygotowanie środowiska

Aby zrealizować proof of concept, konieczne było przygotowanie odpowiedniego środowiska programistycznego. Wykorzystano do tego Google Colab, który umożliwia korzystanie z GPU, co znacznie przyspiesza proces trenowania modelu. Zainstalowano niezbędne biblioteki, takie jak TensorFlow, Keras, OpenCV oraz KaggleHub.

### Pobranie i przygotowanie danych

Dane do trenowania modelu zostały pobrane z Kaggle za pomocą biblioteki KaggleHub. Zbiór danych zawierał obrazy roślin z różnymi chorobami oraz zdrowe rośliny. Dane zostały podzielone na zbiory treningowe i walidacyjne w stosunku 80:20.

### Wczytanie i przetworzenie danych

Obrazy zostały wczytane i przeskalowane do rozmiaru 224x224 pikseli, co jest wymagane przez model ResNet50. Dodatkowo, zastosowano augmentację danych, aby zwiększyć różnorodność zbioru treningowego i zapobiec przeuczeniu modelu.

### Budowa modelu

Wykorzystano gotowy model ResNet50 z pretrenowanymi wagami na zbiorze ImageNet. Dodano dwie dodatkowe warstwy Dense, aby dostosować model do specyficznych danych dotyczących chorób roślin. Model został skompilowany z użyciem optymalizatora Adam oraz funkcji straty SparseCategoricalCrossentropy.

### Trenowanie modelu

Model został wytrenowany na zbiorze treningowym przez 4 epoki. Proces trenowania był monitorowany za pomocą metryk dokładności i straty zarówno dla zbioru treningowego, jak i walidacyjnego. Zastosowano również mechanizm wczesnego zatrzymania, aby zapobiec przeuczeniu modelu.

### Wyniki trenowania

Model osiągnął wysoką dokładność na zbiorze walidacyjnym, co potwierdza skuteczność podejścia opartego na ResNet50. Wyniki trenowania zostały zwizualizowane na wykresach, które pokazują zmiany dokładności i straty w czasie.

### Testowanie modelu

Model został przetestowany na nowych, nieznanych wcześniej obrazach, aby sprawdzić jego zdolność do generalizacji. Wyniki testów były zadowalające, co potwierdza, że model potrafi skutecznie rozpoznawać choroby roślin na podstawie zdjęć.

### Wnioski

Proof of concept potwierdził, że wykorzystanie modelu ResNet50 w TensorFlow jest skutecznym podejściem do rozpoznawania chorób roślin. Model osiągnął wysoką dokładność i jest w stanie generalizować na nowe dane. W przyszłości można rozważyć dalszą optymalizację modelu oraz zwiększenie zbioru danych treningowych, aby jeszcze bardziej poprawić jego wydajność.

# 5. Źródła

- [KN RAI kurs machine learning](https://github.com/KoloNaukowe-RAI/Kurs-Machine-Learning)
- [Podobne projekty na githubie](https://github.com/)
- Wiedza z zajęć
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Scikit-Learn Documentation](https://scikit-learn.org/stable/documentation.html)
