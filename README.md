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

    Tensorflow jest biblioteką która w wygodny dla użytkownika sposób daje możliwość wyuczanego własnej sieci korelacyjnej. Rozwiązanie problemu rozpoznawanią chorób przy pomocy tensorflow jest proste i wygodne. Wykorzystująć odpowiedni model (w moim przypadku jest to gotowy model ResNet50 z dwiema dodatykowymi wartwami Dense) pozwala przy niewielkim wysiłku napisać aplicje która będzie potrafiła rozpoznawac choroby u roślin. Największą wadą tego roziwązania jest to, iż potrzebna jest jednostka obliczenia o dużej wynajości, żeby uczenie zostało zrealizowne szybko i sprawnie (uczelnia nie zapenia takich komputer ＞﹏＜ - chyba).

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

- ### Drugie trzecie: ----

  - ### Podejście do problemu

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

# 3. Opis wybranej koncepcji (5 pkt)

W sekcji trzeciej należy opisać wybraną przez siebie istniejącą lub nową koncepcję rozwiązania problemu. Należy wyraźnie określić jakie dane będą potrzebne (czy są dostępne publicznie), co jest i jaką formę będzie przyjmować wyjście algorytmu, na czym polega zastosowana metoda oraz co jest potrzebne do jej realizacji w rzeczywistym świecie. Należy również przygotwać procedurę testowania rozwiązania oraz zidentyfikować ewentualne problemy.

# 4. Proof of concept (7 pkt)

Należy przygotować demo, które będzie pokazywać działanie wybranej koncepcji. Demo może zostać wykonane dla reprezentatywnej symplifikacji problemu.

# 5. Źródła

ChatGPT - pomoc merytoryczna
