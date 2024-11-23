**Opis rzeczywistego problemu: Rozpoznawanie chorób roślin na podstawie zdjęć**

### Cel projektu
Celem projektu jest opracowanie i implementacja systemu, który automatycznie rozpoznaje choroby roślin na podstawie zdjęć liści lub innych części rośliny. 
Efektem końcowym będzie model, który na podstawie wprowadzonego zdjęcia potrafi z dużą dokładnością zidentyfikować występujący patogen lub wskazać, że roślina jest zdrowa. 
Takie rozwiązanie pozwoli na szybszą diagnostykę w terenie, minimalizując potrzebę angażowania ekspertów na każdym etapie oraz umożliwi szybką interwencję, co jest 
kluczowe w minimalizowaniu strat w uprawach.

### Motywacja i znaczenie problemu
Rozpoznawanie chorób roślin na wczesnym etapie jest kluczowe dla rolnictwa i gospodarki żywnościowej, ponieważ szybka 
identyfikacja chorób pozwala ograniczyć straty plonów, zmniejszyć koszty produkcji i obniżyć stosowanie środków chemicznych. 
Obecnie identyfikacja chorób roślin wymaga dużej wiedzy specjalistycznej i jest czasochłonna, ponieważ wymaga oględzin, a czasem 
laboratoryjnych testów. Wprowadzenie systemu opartego na analizie obrazu pozwoliłoby na automatyzację procesu diagnostycznego, co jest 
szczególnie istotne w rolnictwie intensywnym i na dużą skalę.

### Dane wejściowe
Dane wejściowe będą stanowiły zdjęcia liści, łodyg lub owoców roślin wykonane w różnych warunkach oświetleniowych i klimatycznych. 
Zdjęcia będą mierzone pod kątem występowania cech chorobowych, takich jak zmiany koloru, plamy, przebarwienia, deformacje, czy widoczne objawy zasychania. 
Wstępna obróbka zdjęć obejmie ich standaryzację pod kątem jakości, rozdzielczości oraz normalizację kolorystyczną, co pozwoli wyeliminować 
błędy wynikające z różnic w sprzęcie fotograficznym lub warunkach pogodowych.

### Zagadnienia z zakresu sztucznej inteligencji
Projekt będzie oparty na metodach rozpoznawania obrazów i klasyfikacji wizyjnej przy użyciu technik głębokiego uczenia (ang. *deep learning*). 
Dokładniej, zastosowanie znajdą tutaj sieci konwolucyjne (CNN - *Convolutional Neural Networks*), które są obecnie jednymi z najskuteczniejszych narzędzi do analizy obrazów. 
W ramach systemu będzie zaimplementowany model uczony na dużych zbiorach zdjęć, który rozpozna różnorodne wzorce charakterystyczne dla chorób roślin.
