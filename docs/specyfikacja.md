# Specyfikacja projektu – Smart Parking Vision

**Przedmiot:** Big Data w Przemyśle 4.0  
**Status:** Wersja wstępna (v0.1)  
**Zespół:** do uzupełnienia  
**Data:** 2026-10-19

## 1. Temat projektu

**System wizyjnego wykrywania miejsc parkingowych oraz analizy ich zajętości z wykorzystaniem uczenia maszynowego**

## 2. Cel i motywacja

Celem projektu jest opracowanie systemu analizującego obraz z kamery umieszczonej nad parkingiem. System ma identyfikować miejsca parkingowe, określać ich aktualny stan (wolne/zajęte) i prezentować wyniki w czytelnej formie. Dodatkowym celem jest rejestrowanie zmian zajętości w czasie i analizowanie zgromadzonych danych.

Ze względu na brak stałego dostępu do rzeczywistego parkingu rozwiązanie zostanie zweryfikowane na makiecie z modelami samochodów (np. Hot Wheels). Makieta stanowi **środowisko laboratoryjne**: wyniki testów na miniaturach nie oznaczają automatycznie takiej samej skuteczności w rzeczywistym ruchu parkingowym.

## 3. Założenia i ograniczenia

- Kamera ustawiona nieruchomo ponad makietą, z możliwie pełnym widokiem stanowisk parkingowych.
- Makieta: docelowo 6–12 wyraźnie oznaczonych miejsc; liczba i wymiary do ustalenia po wykonaniu makiety.
- Różne modele i kolory samochodów, zmienna liczba zaparkowanych aut, dodatkowe eksperymenty przy innym oświetleniu i niewielkiej zmianie kąta kamery.
- Jeden strumień wideo (kamera USB, telefon działający jako kamera lub plik wideo); przetwarzanie na lokalnym komputerze.
- Brak konieczności chmury, serwera i bazy rozproszonej w wersji podstawowej.
- Priorytetem jest poprawność rozpoznawania stanów, czytelna demonstracja i mierzalna ewaluacja.

## 4. Zakres funkcjonalny

### F1. Pozyskiwanie obrazu
- Odczyt obrazu na żywo z kamery oraz z pliku wideo lub zdjęcia.
- Możliwość wyboru źródła obrazu i podstawowej konfiguracji rozdzielczości.

### F2. Wykrywanie i identyfikacja miejsc
- **Cel końcowy:** automatyczna detekcja geometrii stanowisk na podstawie oznaczeń makiety (np. linii na podłożu).
- Przypisanie każdemu stanowisku stabilnego identyfikatora, np. `P01`, `P02`, ...
- Zapis wykrytych obszarów w konfiguracji i używanie ich na kolejnych klatkach wideo.
- **Tryb awaryjny / etap MVP:** ręczne zaznaczenie wielokątnych obszarów stanowisk, aby pozostałe funkcje nie zależały od stopnia dopracowania detekcji automatycznej. Tryb ręczny nie zastępuje celu automatycznej lokalizacji miejsc.

### F3. Klasyfikacja zajętości
- Analiza obrazu każdego stanowiska i przypisanie jednej z klas: `FREE`, `OCCUPIED`.
- Klasyfikator oparty na niewielkiej sieci CNN (np. MobileNetV3 z transfer learningiem).
- Zachowanie wyniku i pewności predykcji dla pojedynczego stanowiska.
- Wygładzanie czasowe wyników, aby wyeliminować migotanie stanów na kolejnych klatkach.

### F4. Wizualizacja
- Podgląd obrazu z nałożonymi konturami stanowisk oraz kolorami: zielony – wolne, czerwony – zajęte.
- Liczniki: wszystkie miejsca, wolne miejsca, zajęte miejsca; opcjonalnie procent zajętości.
- Czytelne wskazanie, gdy detekcja stanowiska lub klasyfikacja ma niską pewność.

### F5. Rejestracja i analiza danych
- Zapis stanu parkingu z datą i czasem (np. SQLite lub Parquet/CSV).
- Zapis **zdarzeń zmiany stanu** (`FREE` -> `OCCUPIED` oraz odwrotnie) bez konieczności archiwizowania każdej identycznej klatki.
- Obliczanie zagregowanych statystyk, np. średnia zajętość w czasie, czas zajęcia stanowiska, liczba zmian stanu.
- Eksport danych i wygenerowanie przynajmniej jednego wykresu.

## 5. Architektura logiczna

```text
Kamera / plik wideo
        |
        v
Pozyskiwanie i wstępne przetwarzanie obrazu (OpenCV)
        |
        v
Detekcja stanowisk + identyfikatory (geometria / konfiguracja)
        |
        v
Wycinanie ROI poszczególnych stanowisk
        |
        v
Klasyfikator CNN: FREE / OCCUPIED
        |
        v
Wygładzanie predykcji w czasie
        |
        +----> Podgląd i liczniki
        |
        +----> Dziennik zdarzeń -> analiza statystyczna / wykresy
```

### Warianty metodyczne

| Obszar | Wariant podstawowy / kontrolny | Wariant docelowy / badawczy |
|---|---|---|
| Lokalizacja stanowisk | Ręczna kalibracja obszarów ROI | Automatyczne wykrywanie oznaczonych miejsc (OpenCV: progowanie + detekcja linii/konturów; w razie potrzeby segmentacja) |
| Ocena zajętości | Klasyczna analiza różnic obrazu wobec pustej makiety | CNN trenowana/dostrajana na próbkach ze stanowisk |
| Zmiany w czasie | Stan na pojedynczej klatce | Wygładzanie oraz zapis zdarzeń |

## 6. Zbiór danych i trening

1. Zbudować makietę i ustabilizować kamerę (statyw/uchwyt, stała odległość, oświetlenie).
2. Nagrać / wykonać fotografie dla wielu **niezależnych konfiguracji** samochodów, miejsc i warunków oświetlenia. Startowe założenie organizacyjne: około 200–500 zróżnicowanych scen, a nie setki niemal identycznych klatek jednego filmu.
3. Oznaczyć obszary stanowisk oraz ich rzeczywisty stan (`FREE`/`OCCUPIED`). Każda klatka z kilkoma stanowiskami dostarczy wielu wycinków (ROI), ale nie są one w pełni niezależnymi obserwacjami.
4. Rozdzielić dane na treningowe, walidacyjne i testowe **według sesji nagraniowych / konfiguracji scen**, a nie losowo po pojedynczych sąsiednich klatkach wideo; zapobiega to przeciekowi danych.
5. Przygotować pipeline treningowy z augmentacją (jasność, kontrast, delikatne przesunięcia/obroty), zapisem wersji modelu, parametrów i metryk.
6. Wybrać model na walidacji i ocenić **jednorazowo** na wydzielonym zbiorze testowym.

**Uwaga:** modele trenowane na prawdziwych samochodach lub zbiory PKLot / CNRPark mogą stanowić inspirację albo dodatkową próbę, lecz nie są bezpośrednim zamiennikiem próbek Hot Wheels ze względu na różnicę domeny.

## 7. Proponowane technologie

- **Python 3.11+** – implementacja i eksperymenty.
- **OpenCV** – obsługa kamery, transformacje obrazu, kontury, podgląd.
- **PyTorch + torchvision** – trening i inferencja CNN; alternatywnie TensorFlow/Keras.
- **NumPy, pandas, scikit-learn** – analiza danych, metryki, operacje numeryczne.
- **SQLite** – lekki lokalny magazyn zdarzeń; opcjonalnie Parquet / DuckDB do analiz.
- **Streamlit** lub prosty interfejs OpenCV – demonstracja na żywo (wybór po weryfikacji wydajności).
- **Matplotlib** – wykresy i dokumentacja wyników.

Technologie stanowią propozycję, a nie sztywny wymóg. Użycie Apache Spark i narzędzi rozproszonych **nie jest uzasadnione samą małą makietą**; można je wprowadzić wyłącznie jako świadomy eksperyment skalowania danych.

## 8. Kryteria akceptacji i ewaluacja

### Kryteria minimalne
- Działający podgląd z kamery lub nagrania.
- Co najmniej 6 miejsc zidentyfikowanych w obrazie (na etapie MVP dopuszczalna kalibracja ręczna).
- Klasyfikacja `FREE`/`OCCUPIED` przy użyciu rzeczywiście wytrenowanego/dostrojonego modelu ML.
- Widoczny wynik dla każdego miejsca, liczniki globalne, zapis zdarzeń.
- Raport eksperymentów na wydzielonym zbiorze testowym.

### Kryteria docelowe
- Automatyczne zlokalizowanie co najmniej 6 oznaczonych stanowisk na pustej makiecie i zachowanie konfiguracji na późniejsze klatki.
- Raport `accuracy`, `precision`, `recall`, `F1` oraz macierzy pomyłek zajętości; osobna ocena skuteczności detekcji miejsc (np. odsetek poprawnie znalezionych stanowisk i IoU polygonów).
- Zmierzenie liczby przetwarzanych klatek na sekundę oraz opóźnienia aktualizacji stanu.
- Porównanie bazowego algorytmu klasycznego z CNN na tym samym zbiorze testowym.
- Test w zmienionym oświetleniu i teście niewidzianych wcześniej konfiguracji.

## 11. Etapy prac

| Etap | Główny rezultat |
|---|---|
| 1. Koncepcja i repozytorium | Specyfikacja, struktura katalogów |
| 2. Makieta i kamera | Stabilny obraz 6–12 stanowisk |
| 3. Dataset | Oznaczone próbki i niezależne podziały danych |
| 4. MVP wizyjne | Kalibracja ROI, prosty baseline zajętości, wizualizacja |
| 5. Model AI | Trening/dostrojenie CNN, wyniki walidacji i testów |
| 6. Automatyczne miejsca | Detektor geometrii miejsc, zapis kalibracji |
| 7. Integracja i analityka | Podgląd w czasie rzeczywistym, log zdarzeń, agregacje |
| 8. Badania i oddanie | Porównania, testy, demonstracja i sprawozdanie |

