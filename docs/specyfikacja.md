# Specyfikacja projektu – Smart Parking Vision

**Przedmiot:** Big Data w Przemyśle 4.0  
**Status:** Wersja wstępna (v0.2)  
**Zespół:** Julia Rojek i Mateusz Zubrzycki  
**Data:** 2026-10-19

## 1. Temat projektu

**System wizyjnego wykrywania miejsc parkingowych oraz analizy ich zajętości z wykorzystaniem uczenia maszynowego**

## 2. Cel i motywacja

Celem projektu jest opracowanie systemu analizującego obrazy parkingów rejestrowane przez kamery. System ma identyfikować miejsca parkingowe, określać ich stan (wolne/zajęte) i prezentować wyniki w czytelnej formie. Dodatkowym celem jest rejestrowanie wyników analiz oraz badanie zmian zajętości na podstawie danych zawierających informacje o czasie wykonania zdjęć.

Ze względu na brak stałego dostępu do rzeczywistego parkingu projekt będzie realizowany i weryfikowany **na publicznie dostępnych, oznaczonych zbiorach danych przedstawiających rzeczywiste parkingi**. Rozważane źródła to **PKLot** oraz **CNRPark+EXT**. Pozwala to wykorzystać fotografie wykonane w różnych warunkach oświetleniowych, atmosferycznych i z różnych perspektyw kamer bez konieczności budowania własnego stanowiska pomiarowego.

## 3. Założenia i ograniczenia

- Podstawowym źródłem danych są zdjęcia rzeczywistych parkingów wraz z dostępnymi adnotacjami dotyczącymi położenia stanowisk i ich zajętości.
- Wybór głównego zbioru danych (PKLot lub CNRPark+EXT) nastąpi po analizie formatu adnotacji, dostępności pełnych ujęć i wymagań eksperymentów. Możliwe jest wykorzystanie drugiego zbioru do niezależnych testów.
- Wersja podstawowa przetwarza pojedyncze obrazy lub uporządkowane serie zdjęć; obsługa plików wideo lub kamery na żywo jest rozszerzeniem i nie jest niezbędna do przeprowadzenia badań.
- Przetwarzanie i trening modeli odbywają się na lokalnym komputerze; użycie chmury i infrastruktury rozproszonej nie jest wymagane w wersji podstawowej.
- Model powinien uwzględniać zróżnicowane warunki pogodowe, oświetlenie, perspektywę i częściowe przesłonięcia pojazdów obecne w danych.
- Priorytetem są poprawność rozpoznawania zajętości, mierzalna ewaluacja oraz możliwość demonstracji wyników na pełnym obrazie parkingu.
- Zbiory zawierają przede wszystkim fotografie wykonywane w określonych odstępach, a nie koniecznie ciągłe nagrania. Nie wolno utożsamiać zmian między kolejnymi zdjęciami ze wszystkimi rzeczywistymi zdarzeniami wjazdu i wyjazdu.
- Automatyczna detekcja geometrii miejsc jest **odrębnym zadaniem** od klasyfikacji ich zajętości. Dostępne adnotacje stanowisk mogą służyć jako dane referencyjne oraz rozwiązanie bazowe.

## 4. Zakres funkcjonalny

### F1. Pozyskiwanie obrazu

- Wczytywanie pełnych zdjęć parkingów z wybranego zbioru danych.
- Obsługa odpowiadających im adnotacji oraz metadanych (np. identyfikator kamery, data i czas rejestracji, warunki pogodowe – jeśli są dostępne).
- Wybór zdjęcia, zestawu zdjęć lub serii pochodzącej z jednej kamery.
- **Rozszerzenie:** odczyt obrazu z pliku wideo lub kamery na żywo.

### F2. Wykrywanie i identyfikacja miejsc

- **Cel końcowy:** automatyczna lokalizacja stanowisk parkingowych na pełnym obrazie za pomocą algorytmów przetwarzania obrazu lub modelu uczenia maszynowego.
- Przypisanie stanowiskom identyfikatorów, np. `P01`, `P02`, ...; stabilność identyfikatorów jest wymagana w obrębie danej kamery/konfiguracji parkingu, nie pomiędzy różnymi parkingami.
- Obsługa obszarów stanowisk w postaci wielokątów lub prostokątów, zależnie od adnotacji i zastosowanej metody.
- Zapis i ponowne wykorzystanie konfiguracji stanowisk dla tego samego, nieruchomego ujęcia kamery.
- **Tryb bazowy / etap MVP:** wykorzystanie lokalizacji stanowisk zapisanych w adnotacjach zbioru lub ich ręczne wskazanie. Tryb ten umożliwia niezależną ocenę klasyfikatora zajętości, ale nie stanowi realizacji automatycznej detekcji.

### F3. Klasyfikacja zajętości

- Analiza obrazu każdego stanowiska i przypisanie jednej z klas: `FREE`, `OCCUPIED`.
- Zastosowanie klasyfikatora CNN, np. MobileNetV3 z transfer learningiem.
- Zapis przewidywanej klasy oraz pewności predykcji dla każdego miejsca.
- **Rozszerzenie dla serii czasowych/wideo:** wygładzanie przewidywań między kolejnymi obserwacjami tego samego stanowiska.

### F4. Wizualizacja

- Wyświetlanie zdjęcia parkingu z nałożonymi konturami stanowisk: zielony – wolne, czerwony – zajęte.
- Liczniki: wszystkie rozpoznane miejsca, wolne miejsca, zajęte miejsca; opcjonalnie procent zajętości.
- Możliwość przełączania zdjęć i porównania predykcji z adnotacjami referencyjnymi.
- Czytelne zaznaczenie przypadków o niskiej pewności klasyfikacji lub niepoprawnie wykrytych obszarów.

### F5. Rejestracja i analiza danych

- Zapis wyników analizy: źródło obrazu, identyfikator kamery/parkingu, identyfikator stanowiska, przewidywany stan i prawdopodobieństwo; data/czas rejestracji, jeśli są dostępne.
- Agregowanie statystyk, np. procent zajętych miejsc w danym ujęciu, średnia zajętość dla kamery, wyniki według warunków pogodowych lub pory dnia.
- **Dla uporządkowanych czasowo obrazów tego samego stanowiska:** wykrywanie zaobserwowanych zmian `FREE → OCCUPIED` i `OCCUPIED → FREE` oraz szacowanie czasu utrzymywania się stanu między obserwacjami, z wyraźnym zaznaczeniem ograniczeń częstotliwości próbkowania.
- Eksport wyników do CSV/Parquet lub zapis w SQLite i wygenerowanie co najmniej jednego wykresu.
- Analiza wydajności przetwarzania, np. czas analizy zdjęcia, przepustowość w obrazach na sekundę oraz czas inferencji modelu.

## 5. Architektura logiczna

```text
Publiczny zbiór danych (PKLot / CNRPark+EXT)
       |
       v
Wczytywanie pełnych obrazów, adnotacji i metadanych
       |
       v
Wstępne przetwarzanie obrazu (OpenCV)
       |
       v
Lokalizacja stanowisk:
  adnotacje / konfiguracja (MVP) lub detekcja automatyczna
       |
       v
Wycinanie ROI poszczególnych stanowisk
       |
       v
Klasyfikator CNN: FREE / OCCUPIED
       |
       v
Przypisanie wyników do stanowisk i obrazu źródłowego
       |
       +----> Wizualizacja obrazu i liczniki zajętości
       |
       +----> Zapis predykcji i metryk
                    |
                    v
             Agregacja i wykresy

```

### Warianty metodyczne

| Obszar | Wariant podstawowy / kontrolny | Wariant docelowy / badawczy |
|---|---|---|
| Lokalizacja stanowisk | Wykorzystanie obszarów z adnotacji lub ręczna kalibracja ROI | Automatyczne wykrywanie lokalizacji miejsc na pełnym obrazie (OpenCV i/lub model detekcji/segmentacji) |
| Ocena zajętości | Prosty klasyfikator bazowy, np. cechy obrazu + klasyfikator ML | CNN trenowana lub dostrajana na oznaczonych wycinkach miejsc |
| Zmiany w czasie | Agregacja zajętości dla niezależnych zdjęć | Analiza zmian między chronologicznie uporządkowanymi obserwacjami tego samego miejsca |

## 6. Zbiory danych i trening

### Rozważane zbiory danych

**Opcja A – PKLot**  
Strona projektu: [PKLot – Parking Lot Database](https://web.inf.ufpr.br/vri/databases/parking-lot-database/)

- Zbiór zawiera **12 417 pełnych zdjęć parkingów** i **695 899 oznaczonych wycinków miejsc parkingowych**.
- Dostępne są oznaczenia zajętości (`FREE`/`OCCUPIED`) oraz położenia stanowisk na pełnych obrazach (m.in. adnotacje XML).
- Zdjęcia przedstawiają parkingi obserwowane przez nieruchome kamery w różnych warunkach pogodowych.
- Jest szczególnie przydatny do testowania kompletnego przepływu: od obrazu całego parkingu, przez lokalizację stanowisk, po ocenę ich zajętości.

**Opcja B – CNRPark+EXT**  
Strona projektu: [CNRPark+EXT – Dataset for Visual Occupancy Detection](https://cnrpark.it/)

- Zbiór zawiera **około 150 tys. oznaczonych wycinków stanowisk** z klasami `free` i `busy`.
- Obejmuje różne kamery, perspektywy, warunki pogodowe i oświetleniowe, w tym trudne przypadki częściowego zasłonięcia aut.
- Oprócz gotowych wycinków dostępne są **pełne ujęcia dla części CNR-EXT** oraz pliki z lokalizacjami stanowisk; przy ich łączeniu należy uwzględnić skalowanie współrzędnych do rozdzielczości udostępnionych zdjęć.
- Szczególnie dobrze nadaje się do treningu i oceny klasyfikatora zajętości oraz eksperymentów z uogólnianiem między kamerami.

**Wybór zbioru:** oba zbiory są alternatywnymi źródłami danych. Na początek wystarczy wybrać **jeden zbiór główny**; drugi można później wykorzystać jako niezależny test generalizacji. Przed ostatecznym wyborem należy zweryfikować dostępność plików, format adnotacji, wielkość pobierania i warunki licencyjne.

### Przygotowanie danych i trening modelu

1. Pobrać wybrany zbiór, przeanalizować jego strukturę oraz przygotować skrypt wczytujący obrazy, etykiety i położenia stanowisk.
2. Sprawdzić jakość adnotacji oraz przygotować wycinki ROI przypisane do klas `FREE` i `OCCUPIED`. Tam, gdzie są dostępne gotowe wycinki, można wykorzystać je bez ponownego wycinania.
3. Przygotować podziały treningowy, walidacyjny i testowy **na poziomie scen, dni rejestracji lub kamer**, a nie przez losowe mieszanie wycinków z bardzo podobnych zdjęć. W miarę możliwości uwzględnić osobny test na niewidzianej kamerze/parkingu.
4. Zaimplementować prostą metodę bazową klasyfikacji zajętości, aby mieć punkt odniesienia dla CNN.
5. Przygotować pipeline treningowy CNN z augmentacją (jasność, kontrast, niewielkie przekształcenia geometryczne), zapisem parametrów, wersji modelu i metryk.
6. Wybrać model na podstawie walidacji i ocenić go **jednorazowo na wydzielonym zbiorze testowym**.
7. Zintegrować klasyfikator z przetwarzaniem pełnych obrazów: lokalizacją ROI, nanoszeniem kolorów i podsumowaniem zajętości.
8. **Etap badawczy:** opracować automatyczne wykrywanie lokalizacji miejsc i ocenić je oddzielnie od klasyfikacji, korzystając z dostępnych oznaczeń geometrii jako danych referencyjnych.
9. **Opcjonalnie:** przetestować model na drugim zbiorze (PKLot ↔ CNRPark+EXT), zwracając uwagę na różnice perspektywy, formatu adnotacji i charakterystyki danych.

**Uwaga o analizie czasowej:** statyczne zdjęcia wystarczają do treningu i oceny wykrywania zajętości. Rejestrowanie zmian stanu i szacowanie czasu parkowania jest możliwe wyłącznie wtedy, gdy da się wiarygodnie uporządkować obserwacje **tego samego stanowiska z tej samej kamery** według czasu rejestracji. Wyniki będą przybliżeniem opartym na odstępach między zdjęciami, a nie pełną historią rzeczywistych zdarzeń.

## 7. Proponowane technologie

- **Python 3.11+** – implementacja i eksperymenty.
- **OpenCV** – wczytywanie i przetwarzanie zdjęć, praca z ROI, kontury i wizualizacja.
- **PyTorch + torchvision** – trening i inferencja CNN; alternatywnie TensorFlow/Keras.
- **NumPy, pandas, scikit-learn** – przygotowanie danych, analiza statystyczna i metryki.
- **SQLite** – lokalne przechowywanie predykcji i wyników; opcjonalnie Parquet / DuckDB do analiz większych zbiorów.
- **Streamlit** lub prosty interfejs OpenCV – demonstracja wyników dla zdjęć i serii obrazów.
- **Matplotlib** – wykresy i prezentacja wyników badań.

Technologie stanowią propozycję, a nie sztywny wymóg. Publiczne zbiory umożliwiają eksperymenty na znacznie większej liczbie obserwacji niż niewielka własna makieta, ale **samo ich użycie nie uzasadnia automatycznie wdrażania Apache Spark ani infrastruktury rozproszonej**. Takie narzędzia można rozważyć wyłącznie w ramach odrębnego eksperymentu skalowania przetwarzania.

## 8. Etapy prac

| Etap | Główny rezultat |
|---|---|
| 1. Koncepcja i repozytorium | Specyfikacja, struktura katalogów |
| 2. Wybór i analiza danych | Wybór PKLot lub CNRPark+EXT, rozpoznanie formatu obrazów i adnotacji |
| 3. Przygotowanie datasetu | Import, przetworzenie etykiet i ROI, niezależne podziały danych |
| 4. MVP wizyjne | Praca na adnotowanych ROI, bazowa klasyfikacja, wizualizacja |
| 5. Model AI | Trening/dostrojenie CNN, wyniki walidacji i testów |
| 6. Automatyczne miejsca | Detekcja geometrii stanowisk na pełnych obrazach i ewaluacja |
| 7. Integracja i analityka | Wizualizacja całego parkingu, zapis wyników i agregacje |
| 8. Badania i oddanie | Porównania metod, testy, demonstracja i sprawozdanie |

## Literatura

[1] P. R. L. de Almeida, L. S. Oliveira, A. S. Britto Jr., E. J. Silva Jr., A. L. Koerich, *PKLot – A robust dataset for parking lot classification*, Expert Systems with Applications, t. 42, nr 11, s. 4937–4949, 2015. DOI: 10.1016/j.eswa.2015.02.009.

[2] G. Amato, F. Carrara, F. Falchi, C. Gennaro, C. Meghini, C. Vairo, *Deep learning for decentralized parking lot occupancy detection*, Expert Systems with Applications, t. 72, s. 327–334, 2017. DOI: 10.1016/j.eswa.2016.10.055. Rekord publikacji i dostęp do pełnego tekstu.

[3] A. Howard, M. Sandler, G. Chu, L.-C. Chen, B. Chen, M. Tan, W. Wang, Y. Zhu, R. Pang, V. Vasudevan, Q. V. Le, H. Adam, *Searching for MobileNetV3*, 2019 IEEE/CVF International Conference on Computer Vision (ICCV), s. 1314–1324, 2019. DOI: 10.1109/ICCV.2019.00140. Pełny tekst: https://arxiv.org/abs/1905.02244.

[4] J. Yosinski, J. Clune, Y. Bengio, H. Lipson, *How transferable are features in deep neural networks?*, Advances in Neural Information Processing Systems, t. 27, 2014. Pełny tekst: https://arxiv.org/abs/1411.1792.

[5] G. Amato, F. Carrara, F. Falchi, C. Gennaro, C. Vairo, *Car parking occupancy detection using smart camera networks and Deep Learning*, 2016 IEEE Symposium on Computers and Communication (ISCC), s. 1212–1217, 2016. DOI: 10.1109/ISCC.2016.7543901. Rekord publikacji i dostęp do pełnego tekstu.
