# Diagnostyka modeli parkingowych — 25.10.2026

Pełny model jest zdecydowanie lepszy od modelu próbnego na walidacji PKLot. Nie przenosi jednak tej jakości na dwa zdjęcia spoza zbioru. Sama zmiana progu pewności lub rozdzielczości nie naprawia tego problemu.

Komplet wyników i plansz: [raport HTML](../outputs/diagnostics_v2/index.html), [dane JSON](../outputs/diagnostics_v2/report.json).

## Co zostało sprawdzone

- Oba zapisane modele `best.pt`, bez ponownego treningu: smoke (3 epoki, 540 zdjęć treningowych) i full (najlepsza epoka 5, trening na 8198 zdjęciach).
- Te same **1729 zdjęć walidacyjnych** dla obu modeli przy wejściu 640.
- Te same **135 zdjęć walidacyjnych** (istniejąca próbka modelu smoke, po 15 z każdej pary kamera/pogoda) dla obu modeli przy 640, 960 i 1280. Dodatkowo pełny model przy 160 i 320.
- Dwa własne zdjęcia: oba modele przy 160, 320, 640, 960, 1280. Bez ręcznych etykiet referencyjnych — ocena wizualna, bez deklarowania dokładności procentowej.
- Progi confidence: 0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75.
- CUDA, FP32, kwadratowy letterbox (`rect=False`), `max_det=300`, batch 4 na walidacji. Filtrowanie progami odbywa się na zapisanych predykcjach z confidence 0.01.
- **Zbiór testowy nie był używany.** Nie zmieniono modeli, źródłowych etykiet ani konfiguracji istniejących treningów. Sprawdzono SHA-256 obu modeli i manifestu przed/po głównym przebiegu.

Jeden przebieg smoke/960 przekroczył domyślny limit czasu NMS, co może pozostawić nieprzetworzone zdjęcia z pustymi predykcjami. Cały ten przebieg powtórzono z większym budżetem czasu (10 s/obraz plus narzut biblioteki); powtórka nie zgłosiła timeoutu. Raport zawiera poprawione wyniki, a pierwsze zapisano jako `smoke_960_before_retry_*`. Pozostałe przebiegi nie zgłosiły timeoutu. Zmiana budżetu dotyczyła tylko procesu diagnostycznego, nie plików biblioteki ani modelu.

## Liczenie miejsc: porównanie na pełnej walidacji

MAE na klasę to średnia z bezwzględnych błędów liczby FREE i OCCUPIED: `sum(|pred_free-GT_free| + |pred_occupied-GT_occupied|)/(2*N)`. Dzięki temu błędy przeciwnych znaków nie znoszą się. Jednostką jest liczba miejsc, nie procent.

| Model / wejście | Próg | Średni błąd na klasę |
|---|---:|---:|
| Próbny / 640 | 0.25 | 11.66 |
| Próbny / 640 | 0.40 — najlepszy z badanych | 4.72 |
| Pełny / 640 | 0.25 | 0.68 |
| Pełny / 640 | 0.60 — najlepszy z badanych | **0.30** |

Pełny model przy 640 i confidence 0.60:

- FREE MAE: **0.268** miejsca/zdjęcie.
- OCCUPIED MAE: **0.334** miejsca/zdjęcie.
- MAE sumy miejsc: **0.434** miejsca/zdjęcie.
- Oba liczniki dokładnie zgodne z etykietami na **76.52%** zdjęć. Nie oznacza to, że wszystkie ramki na tych zdjęciach są poprawne.
- Najtrudniejsza grupa: UFPR04/Sunny, MAE na klasę **1.28**; pozostałe grupy **0.02–0.41**.

To wyniki na walidacji używanej do wyboru ustawień, a nie niezależna ocena końcowa. Kamera i parking są znane z treningu; rozdzielone są daty.

## Rozdzielczość: wspólna próbka 135 zdjęć

Pełny model, ten sam próg 0.25:

| Wejście | MAE na klasę |
|---:|---:|
| 160 | 27.82 |
| 320 | 18.25 |
| **640** | **0.93** |
| 960 | 7.28 |
| 1280 | 6.74 |

Wartości 0.93 i 0.68 dotyczą różnych rozmiarów zbioru: odpowiednio 135 i 1729 zdjęć. Nie należy mieszać tych porównań.

Nawet po oddzielnym wyborze najlepszego badanego progu na tej próbce wejście 640 wygrywa: MAE 0.57, wobec 4.40 dla 960 i 5.40 dla 1280. Dla próbnego modelu analogiczne minima wynoszą 4.39, 9.51 i 10.24. Wyższa rozdzielczość nie jest automatycznym ulepszeniem modelu uczonego przy stałej skali.

## Dwa własne zdjęcia

Przy confidence 0.25 pełny model zwraca:

| Wejście | parking1: FREE / OCCUPIED | parking2: FREE / OCCUPIED |
|---:|---:|---:|
| 160 | 0 / 0 | 0 / 0 |
| 320 | 0 / 9 | 1 / 2 |
| 640 | 6 / 25 | 3 / 1 |
| 960 | 19 / 25 | 17 / 5 |
| 1280 | 23 / 25 | 11 / 1 |

To **liczby wykryć, nie pomiar jakości**. Na parking1 model pomija wiele miejsc. Na parking2 ramki obejmują m.in. dachy i fragmenty aut, a klasa FREE jest często błędna. Obniżenie progu dodaje ramki, ale nie rozwiązuje problemu: na parking2/full/640/conf=0.01 występuje 35 FREE i 16 OCCUPIED, nadal z błędną geometrią i klasyfikacją. Podniesienie progu do 0.60 usuwa wszystkie wykrycia na parking2.

Model próbny przy 160/320 rozpoznaje część aut na parking2 lepiej wizualnie niż model pełny, lecz nadal nie daje kompletnego, wiarygodnego wyniku miejsc parkingowych. To pokazuje, że specjalizacja modelu i skala mają znaczenie; nie uzasadnia powrotu do modelu próbnego dla PKLot.

[Porównanie parking1](../outputs/diagnostics_v2/parking1_models.jpg) · [parking2](../outputs/diagnostics_v2/parking2_models.jpg) · [mniejsze wejścia na parking2](../outputs/diagnostics_v2/parking2_small_inputs.jpg) · [wpływ progu na parking2](../outputs/diagnostics_v2/parking2_thresholds.jpg).

## Znaleziony problem w etykietach źródłowych

W obrazie `PUCPR/Rainy/2012-10-11/2012-10-11_15_51_52.jpg` oryginalny XML podaje 64 wolne i 36 zajętych miejsc. Na [planszy wycinków](../outputs/diagnostics_v2/annotation_disagreements.jpg) widać osiem miejsc z samochodami, oznaczonych w XML jako FREE: identyfikatory 17, 13, 77, 86, 59, 62, 65, 8. Model wskazuje OCCUPIED z confidence około 0.88–0.90.

W tym obrazie diagnostyka znalazła 34 dopasowane ramki z różnicą klasy przy confidence 0.25; pokazane osiem sprawdzono wzrokowo. Nie zakładamy automatycznie, że wszystkie rozbieżności modelu z etykietą są błędami etykiet.

Dla trzech najgorszych przykładów (po jednym na kamerę) ponownie przeliczono etykiety z oryginalnego XML i porównano je z plikami YOLO, a także sprawdzono identyczność obrazów przez SHA-256. Wszystkie porównania przeszły. **Podejrzane klasy występują w danych źródłowych; nie powstały przy konwersji.** Żadnych etykiet nie zmieniono i żadnego trudnego zdjęcia nie usunięto z raportowanych metryk.

[Audyt zgodności XML](../outputs/diagnostics_v2/annotation_audit.json) · [najgorsze przykłady pełnego modelu](../outputs/diagnostics_v2/full_640_worst.jpg).

PKLot dodatkowo oznacza tylko wybrane miejsca w kadrze. Model może poprawnie wskazać widoczne miejsce spoza oznaczeń, a metryka potraktuje je jako nadmiarową predykcję. To ogranicza interpretację błędu względem całego widocznego parkingu.

## Wnioski i następny eksperyment

1. Zachować pełny model jako punkt odniesienia dla znanych ujęć PKLot. Kandydat do dalszej oceny: `best.pt`, `imgsz=640`, `conf=0.60`. Jest to minimum błędu zliczania w sprawdzonej siatce progów, nie uniwersalna konfiguracja dla dowolnego zdjęcia. Zapis: [candidate_settings.json](../outputs/diagnostics_v2/candidate_settings.json).
2. Nie wydłużać bez zmian tego samego treningu. Krzywe pokazują malejący błąd treningowy przy pogarszającej się lokalizacji na walidacji po najlepszej epoce; wraz z wynikami własnych zdjęć wskazuje to na ograniczoną zdolność przenoszenia na inne ujęcia.
3. Przed kolejnym treningiem przejrzeć oznaczenia treningowe i walidacyjne według jawnych reguł. Ewentualne korekty zapisać jako nową wersję danych, zachowując oryginały i raport zmian; nie usuwać zdjęć tylko dlatego, że model popełnia na nich błędy.
4. Następny kontrolowany trening może sprawdzić większe zróżnicowanie skali i geometrii przez augmentacje. Efekt jest hipotezą do zmierzenia, nie gwarancją poprawy. Porównać z obecnym modelem na tym samym protokole. Nie używać dwóch własnych zdjęć jako jedynej podstawy wyboru modelu.
5. Jeśli wymaganiem są nowe parkingi/kamery, dodać eksperyment z całkowicie pominiętą w treningu kamerą, korzystając wyłącznie z obecnej części train/val. Dopiero to oraz szerszy, oznaczony zbiór zdjęć zewnętrznych pozwolą ocenić potrzebę kolejnego datasetu. Sam podział według dat nie mierzy tego wymagania.
6. Dopiero po ustaleniu modelu, zakresu zastosowania i ustawień przeprowadzić końcowy test na odłożonych 1568 zdjęciach. Osobno później mierzyć wydajność i eksport pod analizę urządzeń o małej mocy; diagnostyka nie mierzyła Raspberry Pi.

## Metoda i odtwarzanie

Główny skrypt: `src/diagnose_models.py`. Z katalogu głównego repozytorium:

```powershell
uv run --frozen --no-sync python -u src/diagnose_models.py --output outputs/diagnostics_v3
uv run --frozen --no-sync python -m pytest tests/test_diagnose_models.py -q
```

Katalog wyjściowy musi być nowy — skrypt nie nadpisuje istniejącego przebiegu. Dodatkowe jednorazowe analizy tego raportu zachowano w `outputs/diagnostics_v2/finish_diagnostics.py` i `check_smaller_inputs.py`; zawierają jawne ścieżki do tego przebiegu i nie są ogólnymi komendami do wielokrotnego uruchamiania. W szczególności pierwsza odmawia powtórnego nadpisania kopii sprzed retry.

Testy jednostkowe: **5/5**. Obejmują podwójne ramki, pomyłkę klasy, puste predykcje, niedopuszczenie do znoszenia błędów liczników i odrzucenie zdjęcia testowego w próbce walidacyjnej.

Pola `diagnostic_precision`, `diagnostic_recall` i rozbicie pomyłek w JSON używają zachłannego dopasowania 1:1 według confidence, niezależnie od klasy, przy IoU >= 0.5. Zła klasa jest następnie liczona osobno. To pomocnicza analiza błędów, **nie mAP i nie implementacja metryk walidatora Ultralytics**. Progi porównano po filtrowaniu wspólnego wyniku NMS; limit 300 może ograniczać liczbę bardzo słabych wykryć, szczególnie modelu próbnego przy 0.01.

Czasy z raportu nie są miarodajnym benchmarkiem sprzętu: rozruch, obciążenie GPU innymi procesami oraz różna liczba ramek wpływały na wykonanie.
