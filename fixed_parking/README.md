# Smart Parking — konkretna kamera

Nowa, niezależna wersja projektu: **stałe miejsca parkingowe → wycinki → klasyfikator FREE/OCCUPIED**. Dotychczasowe skrypty, modele i raporty YOLO pozostają w swoich katalogach.

## Uruchomienie

Z głównego katalogu repozytorium:

```powershell
uv run --frozen --no-sync python -m fixed_parking.app
```

Otwórz [http://127.0.0.1:8765](http://127.0.0.1:8765). Jeśli aplikacja jest już uruchomiona na tym porcie, wystarczy otworzyć adres. Serwer działa tylko lokalnie. Analiza domyślnie korzysta z CPU; opcjonalnie dodaj `--device cuda`.

1. Wybierz przykład UFPR05 i kliknij **Wczytaj przykład**.
2. Kliknij **Analizuj parking**. Zobaczysz wolne/zajęte miejsca oraz wyniki dla 40 stanowisk.
3. **Pobierz wynik JPG** zapisuje obraz z obrysami. Każda analiza zapisuje także obraz, JSON wyników i użytą konfigurację w `outputs/fixed_parking_ufpr05_v1/analyses/`.
4. Konfigurację możesz wyeksportować przez **Zapisz miejsca JSON**, a następnie wczytać przy kolejnym zdjęciu tej samej kamery.

## Własna konfiguracja

Wczytaj zdjęcie. Dotychczasowe obrysy są wtedy usuwane z widoku, aby nie zastosować omyłkowo geometrii innej kamery. Zaimportuj właściwą konfigurację JSON lub kliknij **Dodaj miejsce**, zaznacz narożniki wokół miejsca w kolejności i wybierz **Zakończ obrys**. Obsługiwane są wypukłe wielokąty z 3–12 punktów. Każde miejsce ma unikalny identyfikator. Można anulować aktualny obrys i usuwać ostatnie miejsce.

Współrzędne są znormalizowane: ta sama konfiguracja działa po zmianie rozdzielczości bez zmiany proporcji i kadru. Inne proporcje są odrzucane. Samych przesunięć kamery aplikacja nie wykrywa automatycznie — operator musi sprawdzić pokrycie obrysów. Po przesunięciu lub obrocie kamery trzeba zmienić konfigurację.

**Model jest wytrenowany dla UFPR05.** Edytor pozwala przygotować inne parkingi, ale sama konfiguracja nie zapewnia jakości klasyfikacji na nowej kamerze. Dla nowego miejsca trzeba zebrać reprezentatywne oznaczone zdjęcia, sprawdzić model i ewentualnie go dostroić. Nie deklarujemy działania na dowolnym zdjęciu parkingu.

## Dane i wynik pierwszej wersji

- Parking UFPR05: 40 miejsc, identyczna geometria XML w sprawdzonych częściach train/val.
- Trening: 2897 zdjęć z 23 dni, **115880 wycinków**.
- Walidacja: 437 zdjęć z 4 innych dni, **17480 wycinków**.
- Zachowano dotychczasowy podział według dat; nie losowano wycinków pomiędzy częściami. Podczas treningu i doboru progu zbiór testowy nie był używany. Po zamrożeniu modelu wykonano końcową ocenę opisaną poniżej.
- Oryginalne obrazy i etykiety pozostają niezmienione. Geometria konfiguracji pochodzi z pierwszego zdjęcia treningowego i nie zawiera etykiet zajętości. Przed wycinaniem dla każdej klatki weryfikowana jest zgodność geometrii i identyfikatorów z konfiguracją.
- Wycinek jest maskowany do wielokąta miejsca i skalowany do 64×64 z zachowaniem proporcji. Tło i dopełnienie: RGB 127. Trening i aplikacja używają tej samej funkcji.
- CNN: 3 bloki Conv/BatchNorm/ReLU/Pool i mała głowica; **89426 parametrów**, checkpoint **365473 bajty** (około 0.35 MiB). Trening od zera, bez pobierania wag.
- 12 epok; najlepsza według ważonej straty walidacyjnej: 10. Przygotowanie danych, trening i utworzenie podglądów zajęły około **114 sekund** na tym komputerze.
- Próg OCCUPIED=0.60 wybrano na walidacji z siatki 0.10–0.90 co 0.05 według balanced accuracy, z błędem licznika jako rozstrzygnięciem remisów.

| Metryka walidacyjna wybranego modelu i progu | Wynik |
|---|---:|
| Poprawne klasyfikacje miejsc | 99.931% |
| Balanced accuracy | 99.937% |
| Wolne błędnie uznane za zajęte | 0 / 7992 |
| Zajęte błędnie uznane za wolne | 12 / 9488 |
| Średni bezwzględny błąd liczby wolnych miejsc / zdjęcie | 0.0275 |
| Zdjęcia z dokładnie poprawnym licznikiem | 97.254% |

To walidacja używana do wyboru modelu i progu, nie końcowa ocena niezależna. Liczba wszystkich miejsc jest zadana konfiguracją, więc błąd sumy FREE+OCCUPIED zawsze wynosi zero i nie jest użyteczną metryką. Poprawny licznik również nie dowodzi poprawności każdej klasyfikacji. Wyników klasyfikatora nie należy bezpośrednio porównywać z mAP detektora YOLO.

Trzy przykłady demonstracyjne wybrano deterministycznie jako środkowe zdjęcie w posortowanej grupie pogodowej, bez wybierania według jakości modelu. Wyniki: pochmurno 0/40, deszcz 1/39, słonecznie 40/0 (FREE/OCCUPIED), zgodne z referencją w tych trzech przykładach.

## CPU i Raspberry Pi

Na CPU tego komputera, 4 wątki, po rozgrzewce, 30 pomiarów po 40 miejsc:

- mediana: **19.36 ms**;
- 95. percentyl: **24.32 ms**.

Zakres pomiaru: wczytany wcześniej obraz RGB → wycinki → CNN → liczniki i obraz PIL z obrysami. Nie obejmuje ładowania modelu, dysku, kodowania JPEG ani przeglądarki. To pomiar desktopowego procesora, **nie Raspberry Pi**. Mały model stanowi podstawę do późniejszej analizy możliwości uruchomienia na słabszym sprzęcie; eksport i pomiary na RPi nie zostały wykonane.

## Pliki

Kod znajduje się w `fixed_parking/`, konfiguracja w `fixed_parking/configs/ufpr05.json`. Wyniki są w `outputs/fixed_parking_ufpr05_v1/`:

- `best.pt`, `last.pt`: nowy model;
- `layout.json`, `dataset.json`: geometria i pochodzenie danych;
- `report.json`, `history.json`, `val_probabilities.npy`: metryki i wyniki;
- `cpu_benchmark.json`: warunki i wyniki pomiaru CPU;
- `demo_*.jpg`, `demo_results.json`: przykłady;
- `train_crops.npy`, `val_crops.npy` oraz etykiety: cache przygotowanych wycinków (około 1.6 GB);
- `progress.json`: status treningu; `analyses/`: wyniki z aplikacji.

Do działania aplikacji nie są potrzebne tablice wycinków, ale zachowano je do odtwarzania eksperymentu. Wagi są lokalnymi plikami projektu, nie są pobierane z sieci.

## Pozostałe komendy

Nowy trening do osobnego katalogu (istniejące wyniki nie są nadpisywane):

```powershell
uv run --frozen --no-sync python -m fixed_parking.train --output outputs/fixed_parking_ufpr05_v2
```

Samodzielna analiza zdjęcia z poprawną konfiguracją, bez przeglądarki:

```powershell
uv run --frozen --no-sync python -m fixed_parking.predict --image "SCIEZKA_DO_ZDJECIA.jpg" --layout fixed_parking/configs/ufpr05.json --output outputs/moja_analiza_v1
```

Testy wycinania, walidacji geometrii, liczenia i rozdzielenia dat:

```powershell
uv run --frozen --no-sync python -m pytest tests/test_fixed_parking.py -q
```

## Końcowy test — 26.10.2026

Zakończono przegląd 12 rozbieżności walidacyjnych. Miejsca wyglądają na puste mimo etykiet OCCUPIED, miejscami z zasłonięciem przez sąsiednie auta. To podejrzenie niespójności etykiet, bez niezależnego potwierdzenia. Nie zmieniano oznaczeń ani modelu. Zamrożono próg 0.60, wagi, geometrię i kod; następnie oceniono wszystkie **680 zdjęć z 6 odłożonych dni UFPR05 (27200 miejsc)**.

| Metryka testowa, oryginalne etykiety XML | Wynik |
|---|---:|
| Poprawne klasyfikacje miejsc | **99.904%** |
| Balanced accuracy | 99.904% |
| Macro-F1 | 0.999042 |
| Wolne błędnie uznane za zajęte | 14 / 12983 |
| Zajęte błędnie uznane za wolne | 12 / 14217 |
| Średni bezwzględny błąd licznika wolnych miejsc | **0.0382** |
| Dokładny licznik | **655 / 680 zdjęć (96.324%)** |
| Maksymalny błąd licznika | 2 miejsca |

Raport z obrazami: [ocena końcowa HTML](../outputs/fixed_parking_ufpr05_v1/final_test/index.html). Podsumowanie do dokumentacji: [ocena UFPR05](../docs/ocena_koncowa_ufpr05.md).

Test dotyczy znanej kamery i nowych dni. Zdjęcia są czasowo powiązane. Rozkład pogody jest nierówny: 605 słonecznych, 72 pochmurne i tylko 3 deszczowe; nie jest to wystarczający test odporności na deszcz. Oryginalne metryki treningu z `test_set_used=false` opisują etap treningowy; aktualny stan końcowej oceny jest w `final_test/report.json` (`test_set_used=true`).

`final_test/frozen/` przechowuje kopię wag, geometrii i kodu użytego w ocenie. `frozen_protocol.json` zawiera hashe, `input_hashes.json` identyfikuje obrazy/XML. Wyniki każdej klatki i miejsca zapisano w CSV. Dane źródłowe i wcześniejsze modele YOLO pozostały niezmienione.

`report_display.json` wylicza balanced accuracy grup wyłącznie dla klas obecnych w referencji; dla brakującej klasy recall jest `null`. To korekta agregacji dla jednego dnia z samymi wolnymi miejscami, wykonana z zapisanej macierzy pomyłek. Surowy raport zachowano; metryki całego testu i predykcje nie zmieniły się.

Kod audytu i testu: `python -m fixed_parking.evaluate audit` oraz `python -m fixed_parking.evaluate test`. Oba polecenia odmawiają nadpisania istniejącego wyniku. Test jest już wykorzystany i nie należy traktować kolejnych eksperymentów na nim jako nowej niezależnej oceny. Można ponownie wygenerować sam raport bez inferencji:

```powershell
uv run --frozen --no-sync python -m fixed_parking.report
uv run --frozen --no-sync python -m pytest tests/test_fixed_parking.py tests/test_fixed_parking_evaluate.py -q
```

Treningu nie trzeba ponawiać. Następny etap projektu to demonstracja działania i opracowanie wyników; ewentualne dalsze strojenie wymaga nowego niezależnego testu.
