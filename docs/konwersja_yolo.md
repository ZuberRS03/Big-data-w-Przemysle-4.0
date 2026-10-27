# PKLot → YOLO detection

Uruchomienie z katalogu głównego istniejącego projektu:

```powershell
uv run --frozen --no-sync python -m src.convert_pklot
uv run --frozen --no-sync python -m pytest -q
```

Źródło: `data/PKLot/PKLot`. Wynik: `data/pklot_yolo_v1`.
Konwerter wymaga nowego katalogu docelowego i nie nadpisuje istniejącego datasetu.
Przy ponownym eksperymencie użyj np. `--output data/pklot_yolo_v2`.
Po przerwanym uruchomieniu pozostaje znacznik `INCOMPLETE`; taki zbiór nie jest
gotowy do treningu. Wybierz nowy katalog przy kolejnym uruchomieniu.

## Format i filtrowanie

- Klasy: `0 = FREE`, `1 = OCCUPIED`.
- Każdy kontur zamieniany jest na najmniejszy prostokąt równoległy do osi obrazu.
  Zapis: `class x_center y_center width height`, współrzędne znormalizowane do 0–1.
  To format zadania **detect**, nie segmentacji ani OBB.
- Osiem miejsc po przecinku; współrzędne spoza obrazu są odrzucane, nie przycinane.
- Odrzucane są całe zdjęcia z nieznanym statusem choć jednego miejsca, brakującym
  XML, błędnymi identyfikatorami, geometrią lub powtórzonymi etykietami wynikowymi.
  Powody znajdują się w `excluded.csv`.
- Obrazy są kopiowane bez skalowania i rekompresji. Nie używamy dowiązań, więc
  zmiana kopii nie zmieni źródła. Potrzebne jest dodatkowe miejsce na dysku.
- Konwerter odczytuje wymiary i sprawdza strukturę pliku obrazowego. Pełne
  dekodowanie wszystkich źródeł wykonuje wcześniejszy audyt `src.audit_pklot`.

Specyfikacja formatu: [Ultralytics – detection datasets](https://docs.ultralytics.com/datasets/detect/).

## Podział danych

Domyślnie wszystkie unikalne daty są sortowane, tasowane z `seed=42`, a następnie
przydzielane w przybliżeniu 70%/15%/15% do train/val/test. Proporcje odnoszą się do
**liczby dat**, nie zdjęć. Ten sam dzień zawsze pozostaje w jednym zbiorze, także
gdy występuje w różnych ujęciach lub katalogach pogody. Nie losujemy pojedynczych
zdjęć. To test nowych dni ze znanych ujęć; nie dowodzi generalizacji na nowy parking.

Daty katalogów są sprawdzane. Jeśli data katalogu jest niepoprawna, a nazwa zdjęcia
zawiera poprawną datę, używana jest ta druga, z zapisem w `date_corrections.csv`
i kolumnie `date_source` manifestu. Dotyczy to źródłowej literówki `2013-14-16`
w UFPR05 (zdjęcia z `2013-04-16`). Dwie poprawne, ale różne daty są odrzucane
jako niejednoznaczne. Katalogi źródłowe nie są zmieniane.

Parametry: `--seed`, `--val-fraction`, `--test-fraction`. `manifest.csv` zapisuje
przydział każdego obrazu, jego datę, ujęcie, pogodę i liczby obiektów każdej klasy.
`summary.json` zawiera rzeczywiste proporcje i rozkłady. Identyczne źródła,
parametry i wersja skryptu odtwarzają podział; zmiana zestawu dat może go zmienić.

Osobny eksperyment z całym ujęciem wyłącznie w teście:

```powershell
uv run --frozen --no-sync python -m src.convert_pklot --output data/pklot_yolo_holdout_pucpr --holdout-camera PUCPR
```

W tym trybie całe PUCPR trafia do testu; daty pozostałych ujęć są dzielone
85%/15% na train/val. `--test-fraction` nie jest wtedy używane do przydziału.
Daty testu mogą występować w pozostałych ujęciach, ale ujęcie testowe nie może
pojawić się w treningu ani walidacji. Taki eksperyment wymaga osobnego treningu
modelu, który wcześniej nie uczył się na PUCPR. UFPR04 i UFPR05 przedstawiają
ten sam parking z różnych ujęć, więc wydzielenie jednego z nich nie jest testem
całkowicie nowej lokalizacji.

## Pliki wynikowe

```text
data/pklot_yolo_v1/
├── dataset.yaml
├── train.txt, val.txt, test.txt
├── images/<ujęcie>/<pogoda>/<data>/*.jpg
├── labels/<ujęcie>/<pogoda>/<data>/*.txt
├── manifest.csv
├── excluded.csv
├── date_corrections.csv
├── summary.json
└── previews/
```

Listy obrazów używają ścieżek `./images/...` względem pliku listy.
`dataset.yaml` zawiera bezwzględną lokalną ścieżkę zbioru; po przeniesieniu
datasetu na inny komputer trzeba zmienić pole `path`. Cały wynik jest ignorowany
przez Git. Do przyszłego treningu należy przekazać `data=.../dataset.yaml`.
Sam konwerter nie pobiera modelu ani nie uruchamia treningu.

Podglądy są rysowane z **odczytanych etykiet YOLO**, aby sprawdzić zapis końcowy.
Zielony oznacza FREE, czerwony OCCUPIED. Przy perspektywie ramki są większe
od konturów i mogą się nakładać; to ograniczenie pierwszego wariantu detekcji.
Nie wszystkie widoczne stanowiska są oznaczone w PKLot, a wykluczanie
niekompletnych zdjęć może usuwać trudne przypadki. Te ograniczenia trzeba
uwzględnić w sprawozdaniu i interpretacji metryk.

## Wynik konwersji lokalnej (2026-10-24)

| Zbiór | Zdjęcia | Daty | FREE | OCCUPIED |
|---|---:|---:|---:|---:|
| train | 8198 | 68 | 263340 | 208616 |
| val | 1729 | 14 | 41857 | 63423 |
| test | 1568 | 14 | 37919 | 38121 |

Łącznie: **11 495 zdjęć i 653 276 ramek**; podział według dni, seed 42.
Każdy zbiór zawiera wszystkie trzy ujęcia, trzy rodzaje pogody i obie klasy.
Rzeczywiste rozkłady nie są identyczne — są zapisane w `summary.json`.

Wykluczono 922 zdjęcia. Powody (mogą się nakładać): brak statusów na 778
zdjęciach, brak XML dla jednego zdjęcia i konflikt dat dla 145 zdjęć.
W `PUCPR/Cloudy/2012-09-16` nazwy plików wskazują `2012-10-16`; te zdjęcia
zostały wykluczone jako niejednoznaczne. Dwa miały także brakujące statusy,
więc pula 11 638 zdjęć z audytu zmniejszyła się o 143.

Dla 52 włączonych zdjęć UFPR05 data została odzyskana z nazwy pliku zamiast
niepoprawnego katalogu `2013-14-16`. Pozostałe zdjęcie z tego katalogu zostało
wykluczone z powodu brakującego statusu.

Zestaw testów po implementacji konwertera: **34 testy zakończone powodzeniem**.
Końcowa kontrola objęła wszystkie 11 495 plików etykiet (653 276 ramek),
ich zakresy, liczby klas i rozłączność 96 dat. Listy obrazów zostały odczytane
przez loader zainstalowanego Ultralytics, a po jednej parze obraz–etykieta
z każdego ujęcia i zbioru (9 par) przeszło jego oficjalny walidator.
Raport kontroli: `outputs/yolo_conversion_validation.json`.
