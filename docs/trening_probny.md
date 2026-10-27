# Trening próbny i założenie urządzeń o małej mocy

## Cel i odtworzenie próby

Próba sprawdza kompletny przepływ: dane YOLO → dostrajanie na GPU → zapis wag →
walidacja → predykcje pojedynczych zdjęć → pomiar inferencji na GPU i CPU.
To nie jest końcowy eksperyment oceniający skuteczność systemu.

```powershell
uv run --frozen --no-sync python -m src.train_smoke
```

Do powtórzenia użyj nowego katalogu, np. `--output outputs/smoke_yolo26n_repeat`.
Skrypt nie nadpisuje istniejących eksperymentów. Zależności pochodzą z istniejącego
środowiska; brakujące pakiety nie są automatycznie instalowane. Przy pierwszym
uruchomieniu pobierane są oficjalne wagi `yolo26n.pt` z Ultralytics do `models/`.

Parametry domyślne:

- YOLO26 Nano, transfer learning z wag ogólnych, dwie klasy FREE/OCCUPIED.
- 3 epoki, `imgsz=640`, batch 8, seed 42, AdamW, początkowy learning rate 0,001.
- GPU CUDA, FP32 (`amp=False`), 0 procesów roboczych ładowania danych na Windows.
- 540 zdjęć treningowych: po 60 z każdej pary ujęcie–pogoda.
- 135 zdjęć walidacyjnych: po 15 z każdej pary ujęcie–pogoda.
- Próbki są losowane powtarzalnie wyłącznie z wcześniej ustalonych train i val.
  Daty pozostają rozłączne. Lista testowa nie jest używana do treningu,
  walidacji, podglądów ani pomiarów czasu.
- Mosaic wyłączony; skala augmentacji 0,2. Pozostałe parametry treningu są
  zapisywane przez Ultralytics w `train/args.yaml`.

Wyniki trafiają do `outputs/smoke_yolo26n/`: `train/weights/best.pt`,
`train/results.csv`, wykresy treningu i macierze pomyłek, `predictions/` oraz
`report.json`. Podglądy używają progu confidence 0,25. Metryki walidacyjne
Ultralytics nie są tym samym co liczenie obiektów na tych dziewięciu podglądach.
Mała próbka walidacyjna jest celowo zrównoważona według ujęcia i pogody,
więc jej wyniki nie zastępują oceny na pełnej walidacji i końcowym teście.

## Pomiar inferencji

Po treningu wykonywane są osobne pomiary na GPU laptopa i CPU laptopa:
pojedyncze obrazy, FP32, wejście sieci 640 × 640, dziewięć zdjęć walidacyjnych,
trzy przebiegi pomiarowe po trzech wywołaniach rozgrzewkowych. Przed i po każdym
pomiarze GPU wykonywana jest synchronizacja CUDA. Raport zawiera medianę,
95. percentyl i średnią, liczbę wątków PyTorch oraz opis środowiska.

Obrazy są wcześniej wczytane do RAM. Pomiar obejmuje wywołanie `predict`,
w tym przygotowanie wejścia, obliczenia modelu i postprocessing; nie obejmuje
odczytu pliku, ładowania modelu, rysowania ani zapisu wyniku. Jest to pomiar
lokalny przy danym obciążeniu komputera, nie benchmark Raspberry Pi.

## Raspberry Pi jako analiza teoretyczna

Zgodnie z ustaleniem projekt nie wymaga fizycznego wdrożenia na Raspberry Pi.
Rozważamy wyłącznie **inferencję**, a trening odbywa się na laptopie z GPU.
W sprawozdaniu ocenimy możliwość uruchomienia modelu na urządzeniu o małej mocy,
oddzielając własne pomiary od danych publikowanych przez producenta narzędzi.

Ultralytics opisuje uruchomienie YOLO26 na Raspberry Pi 4/5 i eksport do NCNN,
formatu zoptymalizowanego m.in. dla procesorów ARM. W opublikowanym benchmarku
YOLO26n na Raspberry Pi 5 przy wejściu 640 i FP32 podano około **67 ms** samej
inferencji dla NCNN oraz **299 ms** dla PyTorch. Wartości nie obejmują
preprocessingu i postprocessingu. Dotyczą modelu i warunków z dokumentacji,
nie naszego modelu parkingowego ani zmierzonego działania całej aplikacji.
Źródło: [Ultralytics — Raspberry Pi, benchmark w wersji 8.4.108](https://docs.ultralytics.com/guides/raspberry-pi).

Wniosek projektowy: mały model Nano i analiza pojedynczych zdjęć stanowią
realistyczny kierunek dla urządzenia tej klasy, lecz osiągalny czas oraz jakość
wymagałyby pomiarów po eksporcie na konkretnym urządzeniu. Wyniku CPU x86 laptopa
nie wolno przeliczać bezpośrednio na wydajność CPU ARM.

Dalsze eksperymenty mogą porównać wejście 640 i 512 oraz rozmiar modelu i czas
inferencji. Mniejsza rozdzielczość może pogorszyć wykrywanie odległych stanowisk,
więc szybkość należy zestawiać z recall/mAP. Ewentualną kwantyzację należy
oceniać wraz ze zmianą jakości, bez zakładania automatycznej poprawy wydajności.

## Wynik lokalnej próby (2026-10-24)

Trening i walidacja końcowa zakończyły się poprawnie na RTX 5070 Laptop GPU,
PyTorch 2.14.1+cu130 i Ultralytics 8.4.187. Czas wywołania treningu z walidacją:
90,4 s. Checkpoint `outputs/smoke_yolo26n/train/weights/best.pt` ma 5 458 351
bajtów (5,46 MB dziesiętnie); odczytany model ma 2 534 372 parametrów.
Szczyt pamięci **zaalokowanej przez PyTorch** wyniósł 2,27 GiB; jest to inna
miara niż pamięć zarezerwowana lub całkowite użycie GPU przez proces.

Metryki finalnej walidacji najlepszego checkpointu na 135 zdjęciach:

| Metryka detekcji | Wynik |
|---|---:|
| Precision | 0,8626 |
| Recall | 0,8091 |
| mAP@50 | 0,8860 |
| mAP@50–95 | 0,4604 |

mAP@50–95 jest bardziej wymagające co do dopasowania ramek. Nie należy
przedstawiać mAP jako procentu poprawnie policzonych miejsc parkingowych.

| Inferencja lokalna, batch 1 | Mediana | P95 | Wątki PyTorch |
|---|---:|---:|---:|
| RTX 5070 Laptop GPU, FP32 | 11,36 ms | 12,68 ms | 4 |
| CPU laptopa, FP32 | 46,78 ms | 53,05 ms | 8 |

Ultralytics ustawiło 8 wątków przy inicjalizacji backendu CPU; raport zapisuje
rzeczywiste wartości. Pomiar to po 27 wywołań na urządzenie, z zakresem
opisanym wyżej. Nie jest to pomiar pamięci ani szybkości Raspberry Pi.

Wizualna kontrola dziewięciu przykładów ujawniła nadmiarowe i nakładające się
ramki. Przykładowo dla PUCPR/Cloudy przy progu 0,25 wykryto 150 miejsc wobec
100 opisanych w XML. Część predykcji leży poza obszarem adnotacji, ale występują
też duplikaty i błędne detekcje. Dlatego obecne liczniki nie są jeszcze
wiarygodnym wynikiem aplikacji. Potrzebny jest dłuższy trening, analiza
pełnej walidacji oraz dobór progu na walidacji przed końcowym testem.

Model mały i szybki na laptopie oraz opublikowane pomiary NCNN na RPi 5
uzasadniają **teoretyczną wykonalność** inferencji pojedynczych zdjęć na
urządzeniu tej klasy. Nie stanowią potwierdzenia szybkości ani skuteczności
naszego modelu na Raspberry Pi. W tej próbie nie wykonano eksportu NCNN.

Kontrole kodu: 36 testów zakończonych powodzeniem, w tym powtarzalność próbkowania,
reprezentacja grup ujęcie–pogoda i brak wspólnych dat train/val. Zbiór testowy
pozostał niewykorzystany.
