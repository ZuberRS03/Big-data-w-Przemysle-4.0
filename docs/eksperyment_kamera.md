# Eksperyment: kamera wyłączona z treningu

Cel: porównać dotychczasowe augmentacje z silniejszymi zmianami geometrii, bez uczenia na kamerze PUCPR. To eksperyment rozwojowy na dotychczasowych częściach train/val. Nie używa odłożonego zbioru testowego.

## Podział danych

| Rola | Dane | Liczba obrazów |
|---|---|---:|
| Trening | Oryginalny train, tylko UFPR04 i UFPR05 | 5314 |
| Dobór najlepszej epoki i progu confidence | Oryginalny val, tylko UFPR04 i UFPR05 | 1012 |
| Ocena na odłożonej kamerze po treningu | Oryginalny val, tylko PUCPR | 717 |

Zdjęcia PUCPR z oryginalnego train nie są używane. Żadne zdjęcie z oryginalnego test nie jest kopiowane ani odczytywane do treningu/oceny. Kamera PUCPR nie wpływa na early stopping ani dobór confidence. Po wykorzystaniu jej wyników do dalszych decyzji pozostaje ona zbiorem rozwojowym, a nie niezależnym testem końcowym.

Dane są kopiowane do `outputs/camera_experiment_v1/data`, a obrazy i etykiety sprawdzane przez SHA-256. Pozwala to odizolować także cache Ultralytics i ewentualne działania weryfikatora obrazów. Oryginalne etykiety, w tym udokumentowane podejrzane oznaczenia, pozostają bez zmian. Oba warianty korzystają z identycznych kopii. Nie dobieramy korekt etykiet do odpowiedzi modelu.

## Dwa warianty uruchamiane kolejno

| Ustawienie | baseline | augmented |
|---|---:|---:|
| scale, zakres odchylenia od 1 | 0.2 | 0.7 |
| degrees, maksymalny obrót w obie strony | 0 | 15 |
| translate, ułamek wymiaru | 0.1 | 0.15 |
| perspective | 0 | 0.0005 |

Silniejszy wariant jest eksperymentalny; nie zakładamy, że będzie lepszy. To porównanie pakietu augmentacji, a nie osobny pomiar wpływu każdej z czterech zmian.

Wspólne ustawienia: inicjalizacja z ogólnego `models/yolo26n.pt` (nie z poprzedniego modelu parkingowego, który widział PUCPR), seed 42, imgsz 640, batch 8, workers 2, AMP, AdamW, lr0=0.001, warmup 3, mosaic=0, maksymalnie 30 epok, patience=8. Checkpoint zapisywany co epokę. Osobny proces dla każdego wariantu zwalnia zasoby przed kolejnym treningiem. Wariant augmented uruchamia się automatycznie po zakończeniu treningu i oceny baseline. Błąd pierwszego wariantu zatrzymuje kolejkę.

Ocena każdego modelu:

1. Dobór progu z siatki 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75 na 1012 zdjęciach znanych kamer, według średniego bezwzględnego błędu liczby miejsc na klasę.
2. Zliczanie na 717 zdjęciach PUCPR przy wspólnym progu 0.25 oraz progu wybranym w punkcie 1. Nie wybieramy najlepszego progu na PUCPR.
3. Standardowe metryki detekcji Ultralytics na PUCPR. Wynik jest zapisywany jako `holdout`, mimo że API biblioteki używa nazwy `split=val`; nie jest to oryginalny zbiór testowy.

W diagnostyce/walidacji zwiększono wyłącznie limit czasu NMS, aby zimny lub obciążony GPU nie pozostawiał nieprzetworzonych obrazów w batchu. Nie zmienia to matematyki filtrowania. Podczas późniejszego porównania należy sprawdzić logi pod kątem timeoutów i błędów.

## Pliki i status

W `outputs/camera_experiment_v1`:

- `experiment.json`: podział, ustawienia, hashe chronionych plików.
- `progress.json`: stan kolejki, aktualny wariant, PID procesu nadrzędnego i procesu treningowego.
- `baseline/progress.json` i `augmented/progress.json`: etap, epoka, czas epoki, metryki i checkpointy.
- `baseline.stdout.log`, `baseline.stderr.log`, analogicznie dla augmented.
- `baseline/weights/best.pt`, `augmented/weights/best.pt`: nowe modele; istniejące modele pozostają w poprzednich katalogach.
- `training_complete.json` wewnątrz wariantu: zakończony trening, przed oceną.
- `evaluation.json`, `count_metrics.json`, `val_counts.csv`, `holdout_counts.csv`: wyniki oceny wariantu.
- `comparison.json`: zestawienie obu wariantów po zakończeniu całej kolejki.

## Uruchamianie i wznowienie

Z katalogu głównego repozytorium, nowy eksperyment:

```powershell
uv run --frozen --no-sync python -u -m src.train_camera_experiment --output outputs/camera_experiment_v2
```

Samo przygotowanie danych: dodaj `--prepare-only`. Katalog docelowy musi być nowy. Parametry użytego eksperymentu zostają zapisane w `experiment.json`.

Przygotowany lub przerwany eksperyment, **wyłącznie gdy poprzednia kolejka i jej proces treningowy już nie działają**:

```powershell
uv run --frozen --no-sync python -u -m src.train_camera_experiment --output outputs/camera_experiment_v1 --continue-run
```

Zakończone warianty są pomijane. Przerwany trening wznawia się z własnego `last.pt`, jeśli checkpoint zawiera stan optymalizatora; trening zakończony z zapisanym `training_complete.json` przechodzi od razu do oceny. Przerwanie przed pierwszym checkpointem wymaga nowego katalogu albo ręcznej diagnozy, zamiast automatycznego usuwania wyników. Nie uruchamiać drugiej kolejki na tym samym katalogu równolegle.

Podgląd w PowerShell:

```powershell
Get-Content outputs/camera_experiment_v1/progress.json
Get-Content outputs/camera_experiment_v1/baseline/progress.json
Get-Content outputs/camera_experiment_v1/baseline.stderr.log -Tail 15
```

Proces uruchomiony w tle nie zapobiega uśpieniu komputera. Zostawić komputer podłączony do zasilania i aktywny. Inne obciążenia GPU mogą znacząco wydłużyć uczenie.

## Granice wnioskowania

Jeden odłożony widok i jeden seed nie wystarczają do deklaracji działania na każdym parkingu. Jeśli augmented wygra, potrzebna będzie analiza błędów i dalsza weryfikacja przed końcowym testem. Dotychczasowy model pełny jest zachowany jako model dla znanych ujęć, ale jego wyniku na PUCPR nie należy przedstawiać jako uczciwej oceny nieznanej kamery: uczył się na tej kamerze.
