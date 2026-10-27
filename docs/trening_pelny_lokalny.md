# Pełny trening na lokalnym GPU

Skrypt: `src/train_full.py`. Pierwszy eksperyment:
`outputs/full_yolo26n_v1`. Model próbny i eksport Kaggle pozostają osobnymi artefaktami.

## Ustawienia

- Początkowe ogólne wagi `models/yolo26n.pt`, dwie klasy FREE/OCCUPIED.
- 8198 zdjęć treningowych, 1729 walidacyjnych, rozłączne daty.
- Maksymalnie 30 epok, wcześniejsze zatrzymanie po 8 epokach bez poprawy walidacji.
- 640 px, batch 8, GPU 0, AMP, 2 procesy ładowania danych, bez cache obrazów w RAM.
- AdamW, lr0 0,001, warmup 3 epoki, mosaic wyłączony, scale 0,2, seed 42.
- Checkpoint po każdej epoce; najlepszy model wybierany według walidacji.
- `trainval.yaml` nie zawiera zbioru testowego; końcowy test pozostaje niewykorzystany.

Laptop musi pozostać włączony i nie przechodzić w uśpienie. Podłączenie zasilacza
pomaga utrzymać wydajność. Skrypt nie zmienia ustawień zasilania systemu.

## Podgląd uruchomionego eksperymentu

Z terminala w katalogu projektu:

```powershell
Get-Content outputs/full_yolo26n_v1/progress.json
Get-Content outputs/full_yolo26n_v1.stdout.log -Tail 20 -Wait
```

Ctrl+C w drugim poleceniu zatrzymuje tylko podgląd logu, nie proces treningowy.
Osobny log diagnostyczny: `outputs/full_yolo26n_v1.stderr.log`.

`progress.json` zawiera stan (`initializing`, `training`, `completed`, `failed`
lub `interrupted`), PID procesu Pythona, ukończone epoki, metryki i szacowany
czas pozostały na podstawie ostatnich epok. Szacunek nie uwzględnia możliwego
wcześniejszego zatrzymania. Przy końcowej walidacji może wskazywać zero przed
całkowitym zakończeniem procesu; zakończenie potwierdza `state=completed`.

## Wyniki i wznowienie

- `weights/best.pt`: model wybrany według walidacji, do inferencji.
- `weights/last.pt`: ostatni checkpoint; przy przerwanym treningu umożliwia resume.
- `weights/epoch*.pt`: dodatkowe checkpointy epok.
- `results.csv`, `results.png`, macierze pomyłek i `validation_metrics.json`.
- `experiment.json`: odcisk manifestu danych i informacja o niewykorzystaniu testu.

Jeżeli trening się przerwie, najpierw upewnij się, że poprzedni proces już
nie działa. Następnie w tym samym katalogu projektu:

```powershell
uv run --frozen --no-sync python -m src.train_full --resume outputs/full_yolo26n_v1/weights/last.pt
```

Przy problemie z pamięcią można dodać `--batch 4`. Wznowienie wymaga checkpointu
z zachowanym stanem optymalizatora; finalny checkpoint po zakończonym treningu
nie służy do ścisłego resume. Gdy nie ukończono nawet jednej epoki, checkpoint
może jeszcze nie istnieć — nowy eksperyment wymaga nowego katalogu.

Nowy, niezależny trening w widocznym terminalu:

```powershell
uv run --frozen --no-sync python -m src.train_full --output outputs/full_yolo26n_v2
```

Nie uruchamiaj drugiego treningu równocześnie z aktywnym. Zwykłe ponowne
uruchomienie odmawia nadpisania istniejącego katalogu wyników. Przy resume
skrypt sprawdza zgodność manifestu danych z poprzednim eksperymentem.
