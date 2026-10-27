# Smart Parking Vision — uruchomienie na Kaggle

## Pliki do użycia

W `outputs/kaggle_export_v1/`:

- `smart_parking_dataset.zip` — wgraj jako nowy **prywatny Dataset**.
- `smart_parking_kaggle.ipynb` — zaimportuj jako **Notebook**.
- `export_report.json` — lokalny raport integralności i zachowania oryginałów.

ZIP zawiera wyłącznie wybrany dataset YOLO, metadane podziału, skrypt obsługi
Kaggle, oryginalne początkowe wagi YOLO26 Nano i informacje o źródłach.
Nie zawiera `.venv`, repozytorium Git, własnych zdjęć, danych PKLotSegmented
ani wyników lokalnego treningu próbnego. Dane są kopiowane, źródła nie są zmieniane.

## Pierwsze uruchomienie

1. Zaloguj się na Kaggle; w razie wymagania włącz dostęp do GPU przez weryfikację konta.
2. Utwórz Dataset, np. `smart-parking-pklot-yolo-v1`. Wgraj `smart_parking_dataset.zip`.
   Ustaw widoczność **Private**. Zachowaj `ATTRIBUTION.md` i informacje licencyjne.
   Paczka zawiera różne rodzaje zasobów: PKLot ma CC BY 4.0, a model/program
   Ultralytics ma osobne warunki. Nie przypisuj licencji PKLot automatycznie całej paczce.
3. Poczekaj na przetworzenie archiwum. W podglądzie powinien być katalog
   `smart_parking/` z `bundle.json`, `data/`, `code/` i `weights/`.
4. Utwórz Notebook i zaimportuj `smart_parking_kaggle.ipynb` (Upload/Import Notebook).
5. Dodaj utworzony Dataset przez **Add Input**. Nie dodawaj jednocześnie dwóch
   wersji paczki Smart Parking — notebook zatrzyma się przy niejednoznaczności.
6. W ustawieniach sesji włącz **GPU** oraz **Internet** dla instalacji bibliotek.
   Używany jest jeden GPU (`device=0`), także jeśli sesja udostępnia dwa.
7. Uruchom komórki instalacji, kontroli GPU i przygotowania danych. Dopiero po
   powodzeniu przejdź do treningu. Nie przenoś Windowsowego `.venv` ani `uv.lock`.
8. Do dłuższego uruchomienia użyj **Save Version → Save & Run All**. Notebook
   wykona wtedy cały przebieg; końcowy test pozostanie wyłączony.
9. Po zakończeniu pobierz z Output `smart_parking/parking_yolo26n_v1_results.zip`.
   Najważniejsze wagi są również w `smart_parking/runs/parking_yolo26n_v1/weights/best.pt`.

Nazwy elementów interfejsu mogą się różnić. Sprawdź bieżący limit GPU na koncie
przed startem. Nie zakładamy określonego typu karty ani identycznego czasu jak
na RTX 5070. Kaggle ma limity sesji i dostępności GPU.
Źródła: [Kaggle Notebooks](https://www.kaggle.com/docs/notebooks),
[wykorzystanie GPU](https://www.kaggle.com/docs/efficient-gpu-usage).

## Co robi notebook

- Weryfikuje SHA-256 plików i zachowuje dokładnie istniejący podział:
  **8198 train / 1729 val / 1568 test**. Nie losuje go ponownie.
- Kopiuje dane z wejścia tylko do odczytu do `/kaggle/working/smart_parking/data`.
  Generuje ścieżki Linux, bez odwołań do dysku E:.
- Instaluje `ultralytics==8.4.187`, zachowując wersje torch/torchvision z Kaggle.
  Pierwsza komórka wykonuje rzeczywiste obliczenie CUDA. Konflikt pakietów lub
  nieobsługiwana karta przerywają przygotowanie zamiast uruchamiać trening na CPU.
- Rozpoczyna nowy eksperyment z ogólnych wag `yolo26n.pt`. Nie nadpisuje ani
  nie kontynuuje automatycznie lokalnego modelu próbnego.
- Trenuje do 30 epok: Nano, 640 px, batch 8, AdamW, lr0 0,001, warmup 3 epoki,
  AMP, seed 42, mosaic wyłączony, scale 0,2; patience 8.
- Zapisuje najlepszy i ostatni model oraz checkpoint każdej epoki.
- Ocenia błędy liczników na walidacji dla czterech progów. Wybiera próg na
  walidacji, nie na teście. Znane ograniczenia niepełnych adnotacji PKLot pozostają.
- Finalny test jest osobną opcją `RUN_FINAL_TEST=False`. Włącz go po wyborze
  modelu i parametrów; nie używaj jego wyników do dalszego strojenia.

Źródło mechanizmu checkpointów i resume: [Ultralytics — trening](https://docs.ultralytics.com/modes/train/).

## Przerwanie, brak pamięci i wznowienie

- Brak pamięci GPU: zmniejsz `BATCH` do 4; jeśli run już istnieje, wybierz nową
  nazwę albo wznów zapisany checkpoint. Zmiana batcha jest obsługiwana przy resume.
- Przerwanie po pełnej epoce: zachowaj **cały katalog run**, zawierający m.in.
  `weights/last.pt`, `weights/best.pt`, `provenance.json` i `results.csv`.
- W nowej sesji dołącz ten katalog jako prywatny Dataset; wpisz pełną ścieżkę
  do jego `last.pt` w `RESUME_FROM`. Notebook sprawdzi identyfikator eksportu,
  skopiuje wyniki do zapisywalnego katalogu i wznowi stan optymalizatora.
- Domyślny `RESUME_FROM=None` oznacza nowy trening. Istniejący run nie jest
  nadpisywany. Przy wznowieniu docelowa liczba epok pochodzi ze starego checkpointu.
- Zakończony `last.pt` może mieć usunięty stan optymalizatora. Nie nadaje się
  wtedy do ścisłego resume; dalsze dostrajanie jest nowym eksperymentem.
- Checkpoint w `/kaggle/working` nie jest gwarantowanym backupem poza sesją.
  Przy utracie maszyny można stracić niezachowane pliki. Zachowuj wersje/output
  notebooka i pobieraj potrzebne checkpointy. Obsługa `finally` nie uratuje plików
  po nagłym usunięciu całej VM.

## Powrót modelu do projektu lokalnego

Po treningu pobierz ZIP wyników i zapisz nowy model np. jako
`models/parking_yolo26n_kaggle_v1.pt`. Nie zastępuj
`outputs/smoke_yolo26n/train/weights/best.pt` — umożliwi to porównanie modeli.
Do dotychczasowej komendy `yolo predict` podaj ścieżkę nowego modelu.

## Odtworzenie eksportu lokalnie

```powershell
uv run --frozen --no-sync python -m cloud.build_notebook
uv run --frozen --no-sync python -m src.export_kaggle
```

Eksporter odmawia nadpisania istniejącej paczki. Dla nowego eksportu użyj
`--output outputs/kaggle_export_v2`. ZIP jest sprawdzany przez CRC, pliki mają
indywidualne SHA-256, a report potwierdza niezmienność chronionych plików.
Notebook nie został jeszcze wykonany w rzeczywistym środowisku Kaggle;
lokalne kontrole nie zastępują jego kontroli GPU i zależności w chmurze.

## Kontrola przygotowanej paczki — 2026-10-25

- ZIP ma 3 591 556 644 bajty (około 3,59 GB); kontrola CRC wszystkich wpisów przeszła.
- Rozpakowano i zweryfikowano SHA-256 wszystkich 23 002 plików payloadu, a następnie
  przetestowano utworzenie i sprawdzenie kopii roboczej danych.
- Potwierdzono podział 8198/1729/1568 i brak testu w konfiguracji treningowej.
- Sprawdzono składnię wszystkich komórek notebooka. Nie instalowano środowiska
  Kaggle lokalnie i nie uruchamiano długiego treningu.
- Przeszło 47 testów całego projektu oraz 3 dodatkowe scenariusze zabezpieczeń
  wznowienia (przy ponownym uruchomieniu 14 testów modułu eksportu).
- Sumy chronionych plików źródłowych, konfiguracji i lokalnego checkpointu
  pozostały identyczne. Raport: `outputs/kaggle_export_validation.json`.

Konto, Dataset i uruchomienie na GPU Kaggle są kolejnym krokiem; żadne pliki
nie zostały automatycznie wysłane do serwisu.
