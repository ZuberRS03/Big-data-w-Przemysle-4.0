# Wyniki eksperymentu z odłożoną kamerą — 26.10.2026

Oba treningi i oceny zakończyły się poprawnie. Mocniejsze augmentacje poprawiły walidację ze znanych kamer UFPR04/UFPR05, ale **nie poprawiły przenoszenia na odłożoną kamerę PUCPR ani na dwa zdjęcia użytkownika**. Nowe warianty nie zastępują poprzedniego pełnego modelu.

[Raport ze zdjęciami](../outputs/camera_experiment_review_v1/index.html) · [pełne wyniki](../outputs/camera_experiment_review_v1/report.json) · [protokół](eksperyment_kamera.md).

## Trening

| Wariant | Epoki | Najlepsza epoka wg CSV | Czas treningu | Z inicjalizacją i oceną |
|---|---:|---:|---:|---:|
| baseline | 11 | 3 | 15 min 42 s | 17 min 23 s |
| augmented | 30 | 28 | 42 min 45 s | 43 min 37 s |

Baseline zatrzymał się po ośmiu epokach bez poprawy. Augmented wykonał limit 30 epok. Oba rozpoczęły od ogólnych wag YOLO26n. Trening: 5314 zdjęć UFPR04/UFPR05; dobór wag i progu: 1012 zdjęć tych kamer z innych dat; ocena odłożonej kamery: 717 zdjęć PUCPR dopiero po treningu.

Poprawiono błąd wyświetlania liczby epok: końcowa walidacja Ultralytics zwiększała licznik użyty w callbacku, zapisując 12/31 zamiast 11/30. Skorygowano kod oraz pochodne pola `completed_epochs` w plikach statusu, zachowując wcześniejszą wartość i opis korekty. CSV, wagi i metryki pozostały bez zmian. Testy `test_camera_experiment.py`: 5/5, w tym test regresji licznika.

## Wspólny próg confidence 0.25

MAE na klasę jest średnią bezwzględnych błędów liczby FREE i OCCUPIED. Jednostka: miejsca na zdjęcie na klasę, nie procent.

| Pomiar | baseline | augmented |
|---|---:|---:|
| Znane kamery: MAE na klasę | 0.287 | 0.166 |
| PUCPR: MAE na klasę | 47.272 | 48.506 |
| PUCPR: średnia liczba wykryć wobec 100 oznaczonych miejsc | 5.47 | 3.00 |
| PUCPR: zdjęcia bez wykryć | 58/717 | 143/717 |

Wszystkie zdjęcia PUCPR mają po 100 oznaczonych miejsc. Liczba ramek nie oznacza liczby poprawnych wykryć. Standardowe mAP@50 jest praktycznie zerowe: baseline 0.000003914 (0.0003914%), augmented 0.0000002733 (0.00002733%).

Progi dobrane tylko na znanych kamerach:

| Wariant | Próg | MAE znane kamery | MAE PUCPR |
|---|---:|---:|---:|
| baseline | 0.50 | 0.131 | 49.122 |
| augmented | 0.60 | 0.058 | 49.643 |

Podniesienie progu pogłębia pominięcia na PUCPR. Nie dobierano na niej progu confidence.

## Zdjęcia użytkownika: GPU, wejście 640, próg 0.25

| Zdjęcie | baseline FREE/OCCUPIED | augmented FREE/OCCUPIED | Poprzedni pełny FREE/OCCUPIED |
|---|---:|---:|---:|
| parking1.jpg | 0/0 | 0/1 | 6/25 |
| parking2.jpg | 11/1 | 1/0 | 3/1 |

[parking1 — porównanie](../outputs/camera_experiment_review_v1/parking1_comparison.jpg) · [parking2 — porównanie](../outputs/camera_experiment_review_v1/parking2_comparison.jpg).

Nowe modele niemal nic nie wykrywają na parking1. Na parking2 baseline oznacza m.in. dachy aut jako FREE, a augmented zwraca pojedynczą błędną ramkę FREE. Przy progu 0.60 augmented nie wykrywa nic na żadnym z obu zdjęć. Brak etykiet referencyjnych dla własnych zdjęć — to ocena wizualna, bez deklarowania procentowej dokładności. Poprzedni pełny model uczył się także na PUCPR, więc pokazano go wyłącznie na własnych zdjęciach; nie jest uczciwym punktem odniesienia dla nieznanej kamery.

## Kontrola wiarygodności

- SHA-256 przygotowanych obrazów, etykiet, manifestów i chronionych modeli: zgodne. Klasy: 0=FREE, 1=OCCUPIED.
- Ponowiono predykcje na 9 zdjęciach (po jednym na kamera/pogoda), dla obu nowych modeli. Wszystkie 18 liczników zgadza się z zapisanymi wynikami przy 0.25.
- Obejrzano [przykłady PUCPR](../outputs/camera_experiment_review_v1/holdout_sample_comparison.jpg): widoczne pominięcia i błędne ramki potwierdzają wynik metryk.
- Logi bez tracebacków i timeoutów NMS. Nie usuwano trudnych zdjęć ani nie poprawiano etykiet. Znany pojedynczy błędny XML nie wyjaśnia spadku na całych 717 obrazach.
- Zbiór testowy pozostaje nietknięty.

Ograniczenia: jeden seed i jedna odłożona kamera; cztery augmentacje zmieniono razem. Oba warianty miały tę samą regułę early stopping, lecz różną faktyczną liczbę epok. Eksperyment nie rozdziela wpływu poszczególnych augmentacji ani nie dowodzi, że inne podejście także zawiedzie.

## Wniosek

Ten pakiet augmentacji nie wystarczył do działania na nowym parkingu. Wynik sugeruje silną zależność od wyglądu, skali i geometrii ujęć treningowych, bez wskazania pojedynczej przyczyny. Nie ma podstaw do kolejnego identycznego treningu.

Zachowujemy poprzedni model pełny dla ujęć odpowiadających jego treningowi. Jeśli wymagane są różne parkingi bez konfiguracji, potrzebne są bardziej reprezentatywne oznaczone ujęcia i szersza ocena między kamerami. Dla stałej kamery można rozważyć jawne wskazanie miejsc oraz ocenę ich zajętości — to osobny wariant rozwiązania, tutaj jeszcze niewdrożony.

Ponowne wygenerowanie predykcji do nowego katalogu, z głównego katalogu repozytorium:

```powershell
uv run --frozen --no-sync python -u -m src.review_camera_experiment --output outputs/camera_experiment_review_v2
```

Skrypt nie uruchamia treningu; zapisuje obrazy, plansze i JSON. HTML bieżącego przeglądu tworzy dodatkowy `build_report.py` zapisany przy raporcie.
