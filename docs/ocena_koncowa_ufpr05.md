# Końcowa ocena Smart Parking — UFPR05

Ocena z 26.10.2026. Model klasyfikuje stałe miejsca parkingowe w znanym ujęciu. Nie jest uniwersalnym detektorem miejsc na dowolnym zdjęciu.

## Protokół

Trening: 2897 zdjęć / 23 dni / 115880 wycinków. Walidacja: 437 zdjęć / 4 dni / 17480 wycinków. Wybrano checkpoint z epoki 10 oraz próg zajętości 0.60. Wagi, geometria i kod zostały skopiowane do `outputs/fixed_parking_ufpr05_v1/final_test/frozen/` przed oceną testową. SHA-256 modelu: `f3e9c86f4f8f8c15768d2b46465610ba592e3d4fcbfbff7e98127d9b4c3b804f`.

Test obejmował wszystkie 680 zdjęć UFPR05 z 6 odłożonych dni i 40 miejsc na zdjęcie (27200 klasyfikacji). Daty testowe są rozłączne z datami treningu i walidacji. Dla każdego zdjęcia potwierdzono zgodność obrazu z oryginałem oraz geometrii i identyfikatorów miejsc z XML. Nie poprawiano etykiet, nie zmieniano progu, nie trenowano ponownie modelu.

## Wyniki

| Miara | Walidacja | Test |
|---|---:|---:|
| Accuracy miejsc | 99.931% | 99.904% |
| Balanced accuracy | 99.937% | 99.904% |
| MAE liczby wolnych miejsc | 0.0275 | 0.0382 |
| Dokładny licznik | 97.254% | 96.324% |

Na teście 27174 z 27200 etykiet zgadzało się z XML. 14 wolnych miejsc zaklasyfikowano jako zajęte, a 12 zajętych jako wolne. Licznik był dokładny na 655/680 zdjęć. Na 24 zdjęciach różnił się o 1, na jednym o 2 miejsca. Macro-F1: 0.999042. W tym teście poprawny licznik pokrywał się z poprawną klasyfikacją wszystkich miejsc, ale ogólnie błędy przeciwnych klas mogą się znosić.

| Pogoda | Zdjęcia | Accuracy miejsc | MAE licznika |
|---|---:|---:|---:|
| Pochmurno | 72 | 99.931% | 0.0278 |
| Deszcz | 3 | 100.000% | 0.0000 |
| Słonecznie | 605 | 99.901% | 0.0397 |

## Rozbieżności i ograniczenia

Wszystkie 12 rozbieżności walidacyjnych obejrzano z obrysami XML. Miejsca wyglądają na puste mimo referencyjnej etykiety zajętości; część obrazu zasłaniają sąsiednie samochody. To podejrzenie niespójności oznaczeń, nie niezależnie potwierdzona korekta. Oryginalne etykiety i wyniki zachowano.

W testowych rozbieżnościach powtarzają się cienie i fragmenty sąsiednich samochodów. Niektóre etykiety również wyglądają niespójnie z obrazem. Wszystkie 26 rozbieżności przedstawiono w raporcie HTML bez usuwania ich z oceny. Przyczyny pozostają obserwacjami wizualnymi, bez kontrolowanego eksperymentu potwierdzającego ich wpływ.

Wynik potwierdza skuteczność dla UFPR05 na odłożonych dniach, nie dla innych kamer. Próbka deszczowa jest zbyt mała do mocnych wniosków o odporności na deszcz. Jeden dzień zawiera wyłącznie wolne miejsca. Kolejne zdjęcia są skorelowane czasowo, dlatego 27200 wycinków nie oznacza tylu niezależnych scen. Test jest teraz wykorzystany; kolejne strojenie wymaga nowego niezależnego zbioru do końcowej oceny.

## Słabsze urządzenia

CNN ma 89426 parametrów i checkpoint 365473 bajty. Wcześniejszy pomiar na desktopowym CPU (4 wątki, 30 pomiarów po rozgrzewce, 40 miejsc) dał medianę 19.36 ms i p95 24.32 ms. Obejmuje przygotowanie wycinków, inferencję, liczniki i obrysy, lecz nie dysk, JPEG ani przeglądarkę. Nie jest to wynik Raspberry Pi. Mały rozmiar modelu uzasadnia dalsze zbadanie możliwości takiego wdrożenia, ale wydajność RPi pozostaje niezmierzona.

## Artefakty

[Raport HTML z podglądami](../outputs/fixed_parking_ufpr05_v1/final_test/index.html), [metryki do prezentacji](../outputs/fixed_parking_ufpr05_v1/final_test/report_display.json), [surowy raport](../outputs/fixed_parking_ufpr05_v1/final_test/report.json), [protokół zamrożenia](../outputs/fixed_parking_ufpr05_v1/final_test/frozen_protocol.json), [wszystkie predykcje](../outputs/fixed_parking_ufpr05_v1/final_test/slot_predictions.csv).

Raport pochodny poprawia wyłącznie balanced accuracy grup bez jednej klasy, licząc średnią recall klas obecnych w XML; nieobecny recall jest `null`. Surowy raport przypisywał nieobecnej klasie zero. Korekta wynika z zapisanych macierzy pomyłek, bez nowej inferencji; wynik całego testu pozostał bez zmian. Pliki w `outputs/` są lokalne i ignorowane przez Git — do przekazania projektu należy dołączyć osobno wybrane wyniki i wagi.
