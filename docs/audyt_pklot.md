# Audyt danych PKLot

Audyt dotyczy pełnych zdjęć w `data/PKLot/PKLot`. Katalog
`data/PKLot/PKLotSegmented` zawiera wycinki i nie jest wejściem tego skryptu.

Z katalogu głównego repozytorium:

```powershell
uv run --frozen --no-sync python -m src.audit_pklot
uv run --frozen --no-sync python -m pytest -q
```

Opcje `--frozen --no-sync` wykorzystują istniejące środowisko bez aktualizowania
zależności ani pliku blokady. Środowisko musi już zawierać wymagane biblioteki.

Inne lokalizacje można wskazać argumentami `--data-root` i `--output`.
Wyniki domyślnie trafiają do ignorowanego przez Git `outputs/pklot_audit/`:

- `report.md`: podsumowanie i interpretacja.
- `summary.json`: liczniki zdjęć, XML, klas, geometrii i problemów.
- `inventory.csv`: jeden wiersz na zdjęcie/XML, z ujęciem, pogodą i datą.
- `issues.csv`: szczegółowe problemy, ze ścieżką i identyfikatorem miejsca.
- `contact_sheet.jpg`: porównanie ujęć i pogody.
- `previews/`: pełne podglądy oznaczeń oraz przykład geometrii z `rotatedRect`.

## Zasady odczytu

- `occupied="0"` oznacza FREE, `occupied="1"` oznacza OCCUPIED.
- Brak lub inna wartość statusu pozostają nieznane; nie są przypisywane do FREE.
- Kontur z XML ma pierwszeństwo. Jeśli go brak, punkty są odtwarzane z obróconego
  prostokąta. Niepoprawny istniejący kontur jest zgłaszany jako błąd.
- Oryginalne pliki używają zarówno `point`, jak i `Point`; parser obsługuje oba warianty.
- Audyt zgłasza brakujące pary, błędy odczytu, brakujące/powtórzone identyfikatory,
  niepoprawną geometrię i współrzędne poza obrazem. Nie naprawia danych źródłowych.
- Każde zdjęcie jest dekodowane. Nie jest to wyłącznie sprawdzenie nagłówka pliku.
- Zielony oznacza FREE, czerwony OCCUPIED, żółty status nieznany.
  Wizualizacje przedstawiają adnotacje referencyjne, nie wyniki modelu.
- Dla każdej pary ujęcie–pogoda wybierany jest środkowy przykład z listy
  posortowanej według ścieżek. To kontrola przykładowa, nie wizualna inspekcja
  każdego zdjęcia. Audyt nie dowodzi poprawności semantycznej wszystkich etykiet.

## Przed treningiem

Najpierw trzeba ustalić obsługę zdjęć z brakującym statusem i geometrią.
Usunięcie tylko etykiety nieznanego miejsca przy pozostawieniu całego zdjęcia
może sprawić, że detektor potraktuje je podczas treningu jako tło.

Po kontroli danych należy zapisać powtarzalny podział według dni (bez mieszania
prawie identycznych zdjęć między zbiorami), przygotować niezależny test na
niewidzianym ujęciu i dopiero wtedy konwersję do wybranego formatu YOLO.
Wybór zwykłych ramek, obróconych ramek lub segmentacji powinien uwzględniać
perspektywę i nakładanie się sąsiednich miejsc.

## Ustalenia z lokalnego zbioru (2026-10-24)

- 12 417 czytelnych zdjęć w rozdzielczości 1280 × 720 i 12 416 plików XML.
- Zdjęcie `PUCPR/Sunny/2012-11-06/2012-11-06_18_48_46.jpg` nie ma XML.
- 357 978 oznaczeń FREE i 335 777 OCCUPIED, czyli 693 755 oznaczeń z klasą.
- 25 773 dodatkowe obszary nie mają statusu zajętości; dotyczą 778 zdjęć.
- W części UFPR04 występuje pisownia `Point`, którą trzeba uwzględnić przy
  odczycie konturów. Nie oznacza ona braku konturu.
- Po uwzględnieniu obu pisowni wszystkie 719 528 obszarów mają odczytywalne
  kontury o niezerowym polu, ze skończonymi współrzędnymi w granicach obrazu.
  W tym zbiorze nie było potrzeby użycia geometrii zastępczej z `rotatedRect`.
- 11 638 zdjęć ma XML, komplet statusów dla wymienionych w nim miejsc oraz
  geometrię przechodzącą kontrole audytu. To pula przed podziałem na zbiory.

Są to liczby adnotacji stanowisk na kolejnych zdjęciach, a nie unikalnych
fizycznych miejsc. Stanowiska powtarzają się między zdjęciami.

Podglądy pokazują też miejsca widoczne w kadrze poza oznaczonym obszarem parkingu.
Komplet statusów w XML nie oznacza więc, że oznaczono wszystkie widoczne
stanowiska. Należy uwzględnić to przy interpretowaniu wykryć spoza adnotacji.

Propozycja pierwszego eksperymentu: wykluczyć całe zdjęcie bez XML i zdjęcia
z nieznanym statusem, zapisać listę wykluczeń, a następnie sprawdzić rozkład
pozostałych danych według ujęcia, dnia i pogody przed podziałem. To wariant bazowy;
wykluczanie może usuwać trudne przypadki i nie zastępuje ich późniejszej oceny.
