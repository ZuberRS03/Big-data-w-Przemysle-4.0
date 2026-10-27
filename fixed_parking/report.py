"""Render saved final-test results without running inference again."""
import csv
import copy
from html import escape
import json
from pathlib import Path

import numpy as np
from PIL import Image

from fixed_parking.core import annotate, load_layout
from fixed_parking.evaluate import error_sheet, read_rows


def display_metrics(raw):
    """Handle absent classes in group balanced accuracy, using stored counts only."""
    report = copy.deepcopy(raw)
    for groups in report["by_group"].values():
        for scores in groups.values():
            matrix = np.asarray(scores["confusion_matrix"])
            support = matrix.sum(axis=1)
            present = support > 0
            scores["support"] = support.tolist()
            scores["balanced_accuracy"] = float((np.diag(matrix)[present] / support[present]).mean())
            scores["recall"] = [float(matrix[i, i] / support[i]) if support[i] else None for i in range(2)]
    report["postprocessing"] = "Group balanced accuracy averages recall only over classes present in XML; absent-class recall is null. Derived from stored confusion matrices, no inference or tuning. Original report.json retained."
    return report


def main():
    run = Path("outputs/fixed_parking_ufpr05_v1")
    output = run / "final_test"
    report = display_metrics(json.loads((output / "report.json").read_text()))
    (output / "report_display.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    training = json.loads((run / "report.json").read_text())
    protocol = json.loads((output / "frozen_protocol.json").read_text())
    m = report["metrics"]
    with (output / "frame_predictions.csv").open(encoding="utf-8", newline="") as stream:
        frames = list(csv.DictReader(stream))
    with (output / "slot_predictions.csv").open(encoding="utf-8", newline="") as stream:
        slots = list(csv.DictReader(stream))
    layout = load_layout(output / "frozen/layout.json")
    errors = read_rows(output / "errors.csv")
    for row in errors:
        row["probability"] = float(row["probability"])
    error_sheet(errors, layout, output, maximum=len(errors))
    # Worst count errors, with chronological/filename tie-break; no rerun of model.
    worst = sorted(frames, key=lambda f: (-abs(int(f["predicted_free"]) - int(f["reference_free"])), f["image"]))[:3]
    examples = []
    for i, frame in enumerate(worst, 1):
        records = [dict(id=s["slot_id"], occupied=bool(int(s["predicted"]))) for s in slots if s["image"] == frame["image"]]
        with Image.open(Path("data/pklot_yolo_v1") / frame["image"]) as image:
            preview = annotate(np.asarray(image.convert("RGB")), layout, records)
        name = f"worst_{i}.jpg"
        preview.save(output / name)
        examples.append(f'<figure><img src="{name}" alt="Przykład błędu licznika"><figcaption>{escape(Path(frame["image"]).stem)} — wolne: XML {frame["reference_free"]}, model {frame["predicted_free"]}; błędne miejsca: {frame["wrong_slots"]}.</figcaption></figure>')
    comparison = [
        ("Poprawność klasyfikacji miejsc", "accuracy", True),
        ("Balanced accuracy", "balanced_accuracy", True),
        ("Średni błąd liczby wolnych miejsc / zdjęcie", "free_count_mae", False),
        ("Zdjęcia z poprawnym licznikiem", "exact_count_fraction", True),
    ]
    rows = []
    for label, key, percent in comparison:
        fmt = lambda x: f"{100*x:.3f}%" if percent else f"{x:.4f}"
        rows.append(f"<tr><td>{label}</td><td>{fmt(training['validation'][key])}</td><td>{fmt(m[key])}</td></tr>")
    groups = []
    for key, heading in (("weather", "Pogoda"), ("date", "Dzień")):
        body = "".join(f'<tr><td>{escape(name)}</td><td>{g["frames"]}</td><td>{100*g["accuracy"]:.3f}%</td><td>{g["free_count_mae"]:.4f}</td><td>{100*g["exact_count_fraction"]:.2f}%</td></tr>' for name, g in report["by_group"][key].items())
        groups.append(f"<h2>{heading}</h2><table><tr><th>Grupa</th><th>Zdjęcia</th><th>Poprawność miejsc</th><th>MAE licznika</th><th>Dokładny licznik</th></tr>{body}</table>")
    cm = m["confusion_matrix"]
    html = f'''<!doctype html><html lang="pl"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>UFPR05 — ocena końcowa</title>
<style>body{{font:17px/1.6 system-ui,sans-serif;color:#172b3a;background:#f4f7f9;margin:0}}main{{max-width:1100px;margin:auto;padding:32px}}h1,h2{{line-height:1.2}}.cards{{display:flex;gap:16px;flex-wrap:wrap}}.card{{background:white;border-radius:12px;padding:20px;flex:1;min-width:190px}}.big{{font-size:32px;font-weight:700;color:#076e63}}table{{border-collapse:collapse;width:100%;background:white;font-size:15px}}td,th{{padding:10px;border-bottom:1px solid #dce3e8;text-align:left}}img{{max-width:100%;height:auto}}figure{{margin:24px 0}}.note{{border-left:4px solid #bd7b20;padding:14px;background:#fff8ea}}code{{overflow-wrap:anywhere}}a{{color:#006d9b}}</style>
<main><p>Smart Parking / stała kamera / 26.10.2026</p><h1>UFPR05: końcowa ocena modelu</h1>
<p>{m['frames']} zdjęć z {m['days']} odłożonych dni, po 40 miejsc: <b>{m['slots']} klasyfikacji</b>. Model i próg {report['threshold']:.2f} zamrożono przed otwarciem testu. Bez ponownego treningu i korekt etykiet.</p>
<div class="cards"><div class="card"><div class="big">{100*m['accuracy']:.3f}%</div>poprawnych klasyfikacji miejsc</div><div class="card"><div class="big">{m['free_count_mae']:.4f}</div>średni błąd liczby wolnych miejsc</div><div class="card"><div class="big">{100*m['exact_count_fraction']:.2f}%</div>zdjęć z dokładnym licznikiem</div></div>
<h2>Walidacja a test</h2><table><tr><th>Metryka</th><th>Walidacja (dobór ustawień)</th><th>Test końcowy</th></tr>{''.join(rows)}</table>
<p>Test: {m['wrong_slots']} rozbieżności z XML. Maksymalny błąd licznika: {m['maximum_count_error']}. Wszystkie miejsca sklasyfikowane poprawnie na {100*m['exact_slot_frame_fraction']:.2f}% zdjęć. Macro-F1: {m['macro_f1']:.6f}.</p>
<h2>Macierz pomyłek</h2><table><tr><th>Oznaczenie XML ↓ / model →</th><th>Wolne</th><th>Zajęte</th></tr><tr><th>Wolne</th><td>{cm[0][0]}</td><td>{cm[0][1]}</td></tr><tr><th>Zajęte</th><td>{cm[1][0]}</td><td>{cm[1][1]}</td></tr></table>
{''.join(groups)}
<h2>Przegląd 12 rozbieżności walidacyjnych</h2><p>Na wszystkich 12 wycinkach miejsca wyglądają na puste, mimo etykiety OCCUPIED w XML. Część zasłaniają fragmenty sąsiednich pojazdów. To podejrzenie niespójności etykiet na podstawie oględzin, bez niezależnego potwierdzenia. Zachowano oryginalne etykiety i wynik walidacyjny; nie deklarujemy „skorygowanej” skuteczności.</p><img src="../validation_audit/errors.jpg" alt="Wszystkie 12 rozbieżności walidacyjnych, żółte obrysy miejsc">
<h2>Rozbieżności testowe</h2><p>Wszystkie {m['wrong_slots']} rozbieżności w kolejności manifestu. Żółty obrys oznacza oceniane miejsce; XML i pred: 0 = wolne, 1 = zajęte. Powtarzają się sytuacje z cieniem i fragmentami sąsiednich aut w obrysie; część oznaczeń XML wygląda niespójnie z obrazem. To obserwacje wizualne, bez korekt referencji i bez dostrajania modelu.</p>{'<img src="errors.jpg" alt="Rozbieżności testowe">' if report['error_sheet'] else '<p>Brak rozbieżności.</p>'}
<h2>Zdjęcia z największym błędem licznika</h2><p>Wybrane według bezwzględnego błędu liczby wolnych miejsc; remisy rozstrzygnięto nazwą pliku. Obrysy odtworzono z zapisanych predykcji.</p>{''.join(examples)}
<h2>Zakres wniosków i wydajność</h2><p class="note">Wynik dotyczy znanej kamery UFPR05 i odłożonych dni. Nie dowodzi działania na nowym parkingu. Kolejne zdjęcia są czasowo powiązane, a miejsc jest zawsze 40; 27 tysięcy wycinków nie oznacza tylu niezależnych scen. Test został już wykorzystany — dalsze dostrajanie na jego podstawie wymaga nowego niezależnego zbioru.</p>
<p>Model ma 89 426 parametrów i zajmuje 365 473 bajty. Wcześniejszy pomiar na desktopowym CPU: mediana 19,36 ms, p95 24,32 ms na 40 miejsc (4 wątki, 30 pomiarów po rozgrzewce; RGB → wycinki → model → liczniki i obrysy, bez dysku/JPEG/przeglądarki). To nie jest pomiar Raspberry Pi; możliwość wdrożenia na RPi pozostaje hipotezą do weryfikacji.</p>
<p>W teście są tylko 3 zdjęcia deszczowe, wobec 605 słonecznych i 72 pochmurnych. Wynik 100% dla deszczu nie wystarcza do wniosku o odporności na deszcz. Dzień 2013-02-24 zawiera wyłącznie wolne miejsca, więc nie mierzy rozpoznawania zajętych stanowisk.</p>
<h2>Odtwarzalność</h2><p>Zamrożenie UTC: {escape(protocol['frozen_at_utc'])}. SHA-256 modelu: <code>{report['model_sha256']}</code>.</p>
<p><a href="report_display.json">Pełne metryki JSON</a> · <a href="report.json">Surowy raport oceny</a> · <a href="frozen_protocol.json">Protokół i hashe</a> · <a href="frame_predictions.csv">Wyniki zdjęć CSV</a> · <a href="slot_predictions.csv">Wyniki wszystkich miejsc CSV</a> · <a href="errors.csv">Rozbieżności CSV</a> · <a href="input_hashes.json">Hashe danych wejściowych</a></p><p>W raporcie pochodnym poprawiono wyłącznie agregację balanced accuracy dla grup bez jednej klasy: uwzględnia klasy obecne w XML. Surowy raport zachowano. Predykcje, ustawienia i wyniki całego testu pozostają bez zmian.</p></main></html>'''
    (output / "index.html").write_text(html, encoding="utf-8")
    print(output / "index.html")


if __name__ == "__main__":
    main()
