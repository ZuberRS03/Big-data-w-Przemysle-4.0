"""Build README figures from saved experiments; never train or run inference.

Run from the repository root. --refresh-source refreshes the small data snapshot
from local outputs; otherwise figures can be rebuilt without the dataset/weights.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs/report_assets"
RUN = "outputs/fixed_parking_ufpr05_v1"
BLUE, TEAL, ORANGE, RED = "#2864ae", "#008477", "#d88a22", "#c44f50"


def snapshot():
    provenance = {}

    def read(path):
        raw = (ROOT / path).read_bytes()
        provenance[path] = hashlib.sha256(raw).hexdigest()
        return raw.decode("utf-8-sig")

    def js(path):
        return json.loads(read(path))

    def rows(path):
        return list(csv.DictReader(read(path).splitlines()))

    manifest = rows("data/pklot_yolo_v1/manifest.csv")
    diagnostic = js("outputs/diagnostics_v2/report.json")
    camera = js("outputs/camera_experiment_review_v1/report.json")
    frames = rows(f"{RUN}/final_test/frame_predictions.csv")
    result = dict(
        dataset={split: dict(images=len(part := [r for r in manifest if r["split"] == split]),
                            days=len({r["date"] for r in part}),
                            free=sum(int(r["free"]) for r in part), occupied=sum(int(r["occupied"]) for r in part))
                 for split in ("train", "val", "test")},
        full_history=rows("outputs/full_yolo26n_v1/results.csv"),
        full_metrics=js("outputs/full_yolo26n_v1/validation_metrics.json"),
        full_progress={key: value for key, value in js("outputs/full_yolo26n_v1/progress.json").items()
                       if key not in ("pid", "output", "last_checkpoint", "best_checkpoint")},
        diagnostic_scores={name: value["scores"] for name, value in diagnostic["evaluations"].items()},
        camera=camera["comparison"]["results"], camera_training=camera["training"],
        fixed_history=js(f"{RUN}/history.json"), fixed_training=js(f"{RUN}/report.json"),
        final=js(f"{RUN}/final_test/report_display.json"), cpu=js(f"{RUN}/cpu_benchmark.json"),
        count_errors=dict(sorted(Counter(abs(int(r["predicted_free"]) - int(r["reference_free"])) for r in frames).items())),
    )
    result["source_sha256"] = provenance
    result["description"] = "Snapshot of measured results for README charts; no synthetic observations or new inference."
    (ASSETS / "source_data.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    copies = {
        "fixed_example.jpg": f"{RUN}/demo_rainy.jpg",
        "own_photo_yolo_comparison.jpg": "outputs/camera_experiment_review_v1/parking2_comparison.jpg",
        "validation_disagreements.jpg": f"{RUN}/validation_audit/errors.jpg",
        "test_disagreements.jpg": f"{RUN}/final_test/errors.jpg",
    }
    image_sources = {}
    for name, source in copies.items():
        shutil.copy2(ROOT / source, ASSETS / name)
        image_sources[name] = dict(source=source, sha256=hashlib.sha256((ROOT / source).read_bytes()).hexdigest())
    (ASSETS / "image_sources.json").write_text(json.dumps(image_sources, indent=2), encoding="utf-8")
    return result


def style(ax, ylabel=None):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.18)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel)


def save(fig, name):
    fig.savefig(ASSETS / f"{name}.png", dpi=170, facecolor="white")
    plt.close(fig)


def figures(d):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.titleweight": "bold", "axes.titlesize": 13,
                         "axes.labelcolor": "#263849", "text.color": "#263849"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), layout="constrained")
    fixed = d["fixed_training"]["dataset"]["splits"]
    for ax, counts, title, dates in (
        (axes[0], [d["dataset"][s]["images"] for s in ("train", "val", "test")], "PKLot po konwersji — wszystkie ujęcia", [d["dataset"][s]["days"] for s in ("train", "val", "test")]),
        (axes[1], [fixed["train"]["frames"], fixed["val"]["frames"], d["final"]["metrics"]["frames"]], "UFPR05 — końcowy klasyfikator", [fixed["train"]["days"], fixed["val"]["days"], d["final"]["metrics"]["days"]]),
    ):
        bars = ax.bar(["Trening", "Walidacja", "Test"], counts, color=[BLUE, ORANGE, TEAL], width=.6)
        ax.bar_label(bars, labels=[f"{n:,}\n{day} dni".replace(",", " ") for n, day in zip(counts, dates)], padding=5)
        ax.set_ylim(0, max(counts) * 1.3)
        ax.set_title(title)
        style(ax, "Liczba zdjęć")
    save(fig, "01_data_split")

    h = d["full_history"]
    epochs = [int(r["epoch"]) for r in h]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), layout="constrained")
    for key, label, color in (("train/box_loss", "Trening", BLUE), ("val/box_loss", "Walidacja", ORANGE)):
        axes[0].plot(epochs, [float(r[key]) for r in h], "o-", label=label, color=color, markersize=4)
    axes[0].set_title("YOLO: strata lokalizacji ramek")
    axes[0].set_ylim(bottom=0)
    style(axes[0], "Box loss")
    for key, label, color in (("metrics/mAP50(B)", "mAP@50", TEAL), ("metrics/mAP50-95(B)", "mAP@50–95", BLUE)):
        axes[1].plot(epochs, [float(r[key]) * 100 for r in h], "o-", label=label, color=color, markersize=4)
    axes[1].set_title("YOLO: mAP na walidacji")
    axes[1].set_ylim(0, 105)
    style(axes[1], "mAP [%]")
    for ax in axes:
        ax.axvline(5, color=RED, linestyle="--", alpha=.7, label="Wybrana epoka: 5")
        ax.set_xlabel("Epoka")
        ax.legend(fontsize=9)
    save(fig, "02_yolo_training")

    scores = d["diagnostic_scores"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for name, label, color in (("smoke_640", "Próbny (3 epoki)", ORANGE), ("full_640", "Pełny (best.pt)", BLUE)):
        values = scores[name]
        keys = sorted(values, key=float)
        axes[0].plot([float(k) for k in keys], [values[k]["class_count_mae"] for k in keys], "o-", label=label, color=color)
    axes[0].set_yscale("log")
    axes[0].set_title("Próg confidence — 1729 zdjęć, wejście 640")
    axes[0].set_xlabel("Confidence")
    axes[0].legend(fontsize=9)
    style(axes[0], "MAE na klasę [miejsca], skala log")
    resolutions = [160, 320, 640, 960, 1280]
    vals = [scores["full_640_subset" if s == 640 else f"full_{s}"]["0.25"]["class_count_mae"] for s in resolutions]
    bars = axes[1].bar([str(s) for s in resolutions], vals, color=[BLUE, BLUE, TEAL, BLUE, BLUE])
    axes[1].bar_label(bars, fmt="%.2f", padding=4)
    axes[1].set_ylim(0, max(vals) * 1.18)
    axes[1].set_title("Rozdzielczość — te same 135 zdjęć")
    axes[1].set_xlabel("Rozmiar wejścia [px], confidence = 0.25")
    style(axes[1], "MAE na klasę [miejsca]")
    save(fig, "03_yolo_diagnostics")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), layout="constrained")
    for ax, group, title in ((axes[0], "val", "Znane ujęcia UFPR04/05 — 1012 zdjęć"), (axes[1], "holdout", "Odłożona kamera PUCPR — 717 zdjęć")):
        vals = [d["camera"][v][group]["0.25"]["class_count_mae"] for v in ("baseline", "augmented")]
        bars = ax.bar(["Baseline", "Augmented"], vals, color=[BLUE, ORANGE], width=.55)
        ax.bar_label(bars, fmt="%.3f", padding=5)
        ax.set_ylim(0, max(vals) * 1.22)
        ax.set_title(title)
        style(ax, "MAE na klasę [miejsca]")
    fig.supxlabel("Wspólny confidence = 0.25. Uwaga: panele mają różne skale osi Y.", fontsize=10)
    save(fig, "04_camera_holdout")

    h = d["fixed_history"]
    epochs = [r["epoch"] for r in h]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for key, label, color in (("train_loss", "Trening + augmentacja", BLUE), ("val_loss", "Walidacja", ORANGE)):
        axes[0].plot(epochs, [r[key] for r in h], "o-", color=color, label=label, markersize=4)
    axes[0].set_yscale("log")
    axes[0].set_title("CNN: ważona entropia krzyżowa")
    style(axes[0], "Strata, skala logarytmiczna")
    axes[1].plot(epochs, [r["free_count_mae"] for r in h], "o-", color=TEAL, label="Próg 0.50 w historii epok", markersize=4)
    axes[1].scatter([10], [d["fixed_training"]["validation"]["free_count_mae"]], color=RED, marker="D", zorder=5, label="Epoka 10 po doborze progu 0.60")
    axes[1].set_ylim(bottom=0)
    axes[1].set_title("CNN: błąd licznika na walidacji")
    style(axes[1], "MAE [miejsca / zdjęcie]")
    for ax in axes:
        ax.axvline(10, color=RED, linestyle="--", alpha=.6)
        ax.set_xlabel("Epoka")
        ax.legend(fontsize=8.5)
    save(fig, "05_cnn_training")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    matrix = np.asarray(d["final"]["metrics"]["confusion_matrix"])
    pct = matrix / matrix.sum(axis=1, keepdims=True) * 100
    axes[0].imshow(pct, cmap="Blues", vmin=0, vmax=100)
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, f"{matrix[i,j]:,}\n{pct[i,j]:.3f}%".replace(",", " "), ha="center", va="center", color="white" if pct[i,j] > 50 else "#263849", fontsize=13)
    axes[0].set_xticks([0, 1], ["Wolne", "Zajęte"])
    axes[0].set_yticks([0, 1], ["Wolne", "Zajęte"])
    axes[0].set_xlabel("Predykcja modelu")
    axes[0].set_ylabel("Oryginalna etykieta XML")
    axes[0].set_title("Test: 27 200 miejsc\nProcenty w obrębie wiersza")
    errors = d["count_errors"]
    bars = axes[1].bar([str(k) for k in errors], list(errors.values()), color=[TEAL, ORANGE, RED], width=.6)
    axes[1].bar_label(bars, labels=[f"{v} ({100*v/d['final']['metrics']['frames']:.2f}%)" for v in errors.values()], padding=5)
    axes[1].set_ylim(0, 760)
    axes[1].set_xlabel("Bezwzględny błąd liczby wolnych miejsc")
    axes[1].set_title("Test: licznik na 680 zdjęciach")
    style(axes[1], "Liczba zdjęć")
    save(fig, "06_final_test")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for ax, key, title in ((axes[0], "date", "Wynik w poszczególnych dniach"), (axes[1], "weather", "Wynik według pogody")):
        groups = d["final"]["by_group"][key]
        translated = {"Cloudy": "Pochmurno", "Rainy": "Deszcz", "Sunny": "Słonecznie"}
        labels = [f"{translated.get(name, name)}\nn={g['frames']}" for name, g in groups.items()]
        bars = ax.bar(labels, [g["free_count_mae"] for g in groups.values()], color=BLUE if key == "date" else [BLUE, TEAL, ORANGE])
        ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=9)
        ax.set_ylim(0, .095)
        if key == "date":
            ax.tick_params(axis="x", labelrotation=35, labelsize=9)
        ax.set_title(title)
        style(ax, "MAE wolnych miejsc / zdjęcie")
    save(fig, "07_test_groups")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-source", action="store_true")
    args = parser.parse_args()
    ASSETS.mkdir(parents=True, exist_ok=True)
    data = snapshot() if args.refresh_source else json.loads((ASSETS / "source_data.json").read_text(encoding="utf-8"))
    figures(data)
    print(f"Saved 7 figures in {ASSETS}")
