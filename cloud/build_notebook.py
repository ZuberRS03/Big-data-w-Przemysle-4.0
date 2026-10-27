"""Build the committed notebook from reviewable, plain-Python cell sources."""
import json
from pathlib import Path
import textwrap


def build(destination: Path):
    cells = []
    def add(kind, source):
        cell = dict(cell_type=kind, metadata={}, source=textwrap.dedent(source).strip() + "\n",
                    id=f"smart-parking-{len(cells):02}")
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    add("markdown", """
    # Smart Parking Vision — trening YOLO26 Nano na Kaggle

    1. Dodaj własny **prywatny** Dataset utworzony z `smart_parking_dataset.zip`.
    2. Włącz **GPU** i **Internet** w ustawieniach notebooka (wymagana instalacja pakietów).
    3. Uruchamiaj komórki po kolei. Dłuższy eksperyment można uruchomić przez **Save Version → Save & Run All**.

    Zbiór: 8198 train, 1729 val, 1568 test; podział według dat, seed 42.
    Dane własnych zdjęć i lokalny model próbny nie są częścią eksportu.
    Domyślnie: 30 epok, patience 8, Nano, 640 px, batch 8, pojedynczy GPU.
    To trening kandydata na finalny model; jakości nie da się zagwarantować z góry.
    **Końcowy test jest domyślnie wyłączony.** Włącz go dopiero po zakończeniu doboru modelu na walidacji.
    """)
    add("markdown", """
    ## 1. Biblioteki — zachowujemy PyTorch i CUDA dostarczone przez Kaggle

    Instalujemy tę samą wersję Ultralytics co lokalnie. Nie kopiujemy Windowsowego `.venv`
    ani `uv.lock`. Instalacja ogranicza wersje torch/torchvision do już obecnych,
    aby resolver nie podmienił środowiska GPU. Jeśli wystąpi konflikt, komórka przerwie działanie.
    """)
    add("code", """
    import os, sys, subprocess, importlib.metadata as metadata
    from pathlib import Path
    import torch, torchvision

    WORK_ROOT = Path('/kaggle/working/smart_parking')
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    os.environ['YOLO_CONFIG_DIR'] = str(WORK_ROOT / 'ultralytics_config')
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    os.environ['WANDB_MODE'] = 'disabled'
    PIN = '8.4.187'
    original_torch = metadata.version('torch')
    original_vision = metadata.version('torchvision')
    constraints = WORK_ROOT / 'gpu_constraints.txt'
    constraints.write_text(f'torch=={original_torch}\\ntorchvision=={original_vision}\\n')
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', '--no-deps', f'ultralytics=={PIN}'], check=True)
    from packaging.requirements import Requirement
    missing = []
    for spec in metadata.requires('ultralytics'):
        req = Requirement(spec)
        if req.marker and not req.marker.evaluate({'extra': ''}):
            continue
        try:
            satisfied = metadata.version(req.name) in req.specifier
        except metadata.PackageNotFoundError:
            satisfied = False
        if not satisfied:
            missing.append(str(req))
    if missing:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet',
                        '--upgrade-strategy', 'only-if-needed', '-c', str(constraints), *missing], check=True)
    assert metadata.version('torch') == original_torch
    assert metadata.version('torchvision') == original_vision
    import ultralytics
    from ultralytics import settings
    settings.update({key: False for key in ('wandb', 'clearml', 'comet', 'mlflow', 'neptune', 'dvc') if key in settings})
    assert ultralytics.__version__ == PIN
    if not torch.cuda.is_available():
        raise RuntimeError('Włącz GPU w ustawieniach notebooka.')
    # Check real GPU execution, not only CUDA availability.
    tensor = torch.randn((64, 64), device='cuda:0')
    assert torch.isfinite(tensor @ tensor).all().item()
    torch.cuda.synchronize()
    print('GPU:', torch.cuda.get_device_name(0), '| torch:', torch.__version__, '| CUDA:', torch.version.cuda)
    print('Ultralytics:', ultralytics.__version__)
    """)
    add("markdown", """
    ## 2. Odnalezienie paczki, kontrola plików i kopia robocza

    Paczka jest wyszukiwana automatycznie w `/kaggle/input`, niezależnie od nazwy Datasetu.
    Po weryfikacji SHA-256 dane trafiają do zapisywalnego `/kaggle/working/smart_parking/data`.
    Nie zmieniamy plików wejściowych; kopia pozwala Ultralytics zapisywać cache.
    """)
    add("code", """
    import json, hashlib, importlib.util
    candidates = []
    for path in Path('/kaggle/input').rglob('bundle.json'):
        try:
            if json.loads(path.read_text()).get('kind') == 'smart-parking-kaggle':
                candidates.append(path)
        except ValueError:
            pass
    if len(candidates) != 1:
        raise RuntimeError(f'Dodaj dokładnie jeden Dataset z paczką Smart Parking. Znaleziono: {candidates}')
    BUNDLE_ROOT = candidates[0].parent
    bundle_meta = json.loads(candidates[0].read_text())
    helper = BUNDLE_ROOT / 'code/kaggle_runtime.py'
    expected = next(item['sha256'] for item in bundle_meta['files'] if item['path'] == 'code/kaggle_runtime.py')
    assert hashlib.sha256(helper.read_bytes()).hexdigest() == expected, 'Uszkodzony skrypt paczki'
    spec = importlib.util.spec_from_file_location('kaggle_runtime', helper)
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    BUNDLE = runtime.prepare_data(BUNDLE_ROOT, WORK_ROOT)
    settings.update({'weights_dir': str(BUNDLE_ROOT / 'weights'),
                     'datasets_dir': str(WORK_ROOT / 'data'), 'runs_dir': str(WORK_ROOT / 'runs')})
    print('Zweryfikowano:', BUNDLE['dataset_images'], 'zdjęć;', BUNDLE['splits'])
    """)
    add("markdown", """
    ## 3. Konfiguracja i trening

    `RESUME_FROM=None` zaczyna niezależny trening od oryginalnych wag Nano, nie od modelu próbnego.
    W razie błędu pamięci zmniejsz BATCH do 4. Jeśli poprzedni trening już utworzył katalog,
    użyj nowego RUN_NAME albo wznów z własnego przerwanego `last.pt`.

    Wznowienie: dołącz zachowane wyniki poprzedniej sesji jako prywatny Dataset i ustaw
    pełną ścieżkę do `.../<nazwa_run>/weights/last.pt` w RESUME_FROM.
    Musi być dołączony **cały katalog eksperymentu**, w tym `provenance.json`,
    `best.pt` i `results.csv`. Docelowa liczba epok przy wznowieniu pochodzi z checkpointu.
    Checkpoint z zakończonego treningu ma usunięty stan optymalizatora i nie służy do resume.
    """)
    add("code", """
    RUN_NAME = 'parking_yolo26n_v1'
    EPOCHS = 30
    BATCH = 8
    WORKERS = 2
    RESUME_FROM = None  # własny last.pt z przerwanej sesji, nigdy checkpoint z nieznanego źródła
    RUN_DIR = WORK_ROOT / 'runs' / RUN_NAME
    """)
    add("code", """
    BEST = runtime.train(BUNDLE_ROOT, WORK_ROOT, run_name=RUN_NAME, epochs=EPOCHS,
                         batch=BATCH, workers=WORKERS, resume_from=RESUME_FROM)
    print('Najlepszy model:', BEST)
    """)
    add("markdown", """
    ## 4. Wyniki walidacyjne i liczniki

    Sprawdzamy błąd liczników na pełnej walidacji przy progach 0,25 / 0,40 / 0,50 / 0,60.
    Wybieramy próg o najmniejszym średnim błędzie bezwzględnym liczników FREE i OCCUPIED.
    Nie używamy zbioru testowego do wyboru progu. PKLot nie opisuje wszystkich miejsc
    widocznych w kadrze, więc błędy liczników są względem obszarów opisanych w XML.
    """)
    add("code", """
    import pandas as pd
    from IPython.display import display, Image, FileLink
    print((RUN_DIR / 'validation_metrics.json').read_text())
    count_scores = runtime.evaluate_counts(BEST, WORK_ROOT, RUN_DIR)
    display(pd.DataFrame(count_scores).T)
    print('Wybrany próg:', (RUN_DIR / 'selected_threshold.json').read_text())
    for filename in ('results.png', 'confusion_matrix_normalized.png', 'val_batch0_pred.jpg'):
        if (RUN_DIR / filename).is_file():
            display(Image(filename=str(RUN_DIR / filename), width=1000))
    """)
    add("markdown", """
    ## 5. Opcjonalny, końcowy test — domyślnie pomijany

    Ustaw True dopiero po zamrożeniu modelu i progu. Nie dobieraj kolejnych ustawień
    na podstawie testu. Po pierwszym zapisanym teście skrypt odmawia ponownego uruchomienia
    testu w tym samym katalogu. Normalne „Run All” zachowuje ten krok wyłączony.
    """)
    add("code", """
    RUN_FINAL_TEST = False
    if RUN_FINAL_TEST:
        runtime.final_test(BEST, WORK_ROOT, RUN_DIR)
        print((RUN_DIR / 'test_metrics.json').read_text())
    else:
        print('Końcowy test pominięty. Model wybrano wyłącznie według walidacji.')
    """)
    add("markdown", """
    ## 6. Pobranie wyników

    Archiwum zawiera wagi, konfigurację, metryki, wykresy i wersje bibliotek, bez kopii datasetu.
    `best.pt` służy do inferencji. `last.pt` umożliwia resume tylko po przerwanym treningu.
    Zapisywanie po każdej epoce **nie oznacza trwałego backupu poza maszyną Kaggle**.
    Utrata całej sesji może usunąć niezachowane pliki; zachowuj wyniki przez wersję notebooka
    i pobieraj potrzebne checkpointy. Po zakończeniu pobierz ZIP z zakładki Output.
    """)
    add("code", """
    archive = runtime.archive_run(RUN_DIR, WORK_ROOT)
    display(FileLink(str(BEST)))
    display(FileLink(str(archive)))
    print('Pobierz best.pt lub całe archiwum:', archive)
    """)
    notebook = dict(cells=cells, metadata=dict(kernelspec=dict(display_name="Python 3", language="python", name="python3"),
                                             language_info=dict(name="python", version="3.11")),
                    nbformat=4, nbformat_minor=5)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build(Path("notebooks/smart_parking_kaggle.ipynb"))
