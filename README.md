# RPS-classification

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

CNN for Rock-Paper-Scissors Classification

This repo trains a CNN to classify Rock / Paper / Scissors hand gestures.

Expected dataset layout
    ./data/raw/RPS/
    paper/*.jpg
    rock/*.jpg
    scissors/*.jpg

uv pip install -r requirements.lock.txt

# RPS Classification (PyTorch) — Rock–Paper–Scissors

Convolutional Neural Networks per classificare **rock / paper / scissors** con metodologia rigorosa:
- **Nessun leakage** del test set
- **Selezione architettura** con mini **k-fold CV**
- **Tuning iperparametri** con **k-fold CV** automatizzato
- **Retrain finale** su train+val, **valutazione una sola volta** su test (blind)
- Pipeline immagini con **resize lato corto + CenterCrop** (facile da cambiare)

## Struttura progetto
```
## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         rps_classification and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── rps_classification   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes rps_classification a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```


## Requisiti & setup ambiente
```bash
# Attiva il tuo venv (esempi)
# Windows (Git Bash):
source .venv/Scripts/activate
# Linux/Mac:
# source .venv/bin/activate

pip install -r requirements.txt
# (Opzionale GPU) Installa PyTorch con CUDA seguendo le istruzioni ufficiali per la tua GPU/driver.
```

## Dataset

Usa il dataset Rock-Paper-Scissors (Kaggle) copiandolo in:
data/data_rps/{rock,paper,scissors}
Le immagini originali sono circa 300×200 (RGB). La pipeline di default preserva l’aspect ratio con Resize(int) e poi fa CenterCrop a 192×192.


## how to use uv venv
uv sync
source .venv/Scripts/activate
uv run python -c "import sys; print(sys.executable)"

per disattivarlo:
deactivate
rm -rf .venv