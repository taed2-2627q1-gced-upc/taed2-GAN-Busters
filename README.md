# GAN-Busters

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Supervised binary image classification component for distinguishing between real and AI-generated images.

---

## Environment Setup

This project uses **Python 3.10.x** and a **pip-based virtual environment** named `gan-busters`.

### Activate an existing environment

If the environment has already been created, activate it before running the project.

#### Windows PowerShell

```powershell
& "$HOME\.virtualenvs\gan-busters\Scripts\Activate.ps1"
```

#### Linux/macOS/WSL

```bash
source ~/.virtualenvs/gan-busters/bin/activate
```

#### If created with `virtualenvwrapper`

```bash
workon gan-busters
```

Verify that the correct Python version is active:

```bash
python --version
```

The output should report:

```text
Python 3.10.x
```

---

### First-time setup

Python **3.10.x** must be installed before creating the virtual environment.

Verify that Python 3.10 is available:

#### Linux/macOS/WSL

```bash
python3.10 --version
```

#### Windows PowerShell

```powershell
py -3.10 --version
```

If Python 3.10 is not installed, it can be downloaded from the
[official Python releases page](https://www.python.org/downloads/release/python-31011/).

### Option 1 — Makefile

On Linux/macOS, WSL, or Git Bash, the environment can be created using the provided Makefile.

This requires:

- GNU Make
- `virtualenvwrapper`
- Python 3.10 available to the Makefile

Create the environment:

```bash
make create_environment
```

Activate it:

```bash
workon gan-busters
```

Install the project dependencies:

```bash
make requirements
```

Verify the Python version:

```bash
python --version
```

> The Makefile must resolve `PYTHON_INTERPRETER` to a Python 3.10 interpreter.

### Option 2 — Python `venv`

The environment can also be created directly using Python 3.10.

#### Linux/macOS/WSL

```bash
mkdir -p ~/.virtualenvs
python3.10 -m venv ~/.virtualenvs/gan-busters
source ~/.virtualenvs/gan-busters/bin/activate
```

#### Windows PowerShell

Create the virtual-environment directory:

```powershell
New-Item -ItemType Directory -Force -Path "$HOME\.virtualenvs"
```

Create the environment:

```powershell
py -3.10 -m venv "$HOME\.virtualenvs\gan-busters"
```

Activate it:

```powershell
& "$HOME\.virtualenvs\gan-busters\Scripts\Activate.ps1"
```

### Install dependencies

Once the environment is active:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## Project Organization

The project structure is based on the
[Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/)
template and has been adapted to the needs of this project.

```text
├── LICENSE
├── Makefile
├── README.md
│
├── data
│   ├── dataset_card.md      <- Dataset documentation, provenance, limitations, and usage
│   ├── external             <- Data from external sources
│   ├── interim              <- Intermediate transformed data
│   ├── processed            <- Final data prepared for modeling
│   └── raw                  <- Original immutable data
│
├── docs                     <- Project documentation
│
├── models
│   └── model_card.md        <- Model architecture, intended use, evaluation, and limitations
│
├── notebooks                <- Jupyter notebooks for exploration and experimentation
│
├── pyproject.toml           <- Project and tool configuration
│
├── references
│   ├── README.md            <- Centralized list of datasets, papers, standards, and related work
│   └── figures              <- Figures used in project documentation
│
├── reports                  <- Generated reports and analysis outputs
│   └── figures              <- Figures produced specifically for reporting
│
├── requirements.txt         <- Python dependencies required to reproduce the environment
│
├── setup.cfg                <- Additional project/tool configuration
│
└── gan_busters              <- Project source code
    ├── __init__.py
    ├── config.py            <- Shared configuration
    ├── dataset.py           <- Data loading and processing
    ├── features.py          <- Feature and preprocessing logic
    │
    ├── modeling
    │   ├── __init__.py
    │   ├── predict.py       <- Model inference
    │   └── train.py         <- Model training
    │
    └── plots.py             <- Visualization utilities
```

---

## Project Documentation

Additional project documentation can be found in:

- [`data/dataset_card.md`](data/dataset_card.md) — dataset description, provenance, preprocessing, and limitations
- [`models/model_card.md`](models/model_card.md) — model architecture, evaluation, intended use, and limitations
- [`references/README.md`](references/README.md) — datasets, literature, standards, and related work
