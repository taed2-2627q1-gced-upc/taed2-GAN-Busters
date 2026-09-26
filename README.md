# GAN-Busters

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Supervised binary image classification component for distinguishing between real and AI generated images.

--

## Environment Setup

This project uses **Python 3.10.x** and a **pip-based virtual environment**. 

### Prerequisites

Python **3.10.x** must be installed on the system before creating the virtual environment. The virtual environment is created from an existing Python 3.10 interpreter.

Verify that Python 3.10 is available:

```bash
# Linux/macOS/WSL
python3.10 --version
```

```powershell
# Windows PowerShell, if the Python Launcher is installed
py -3.10 --version
```

The command should report:

```text
Python 3.10.x
```

If Python 3.10 is not available, ([install it](https://www.python.org/downloads/release/python-31011/)) before proceeding.

### Option 1 — Makefile

On Linux/macOS, WSL, or Git Bash, the environment can be created using the provided Makefile.

This requires:

- GNU Make
- `virtualenvwrapper`
- A Python 3.10 interpreter available to the Makefile

For example, GNU Make can be installed on Ubuntu/WSL with:

```bash
sudo apt install make
```

Then create and activate the environment and install the project dependencies:

```bash
make create_environment
workon gan-busters
make requirements
```

After creating the environment, verify that the correct Python version is being used:

```bash
python --version
```

The output must report `Python 3.10.x` before installing or running the project dependencies.

> **Important:** The Makefile must resolve `PYTHON_INTERPRETER` to a Python 3.10 interpreter. If another Python version is used, the generated environment will not satisfy the project's Python requirement.

### Option 2 — Python `venv`

The environment can alternatively be created directly from a Python 3.10 interpreter.

#### Linux/macOS/WSL

```bash
mkdir -p ~/.virtualenvs
python3.10 -m venv ~/.virtualenvs/gan-busters
source ~/.virtualenvs/gan-busters/bin/activate
```

#### Windows PowerShell

Create the directory used to store virtual environments:

```powershell
New-Item -ItemType Directory -Force -Path "$HOME\.virtualenvs"
```

If the Windows Python Launcher is installed:

```powershell
py -3.10 -m venv "$HOME\.virtualenvs\gan-busters"
```

Otherwise, use the path to the installed Python 3.10 executable:

```powershell
& "<path-to-python-3.10>\python.exe" -m venv "$HOME\.virtualenvs\gan-busters"
```

Activate the environment:

```powershell
& "$HOME\.virtualenvs\gan-busters\Scripts\Activate.ps1"
```

### Verify the Environment

After activation, verify that the environment is using the required Python version:

```bash
python --version
```

The output must report:

```text
Python 3.10.x
```

Then install the project dependencies:

```bash
python -m pip install -U pip
python -m pip install -r requirements.txt
```

--

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── dataset_card.md<- Data from third party sources.
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│   └── model_card.md  <- The original, immutable data dump.
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         gan_busters and configuration for tools like black
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
└── gan_busters   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes gan_busters a Python module
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

--------

