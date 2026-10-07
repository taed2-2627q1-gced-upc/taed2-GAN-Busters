# GAN-Busters

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Supervised binary image classification component for distinguishing between real and AI-generated images.

---

## Project Documentation

Additional project documentation can be found in:

- [`data/dataset_card.md`](data/dataset_card.md) — dataset description, provenance, preprocessing, and limitations
- [`models/model_card.md`](models/model_card.md) — model architecture, evaluation, intended use, and limitations
- [`references/README.md`](references/README.md) — datasets, literature, standards, and related work

---

## Daily Workflow

### 1. Activate the environment

#### Windows PowerShell

```powershell
& "$HOME\.virtualenvs\gan-busters\Scripts\Activate.ps1"
```

#### Linux/macOS/WSL

```bash
source ~/.virtualenvs/gan-busters/bin/activate
```

If the environment was created with `virtualenvwrapper`:

```bash
workon gan-busters
```

### 2. Get the latest project version

Pull the latest Git changes first:

```bash
git pull origin dev
```

Then retrieve the DVC-managed data corresponding to the current project version:

```bash
python -m dvc pull
```

`data/interim/` and `data/processed/` are versioned through DVC. The original
images in `data/raw/` are downloaded from Kaggle and are not stored in the DVC
remote.

If `data/raw/` is not available locally:

```bash
python -m gan_busters.data_pipeline.download
```

### 3. Reproduce data pipeline changes

If code or dependencies affecting the data pipeline have changed:

```bash
python -m dvc repro
```

Verify the pipeline state with:

```bash
python -m dvc status
```

### 4. Publish changes

If `data/interim/` or `data/processed/` changed after reproducing the pipeline,
push the new DVC artifacts:

```bash
python -m dvc push
```

Then commit and push the corresponding code and DVC metadata:

```bash
git add .
git commit -m "commit_title" -m "commit_description"
git push
```

`dvc push` is only required when DVC-managed artifacts have changed.

---

## First-Time Setup

This project uses **Python 3.10.x** and a **pip-based virtual environment**
named `gan-busters`.

### 1. Environment setup

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

#### Option 1 — Makefile

On Linux/macOS, WSL, or Git Bash, the environment can be created using the
provided Makefile.

This requires:

- GNU Make
- `virtualenvwrapper`
- Python 3.10 available to the Makefile

Create and activate the environment:

```bash
make create_environment
workon gan-busters
```

Install the project dependencies:

```bash
make requirements
```

> The Makefile must resolve `PYTHON_INTERPRETER` to a Python 3.10 interpreter.

#### Option 2 — Python `venv`

##### Linux/macOS/WSL

```bash
mkdir -p ~/.virtualenvs
python3.10 -m venv ~/.virtualenvs/gan-busters
source ~/.virtualenvs/gan-busters/bin/activate
```

##### Windows PowerShell

```powershell
New-Item -ItemType Directory -Force -Path "$HOME\.virtualenvs"
py -3.10 -m venv "$HOME\.virtualenvs\gan-busters"
& "$HOME\.virtualenvs\gan-busters\Scripts\Activate.ps1"
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify the environment:

```bash
python --version
```

The output should report:

```text
Python 3.10.x
```

### 2. DVC and data setup

Once the environment and dependencies are installed, choose one of the
following approaches.

#### Use the existing validated dataset

Recommended when the goal is model development:

```bash
python -m dvc pull
python -m gan_busters.data_pipeline.download
```

This retrieves the versioned records in `data/interim/` and `data/processed/`
from the DVC remote and downloads the original CIFAKE images into `data/raw/`.

#### Reproduce the complete data pipeline

To regenerate all derived data from the original CIFAKE source:

```bash
python -m dvc repro
```

DVC executes the complete pipeline defined in `dvc.yaml`:

```text
download_data
      ↓
   data/raw
      ↓
 inspect_data
      ↓
 data/interim
      ↓
data_integrity
      ↓
data/processed
```

The resulting `data/interim/` and `data/processed/` outputs are managed and
versioned by DVC. `data/raw/` participates in the pipeline but is not stored in
the DVC remote.


### 3. MLflow and DagsHub setup

The project uses **MLflow** for experiment tracking and **DagsHub** as the remote
tracking server and experiment visualization interface.

Before running model experiments for the first time, verify that the local
environment can authenticate with DagsHub and successfully log MLflow runs.

With the `gan-busters` environment activated and from the project root, run:

```bash
python -m gan_busters.modeling.smoke_test
```

On the first execution, DagsHub may request authentication. Follow the
instructions displayed in the terminal and authorize access to the project
repository.

The smoke test does not train a model or modify the dataset. It only creates a
small test experiment and logs dummy parameters and metrics through the same
tracking utilities used by the modeling pipeline.

After the command finishes, open the **Experiments** section of the project
repository in DagsHub.

A successful setup should show the `smoke-test` experiment containing a
`connection-test` run with the test parameter and metric logged by the script.

If the run appears correctly in DagsHub, the connection is working:

```text
GAN-Busters
    ↓
gan_busters/modeling/tracking.py
    ↓
MLflow
    ↓
DagsHub
```

The environment is then ready to track the model-selection experiments executed
through the GAN-Busters CLI.

---

## Project Organization

The project structure is based on the
[Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/)
template and has been adapted to the needs of this project.

```text
├── Makefile
├── README.md
├── requirements.txt          <- Python dependencies required to reproduce the environment
├── dvc.yaml                  <- DVC pipeline definition
├── dvc.lock                  <- Versioned pipeline dependencies and outputs
├── pyproject.toml            <- Project and tool configuration
│
├── .dvc                    
│   ├── .gitignore
│   └── config
| 
├── data
│   └── dataset_card.md       <- Dataset documentation, provenance, limitations, and usage
│
├── docs                      <- Project documentation
│   ├── index.md
│   └── getting-started.md 
│
├── models
│   └── model_card.md         <- Model architecture, intended use, evaluation, and limitations
│
├── notebooks                 <- Exploration, validation, and experimentation notebooks
│   └── 01_data_integrity_analysis.ipynb 
│
├── references
│   ├── README.md             <- Centralized datasets, papers, standards, and related work
│   └── figures               <- Figures used in project documentation
│
├── reports
│   ├── figures               <- Generated report figures
│   └── tables                <- Generated report tables
│
└── gan_busters               <- Project source code
    │
    ├── data_pipeline
    │   ├── __init__.py
    │   ├── corrupt.py
    │   ├── download.py       <- Download the original CIFAKE dataset
    │   ├── inspect.py        <- File-level inspection and metadata extraction
    │   └── data_integrity.py <- Duplicate handling, leakage prevention, and dataset splitting
    │
    └── modeling
        ├── __init__.py
        ├── config.py         <- Shared paths, defaults, and project
        ├── main.py           <- Central CLI entry point and execution routing
        └── plots.py          <- Visualization utilities
```
