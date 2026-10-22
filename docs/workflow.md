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