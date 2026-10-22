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