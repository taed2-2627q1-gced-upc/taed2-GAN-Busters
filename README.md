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
- [`docs/first-time-setup.md`](docs/first-time-setup.md) - environment installation, data setup, and MLflow/DagsHub configuration
- [`docs/workflow.md`](docs/workflow.md) - daily development workflow, Git/DVC synchronization, and publishing changes
- [GAN-Busters Web Application](http://nattech.fib.upc.edu:40510/Statistics) — public application, available temporarily until January 2027

---

## Project Organization

The project structure is based on the
[Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/)
template and has been adapted to the needs of this project.

```text

├── README.md                 <- Project overview and documentation
├── LICENSE                   <- MIT License
├── Makefile                  <- Convenience commands for project automation
├── pyproject.toml            <- Project and tool configuration
├── requirements.txt          <- Main Python dependencies
├── requirements_front.txt    <- Streamlit frontend dependencies
├── requirements_back.txt     <- FastAPI backend dependencies
├── compose.yaml              <- Docker Compose deployment configuration
├── Dockerfile.api            <- FastAPI container definition
├── Dockerfile.frontend       <- Streamlit container definition
├── dvc.yaml                  <- DVC pipeline definition
├── dvc.lock                  <- Versioned pipeline dependencies and outputs
│
├── gan_busters               <- Project source code
│   ├── __init__.py
│   ├── config.py             <- Shared paths and project configuration
│   ├── main.py               <- Central CLI entry point
│   ├── api                   <- FastAPI inference service
│   ├── data_pipeline         <- Data preparation and validation
│   ├── data_testing          <- Data quality and distribution checks using Great Expectations and Deepchecks
│   ├── external_evaluation.py <- Evaluation on external datasets
│   ├── modeling              <- CNN training, evaluation, inference, and MLflow tracking
│   └── pages                 <- Streamlit application pages
│
├── tests                     <- Automated unit and robustness tests
│   └── conftest.py           <- Shared pytest fixtures and test configuration
│
├── docs                      <- Setup and development documentation
│   ├── first-time-setup.md   <- Environment, DVC, and MLflow setup
│   └── workflow.md           <- Daily Git and DVC workflow
│
├── data
│   ├── external/          <- External evaluation datasets managed by DVC
│   └── dataset_card.md    <- Dataset documentation, provenance, and limitations
│
├── models
│   └── model_card.md         <- Model architecture, evaluation, and limitations
│
├── notebooks
│   └── 01_data_integrity_analysis.ipynb <- Data integrity and distribution analysis
│
├── reports                    <- Analysis results and generated reports
│   ├── figures/               <- Generated report figures
│   └── tables/                <- Generated report tables
│
└── references
    ├── README.md             <- Centralized sources and related work
    └── figures               <- Figures used in project documentation

```
