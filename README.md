# Name me


# HOW TO RUN
make create_environment
conda activate src
make requirements
pip install dvc
pip install dvc_gdrrive
dvc pull (might be messy with loggin stuff)

## Project structure

The directory structure of the project looks like this:

```txt

├── Makefile                  <- Makefile with convenience commands like `make data` or `make train`
├── README.md                 <- The top-level README for developers using this project.
├── data
│   └── processed             <- The final, canonical data sets for modeling.
│
├── models                    <- Trained and serialized models, model predictions, or model summaries
│
├── pyproject.toml            <- Project configuration file
│
├── reports                   <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures               <- Generated graphics and figures to be used in reporting
│
├── dockerfiles               <- Docker files for deployment and training
|
├── config                    <- Config files
|
├── requirements.txt          <- The requirements file for reproducing the analysis environment
|
├── requirements_app.txt      <- The requirements file for deploying the FastAPI based app
|
├── requirements_gapp.txt     <- The requirements file for deploying the graphical based app
|
├── requirements_tests.txt    <- The requirements file for performing tests
|
├── requirements_dev.txt      <- The requirements file for reproducing the analysis environment
│
├── tests                     <- Test files
│
├── cloudbuild.yaml           <- YAML file with steps for using GCP services
│
├── gramai_app_service.yaml   <- YAML file with instructions to replace a service
│
├── src  <- Source code for use in this project.
│   │
│   ├── __init__.py           <- Makes folder a Python module
│   │
│   ├── data                  <- Scripts to download or generate data
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   │
│   ├── train_model.py        <- Script for training the model
│   ├── gramai_app.py         <- Script for deploying app with FastAPI
│   ├── gramai_gapp.py        <- Script for deploying app with Dash to produce a graphical interface for users
│   ├── README.md             <- The top-level README for developers using this project.
│   └── assets
│       │
│       └── style.css         <- CSS file to styling the user graphical app
│
└── LICENSE                   <- Open-source license if one is chosen
```
