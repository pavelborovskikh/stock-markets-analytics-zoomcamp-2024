## Course Project

### Objective

The goal of this project is build an end-to-end machine learning system that will be able to create a recommender system for trading strategies.


## Problem statement

For the project, we are required to build an end-to-end algo-trading strategy.

For that, we accomplish the following steps:

* Select data sources (or APIs) to be used.
* Generate a unified combined dataset (one table from all sources).
* Perform necessary data transformations, create new derived features (e.g., dummies), and prepare the dataset for ML.
* Train one or several ML models to predict future returns.
* Define a trading strategy based on predictions and simulate it.
* Automate the solution.

## Deliverables

Included files:
* README.md
* Notebooks (research outcomes in Colab or Jupiter Notebooks),
* .py files with scripts for each step
* data workflow and command list to run it automatically
* requirements.txt

[!IMPORTANT]  Please don’t submit data files to GitHub. Instead, save them to a drive and provide the link, or make them available for full download from APIs.

# Local Automation Instructions

## Setting Up the Project Environment (in Terminal)

* Install virtual environment: `pip3 install virtualenv`
* Create a new virtual environment (venv): `virtualenv venv` (or run `python3 -m venv venv`)
* Activate the new virtual environment: `source venv/bin/activate`

* Install Ta-lib (on Mac):
  * Step 1: Install Homebrew (if not already installed): `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`
  * Step 2: Install TA-Lib using Homebrew: `brew install ta-lib`
  * Step 3: Install the ta-lib Python package (After the TA-Lib C library is installed, you can proceed to install the Python package using pip. Make sure you're in your virtual environment if you're using one):
`pip3 install ta-lib`
  * Step 4: Make sure you have Numpy of version earliar than 2, so that ta-lib can be successfully imported (e.g. "numpy==1.26.4" in requirements.txt). [LINK](https://stackoverflow.com/questions/78634235/numpy-dtype-size-changed-may-indicate-binary-incompatibility-expected-96-from)

* Install all requirements to the new environment (venv): `pip3 install -r requirements.txt`

## Running the Project

* Start the local Jupyter Server (after activating venv): `jupyter notebook` (you can check all servers running with `jupyter notebook list`)
* Open `system_test.ipynb` to check the system's operation:
  * From your web browser (navigate to http://localhost:8888/tree or similar)
  * Or via the VS Code UI / PyCHarm (specify the server address kernel) 
* Run `main.py` from the Terminal (or Cron) to simulate one new day of data.