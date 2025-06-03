# Bitcoin-Price-Prediction.

This project is a machine learning-based approach to predicting Bitcoin prices using historical data from 2017 to 2023. It leverages a variety of classification models to forecast trends in Bitcoin's closing price.

## 🧠 Project Objectives

- Load and explore historical Bitcoin trading data
- Preprocess and engineer features
- Apply classification algorithms to predict price movement
- Evaluate model performance using confusion matrices and other metrics

## 🗃️ Dataset

Dataset Link: https://www.kaggle.com/datasets/jkraak/bitcoin-price-dataset

The dataset used spans Bitcoin transactions from 2017 to 2023. It includes features like:

- `timestamp`
- `open`, `high`, `low`, `close` prices
- `volume`
- `number_of_trades`
- `taker_buy_volume` and related quote volumes

> Note: The CSV file used (`bitcoin_2017_to_2023.csv`) is large and has been excluded from the GitHub repository. Consider downloading from the given kaggle link.

## 📊 Exploratory Data Analysis

The notebook performs:

- Data inspection using `.head()`, `.info()`, and `.describe()`
- Null value checks and basic cleaning
- Feature engineering to derive price movement classes

## 🧪 Models Implemented

- Logistic Regression
- Decision Tree Classifier
- K-Nearest Neighbors (KNN)
- Histogram-based Gradient Boosting

Each model is trained and evaluated, with metrics and confusion matrices used for comparison.

## 📁 Project Structure
bitcoin-price-prediction/
│
├── data/ # Raw data (excluded from repo)
├── models/ # Trained models (.pkl/.zip)
├── notebooks/
│ └── Project1_Bitcoin_Price_Prediction.ipynb
├── reports/
│ └── Confusion Matrix of Models.png
├── src/ # Source code (if applicable)
├── requirements.txt # Python dependencies
├── environment.yml # Conda environment file
└── README.md


## 🚀 Getting Started

### Prerequisites

- Python 3.10
- Recommended to use conda or virtualenv

### Install Dependencies

pip install -r requirements.txt
or
conda env create -f environment.yml
conda activate env-name
