# Bitcoin Price Trend Classification (2017–2023)

This project presents a **comparative machine learning pipeline** for predicting whether the **Bitcoin price will go up (1)** or **not (0)**, using OHLCV data from 2017 to 2023. It is built for scalability and performance, operating on ~3 million samples with support for fast inference and real-time data simulation.

---

## Objective

- **Binary classification task** – predict directional price movement
- **Scalable ML pipeline** – built to train and infer on millions of rows
- **Model comparison** – Logistic Regression, Random Forest, and HistGradientBoosting
- **Minimal production-ready interface** – using synthetic or live input data

---

## Project Structure

Project/
├── data/
│ └── bitcoin_2017_to_2023.csv
├── models/
│ ├── lr_fitted.joblib
│ ├── rfecv_rfc.joblib
│ └── rob_scaler.joblib
├── utils.py
├── training_pipeline.py
├── inference.py
├── environment.yaml
├── requirements.txt
└── Project1_Bitcoin_Price_Prediction.ipynb


---

## Environment Setup

You can use either **Conda** or **pip** to get started.

### Conda (Recommended)

conda env create -f environment.yaml
conda activate btc-classifier

### pip + Virtualenv
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

# How to Use
## Train Models
python training_pipeline.py
* Loads and preprocesses ~3M records from data/bitcoin_2017_to_2023.csv

* Trains 3 models: Logistic Regression, Random Forest, HistGradientBoosting

* Applies outlier removal, feature engineering, RFECV, and scaling

* Saves models to models/

# Run Inference
python inference.py

* Simulates inference on synthetic OHLCV data

* Preprocesses input and applies same transformation pipeline

* Predicts with a pre-trained model (lr_fitted.joblib)

* Output: "Bitcoin Price will go up" or "Bitcoin Price will not go up"

# Models Compared
| Model                | Accuracy (%) | F1 Score (%) | Training Speed | Inference Speed |
| -------------------- | ------------ | ------------ | -------------- | --------------- |
| Logistic Regression  | \~91         | \~90         |   Very Fast    |  Very Fast      |
| HistGradientBoosting | \~94.5       | \~94.5       |   Moderate     |  Fast           |
| Random Forest        | \~94         | \~94         |   Moderate     |  Fast           |

All models achieve high accuracy. You can choose based on your use case (speed vs accuracy vs interpretability).

# Feature Engineering
* utils.py provides a clean, reproducible pipeline that includes:

* Temporal features: month, day, hour, minute

* Price-based deltas: close-open, high-low

* Wick-based features: wick_length_high, wick_length_low

* Volume-driven metrics: volume_delta, trade_activity_rate, buy_ratio

* Feature selection using RFECV

* Robust scaling

# Real-Time Compatibility
### The current setup uses synthetic OHLCV data via:
from utils import get_inference_data

### To connect to live exchanges:
Replace get_inference_data() with your API logic (e.g., Binance REST API)
Keep the rest of the inference logic unchanged
This makes the pipeline instantly extensible to real-time deployments.

# Reports & Visualization
* Model metrics: ROC, Precision-Recall, Confusion Matrix
* Feature distributions (histograms, boxplots)
* Feature importance scores (via SHAP or RFECV)
* Saved in /reports/ (if run during experimentation).

## License
This project is licensed under the MIT License.

## Author
Ajinkya Tamhankar
ajinkya.tamhankar18@gmail.com
Aspiring Researcher in AI & Computational Neuroscience

## Future Enhancements
* Live exchange integration (Binance, AlphaVantage)
* FastAPI-based deployment for RESTful inference
* Auto-retraining hooks with drift monitoring
* Streamlit or Dash dashboard for predictions

Designed for large-scale learning, real-time inference, and educational value in financial ML.