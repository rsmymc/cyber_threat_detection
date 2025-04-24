
# Cyber Threat Detection Using Machine Learning

This project implements a full pipeline for detecting cyber threats using various machine learning techniques, including supervised learning, neural networks, and unsupervised anomaly detection.

It is structured for **Docker-based execution**, using a simple `Makefile` to trigger training, evaluation, and reporting of each model.

---

## 📂 Project Structure

```
📁 data/                     # Input data folder (e.g., friday_afternoon.csv)
📁 output/                   # Output plots, logs, saved models
📄 supervised.py             # Supervised ML models (logistic, RF, XGBoost, etc.)
📄 neural_network.py         # Deep learning models (ReLU, tanh variants)
📄 unsupervised_anomaly.py   # KMeans & Gaussian anomaly detection
📄 preprocessing.py          # Data cleaning and standardization
📄 runner.py                 # Main execution point (CLI)
📄 cyber_threat_detection.ipynb 
📄 requirements.txt
📄 Dockerfile
📄 Makefile
```

---

## 🚀 How to Run This Project

You can run this project in **four different ways**:

---

### 1️⃣ Recommended: Docker + Makefile (Cleanest)

```bash
make build                     # Build the Docker image
make logistic_regression       # Run a specific model
make all_models                # Run all models end-to-end
```

> Requires: `docker`, `make`

This automatically:
- Mounts `data/` and `output/`
- Executes `runner.py` inside the container

---

### 2️⃣ Docker Only (Without Makefile)

If you prefer direct Docker commands:

```bash
# Build image
docker build -t cyber-threat-detection .

# Run model
docker run --rm -v ${PWD}/data:/app/data -v ${PWD}/output:/app/output   cyber-threat-detection python runner.py --model xgboost
```

> Replace `--model xgboost` with any of the available methods:
> `logistic`, `logistic_scratch`, `random_forest`, `neural`, `kmeans`, `anomaly`, etc.

---

### 3️⃣ Local Python Execution (Outside Docker)

If you have Python + dependencies installed locally:

```bash
pip install -r requirements.txt
python runner.py --model neural
```

### 4️⃣ Run Jupyter Notebook

Use the full visual notebook with plots, metrics, and model results:

```bash
jupyter notebook cyber_threat_detection.ipynb
```

---

## 🔍 Available Models

Use any of the following methods with `--model` or `--method`:

| Method Flag         | Model Description                   |
|---------------------|--------------------------------------|
| `logistic`          | Logistic Regression (sklearn)        |
| `logistic_scratch`  | Logistic Regression from scratch     |
| `decision_tree`     | Decision Tree Classifier             |
| `random_forest`     | Random Forest Classifier             |
| `xgboost`           | XGBoost Classifier                   |
| `neural`            | Neural Networks (3 variants)         |
| `kmeans`            | Unsupervised Clustering              |
| `anomaly`           | Gaussian Anomaly Detection           |

---

## 📊 Output

Each run saves artifacts to the `output/` folder:
- Confusion matrices
- ROC curves
- Training curves
- Cluster visualizations
- Saved model weights (.keras)
- Console logs

---

## 🧪 Evaluation

Each model is evaluated using:
- **Precision, Recall, F1-Score**
- **Confusion Matrix**
- **ROC AUC**
- **Learning Curves**
- **Loss & Accuracy Curves** (for neural nets)
- **Silhouette Score & ARI** (for KMeans)
- **PCA Visualization** (for clustering & anomaly)

---
