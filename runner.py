import logging
from preprocessing import load_and_clean_data, preprocess_features
from supervised import (
    train_logistic_regression,
    train_logistic_regression_from_scratch,
    train_decision_tree,
    train_random_forest,
    train_xgboost,
)
from neural_network import train_neural_network_variants
from unsupervised_anomaly import run_kmeans_clustering, run_gaussian_anomaly_detection

# === Setup logging ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class MLRunner:
    def __init__(self, data_path="data/friday_afternoon.csv"):
        logging.info(f"Loading and preprocessing data from: {data_path}")
        df = load_and_clean_data(data_path)
        self.X, self.y, self.feature_names = preprocess_features(df)
        logging.info("Data loaded and preprocessed successfully.")

    def run(self, method):
        logging.info(f"Method selected: {method}")

        if method == "logistic":
            train_logistic_regression(self.X, self.y)
        elif method == "logistic_scratch":
            train_logistic_regression_from_scratch(self.X, self.y)
        elif method == "decision_tree":
            train_decision_tree(self.X, self.y)
        elif method == "random_forest":
            train_random_forest(self.X, self.y, self.feature_names)
        elif method == "xgboost":
            train_xgboost(self.X, self.y, self.feature_names)
        elif method == "neural":
            train_neural_network_variants(self.X, self.y)
        elif method == "kmeans":
            run_kmeans_clustering(self.X, self.y)
        elif method == "anomaly":
            run_gaussian_anomaly_detection(self.X, self.y)
        else:
            logging.error(f"Unsupported method: {method}")
            raise ValueError(f"❌ Unsupported method: {method}")

        logging.info(f"✅ Finished running method: {method}")

# === CLI entry point ===
if __name__ == "__main__":
    import sys

    method = "logistic"
    if len(sys.argv) > 2 and sys.argv[1] in ("--model", "--method"):
        method = sys.argv[2]

    logging.info(f"🚀 Starting MLRunner with method: {method}")
    MLRunner().run(method)
