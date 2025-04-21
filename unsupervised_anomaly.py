import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.covariance import EllipticEnvelope

# === Setup output folder and logging ===
OUTPUT_FOLDER = "output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def save_plot(filename):
    path = os.path.join(OUTPUT_FOLDER, filename)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    logging.info(f"Saved plot: {path}")
    return path

def run_kmeans_clustering(X, y_true, n_clusters=2):
    logging.info("=== K-Means Clustering ===")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)

    ari = adjusted_rand_score(y_true, cluster_labels)
    sil_score = silhouette_score(X, cluster_labels)

    logging.info(f"K-Means ARI: {ari:.4f}, Silhouette Score: {sil_score:.4f}")

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=cluster_labels, palette='Set2', legend='full')
    plt.title(f'K-Means Clustering (k={n_clusters})\nARI: {ari:.2f}, Silhouette Score: {sil_score:.2f}')
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    save_plot("kmeans_clustering.png")

def run_gaussian_anomaly_detection(X, y_true):
    logging.info("=== Gaussian Anomaly Detection ===")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_benign = X[y_true == 0]
    X_benign_scaled = scaler.transform(X_benign)

    model = EllipticEnvelope(contamination=0.1, support_fraction=1.0, random_state=42)
    model.fit(X_benign_scaled)

    y_pred = model.predict(X_scaled)
    y_pred_binary = (y_pred == -1).astype(int)
    y_true_binary = y_true

    report = classification_report(y_true_binary, y_pred_binary)
    cm = confusion_matrix(y_true_binary, y_pred_binary)

    logging.info("Classification Report (Gaussian Anomaly Detection):")
    for line in report.split("\n"):
        logging.info(line)
    logging.info(f"Confusion Matrix:\n{cm}")

    plt.figure(figsize=(5, 4))
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap='Blues')
    plt.title("Confusion Matrix - Gaussian Anomaly Detection")
    save_plot("confusion_gaussian_anomaly.png")

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_pred_binary, palette='Set1', alpha=0.6)
    plt.title("Gaussian Anomaly Detection (Elliptic Envelope)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend(title='Predicted Anomaly (1=Attack)')
    save_plot("gaussian_anomaly_detection.png")
