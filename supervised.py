import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from sklearn.model_selection import train_test_split, learning_curve, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

# Setup logging and output
OUTPUT_FOLDER = "output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def save_plot(filename):
    path = os.path.join(OUTPUT_FOLDER, filename)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    logging.info(f"✅ Plot saved: {path}")

def plot_learning_curve(model, X, y, title, filename):
    logging.info(f"📈 Learning curve: {title}")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_sizes, train_scores, val_scores = learning_curve(model, X, y, cv=cv, scoring='f1', n_jobs=-1,
                                                            train_sizes=np.linspace(0.1, 1.0, 10))
    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Training F1')
    plt.plot(train_sizes, np.mean(val_scores, axis=1), 'o-', label='Validation F1')
    plt.title(title)
    plt.xlabel("Training Set Size")
    plt.ylabel("F1 Score")
    plt.grid(True)
    plt.legend()
    save_plot(filename)

def plot_roc_curve(model, X_test, y_test, title, filename):
    logging.info(f"📉 ROC curve: {title}")
    y_score = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    save_plot(filename)

# --- Logistic Regression (sklearn) ---
def train_logistic_regression(X, y):
    logging.info("Training: Logistic Regression (sklearn)")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    logging.info(classification_report(y_test, y_pred))
    logging.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap='Blues')
    plt.title("Logistic Regression (sklearn) Confusion Matrix")
    save_plot("confusion_logistic_sklearn.png")
    plot_roc_curve(model, X_test, y_test, "Logistic Regression ROC Curve", "roc_logistic_regression.png")
    plot_learning_curve(model, X, y, "Learning Curve - Logistic Regression (sklearn)", "learning_curve_logistic_sklearn.png")

# --- Logistic Regression (from scratch) ---
def sigmoid(z): return 1 / (1 + np.exp(-z))

def logistic_regression_gd(X, y, lr=0.01, epochs=1000, reg_lambda=0.0):
    m, n = X.shape
    X = np.hstack([np.ones((m, 1)), X])
    weights = np.zeros(n + 1)
    loss_history = []
    for _ in range(epochs):
        z = np.dot(X, weights)
        h = sigmoid(z)
        loss = -np.mean(y * np.log(h + 1e-15) + (1 - y) * np.log(1 - h + 1e-15))
        if reg_lambda > 0:
            loss += (reg_lambda / (2 * m)) * np.sum(weights[1:] ** 2)
        loss_history.append(loss)
        gradient = np.dot(X.T, (h - y)) / m
        if reg_lambda > 0:
            gradient[1:] += (reg_lambda / m) * weights[1:]
        weights -= lr * gradient
    return weights, loss_history

def predict_gd(X, weights):
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    return (sigmoid(np.dot(X, weights)) >= 0.5).astype(int)

def train_logistic_regression_from_scratch(X, y):
    logging.info("Training: Logistic Regression (From Scratch)")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    weights, loss_history = logistic_regression_gd(X_train, y_train, lr=0.1, epochs=1000, reg_lambda=0.1)
    y_pred = predict_gd(X_test, weights)
    logging.info(classification_report(y_test, y_pred))
    logging.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap='Blues')
    plt.title("Logistic Regression (From Scratch) Confusion Matrix")
    save_plot("confusion_logistic_from_scratch.png")
    plt.figure(figsize=(10, 4))
    plt.plot(loss_history, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs (From Scratch)")
    plt.grid()
    plt.legend()
    save_plot("loss_logistic_from_scratch.png")
    model = LogisticRegression(max_iter=1000)
    plot_learning_curve(model, X, y, "Learning Curve - Logistic Regression (From Scratch)", "learning_curve_logistic_from_scratch.png")

# --- Decision Tree ---
def train_decision_tree(X, y):
    logging.info("Training: Decision Tree")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = DecisionTreeClassifier(max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    logging.info(classification_report(y_test, y_pred))
    logging.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap='Blues')
    plt.title("Decision Tree Confusion Matrix")
    save_plot("confusion_decision_tree.png")
    plot_roc_curve(model, X_test, y_test, "Decision Tree ROC Curve", "roc_decision_tree.png")
    plot_learning_curve(model, X, y, "Learning Curve - Decision Tree", "learning_curve_decision_tree.png")

# --- Random Forest ---
def train_random_forest(X, y, feature_names):
    logging.info("Training: Random Forest")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    logging.info(classification_report(y_test, y_pred))
    logging.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap='Blues')
    plt.title("Random Forest Confusion Matrix")
    save_plot("confusion_random_forest.png")
    plot_roc_curve(model, X_test, y_test, "Random Forest ROC Curve", "roc_random_forest.png")
    plot_learning_curve(model, X, y, "Learning Curve - Random Forest", "learning_curve_random_forest.png")
    plt.figure(figsize=(10, 4))
    importances = model.feature_importances_
    indices = np.argsort(importances)[-10:]
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), feature_names[indices])
    plt.xlabel("Importance")
    plt.title("Top 10 Features - Random Forest")
    save_plot("importance_feature_random_forest.png")

# --- XGBoost ---
def train_xgboost(X, y, feature_names):
    logging.info("Training: XGBoost")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = XGBClassifier(eval_metric='logloss')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    logging.info(classification_report(y_test, y_pred))
    logging.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap='Blues')
    plt.title("XGBoost Confusion Matrix")
    save_plot("confusion_XGBoost.png")
    plot_roc_curve(model, X_test, y_test, "XGBoost ROC Curve", "roc_XGBoost.png")
    plot_learning_curve(model, X, y, "Learning Curve - XGBoost", "learning_curve_XGBoost.png")
    plt.figure(figsize=(10, 4))
    importances = model.feature_importances_
    indices = np.argsort(importances)[-10:]
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Importance")
    plt.title("Top 10 Features - XGBoost")
    save_plot("importance_feature_XGBoost.png")
