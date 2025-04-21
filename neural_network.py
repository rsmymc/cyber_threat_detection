
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import os

OUTPUT_FOLDER = "output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def build_model(name, input_dim, activation='relu', layers=[64, 32]):
    print(f"\n--- Training {name} ---")
    model = Sequential()
    model.add(Dense(layers[0], input_shape=(input_dim,), activation=activation))
    model.add(Dropout(0.3))
    for units in layers[1:]:
        model.add(Dense(units, activation=activation))
        model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def train_neural_network_variants(X, y):
    print("\n=== Neural Network ===")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    configs = [
        ("Model A: ReLU 2-layer", 'relu', [64, 32]),
        ("Model B: tanh 2-layer", 'tanh', [64, 32]),
        ("Model C: ReLU 3-layer deep", 'relu', [128, 64, 32]),
    ]

    for name, act_fn, layer_config in configs:
        model = build_model(name, input_dim=X.shape[1], activation=act_fn, layers=layer_config)
        early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        history = model.fit(X_train, y_train,
                            validation_split=0.2,
                            epochs=30,
                            batch_size=512,
                            callbacks=[early_stop],
                            verbose=0)

        # Predict & Evaluate
        y_proba = model.predict(X_test)
        y_pred = (y_proba > 0.5).astype("int32")

        print(classification_report(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap='Blues')
        plt.title(f"{name} - Confusion Matrix")
        save_plot(f"confusion_{name.replace(':', '').replace(' ', '_')}.png")

        # ROC Curve
        plot_roc_curve_nn(y_test, y_proba, title=f"{name} - ROC Curve", filename=f"roc_{name.replace(':', '').replace(' ', '_')}.png")

        # Loss & Accuracy Plot
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title(f'{name} - Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Val Accuracy')
        plt.title(f'{name} - Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        save_plot(f"metrics_{name.replace(':', '').replace(' ', '_')}.png")

        # Save model
        model.save(os.path.join(OUTPUT_FOLDER, f"{name.replace(':', '').replace(' ', '_')}.keras"))

def save_plot(filename):
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, filename))
    plt.close()

def plot_roc_curve_nn(y_true, y_proba, title="ROC Curve", filename=None):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    if filename:
        save_plot(filename)
