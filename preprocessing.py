import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Drop irrelevant columns
    drop_cols = ['Flow ID', 'Timestamp']
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    # Remove inf and NaNs
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Label encode the target
    le = LabelEncoder()
    df['Label'] = le.fit_transform(df['Label'])

    return df


def preprocess_features(df):
    X = df.drop('Label', axis=1)
    y = df['Label']
    feature_names = X.columns
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, feature_names
