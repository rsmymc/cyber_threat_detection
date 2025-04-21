from sklearn.ensemble import RandomForestClassifier
from preprocessing import load_and_clean_data, preprocess_features
from neural_network import train_neural_network_variants
from unsupervised_anomaly import run_kmeans_clustering, run_gaussian_anomaly_detection
from supervised import train_logistic_regression, train_logistic_regression_from_scratch, train_decision_tree, train_random_forest, train_xgboost

def main():
    df = load_and_clean_data("data/friday_afternoon.csv")
    X, y, feature_names = preprocess_features(df)

    #train_logistic_regression(X, y)
    #train_logistic_regression_from_scratch(X, y)
    #train_decision_tree(X, y)
    #train_random_forest(X, y, feature_names)
    #train_xgboost(X, y, feature_names)
    #train_neural_network_variants(X, y)
    #run_kmeans_clustering(X, y)
    run_gaussian_anomaly_detection(X, y)

if __name__ == "__main__":
    main()
