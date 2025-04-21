PROJECT_NAME=cyber-threat-detection
DOCKER_RUN=docker run --rm -v ${PWD}/output:/app/output -v ${PWD}/data:/app/data $(PROJECT_NAME)

# === Build and Shell ===
build:
	docker build -t $(PROJECT_NAME) .

shell:
	docker run --rm -it -v ${PWD}/output:/app/output --entrypoint bash $(PROJECT_NAME)

# === Supervised Models ===
logistic_regression:
	$(DOCKER_RUN) python runner.py --model logistic

logistic_scratch:
	$(DOCKER_RUN) python runner.py --model logistic_scratch

decision_tree:
	$(DOCKER_RUN) python runner.py --model decision_tree

random_forest:
	$(DOCKER_RUN) python runner.py --model random_forest

xgboost:
	$(DOCKER_RUN) python runner.py --model xgboost

# === Neural Networks ===
neural_network:
	$(DOCKER_RUN) python runner.py --model neural

# === Unsupervised / Anomaly Detection ===
kmeans:
	$(DOCKER_RUN) python runner.py --method kmeans

anomaly:
	$(DOCKER_RUN) python runner.py --method anomaly

# === Batch Commands ===
all_supervised:
	$(MAKE) logistic_regression
	$(MAKE) logistic_scratch
	$(MAKE) decision_tree
	$(MAKE) random_forest
	$(MAKE) xgboost

all_unsupervised:
	$(MAKE) kmeans
	$(MAKE) anomaly

all_models:
	$(MAKE) all_supervised
	$(MAKE) neural_network
	$(MAKE) all_unsupervised
