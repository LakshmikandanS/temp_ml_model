import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Import the clean, modular evaluation definitions
from src.evaluate import evaluate_model, plot_confusion_matrix, compare_models

# Configure standard console logging for the pipeline
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

def main():
    logging.info("Starting Machine Learning Pipeline...")
    logging.info("1. Loading and preparing data...")
    
    try:
        df = pd.read_csv("../Titanic-Dataset.csv")
    except FileNotFoundError:
        logging.error("Titanic-Dataset.csv not found in the root directory. Exiting.")
        return

    # Basic reproducible feature pipeline (Not part of evaluation concern)
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)
    
    features = ["Pclass", "Sex", "Age", "Fare"]
    X = df[features]
    y = df["Survived"]
    
    # Stratified split to ensure reliable testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    logging.info("2. Training baseline models...")
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train, y_train)

    tree_clf = DecisionTreeClassifier(max_depth=4, random_state=42)
    tree_clf.fit(X_train, y_train)

    logging.info("========================================")
    logging.info("3. EXECUTING PROFESSIONAL EVALUATION")
    logging.info("========================================\n")
    
    # Store metrics from multiple models for later comparison
    results_list = []

    # ---------------------------------------------------------
    # Evaluate: Logistic Regression
    # ---------------------------------------------------------
    log_metrics = evaluate_model(log_reg, X_test, y_test, "Logistic Regression")
    results_list.append(log_metrics)
    
    # Only compute y_pred internally for visuals when necessary
    log_y_pred = log_reg.predict(X_test)
    plot_confusion_matrix(y_test, log_y_pred, "Logistic Regression")

    # ---------------------------------------------------------
    # Evaluate: Decision Tree
    # ---------------------------------------------------------
    tree_metrics = evaluate_model(tree_clf, X_test, y_test, "Decision Tree")
    results_list.append(tree_metrics)
    
    tree_y_pred = tree_clf.predict(X_test)
    plot_confusion_matrix(y_test, tree_y_pred, "Decision Tree")

    # ---------------------------------------------------------
    # Final Model Comparison
    # ---------------------------------------------------------
    comparison_df = compare_models(results_list)
    print("\n--- FINAL MODEL COMPARISON ---\n")
    print(comparison_df.to_string())

if __name__ == "__main__":
    main()
