import logging
from typing import Any, Dict, List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Configure module-level logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)

def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> Dict[str, Any]:
    """
    Evaluates a classification model on test data and logs core metrics.

    Args:
        model: Trained machine learning model (must implement .predict()).
        X_test: Testing features.
        y_test: True labels for the testing set.
        model_name: Name of the model for logging and reporting.

    Returns:
        A dictionary containing the computed metrics (Accuracy, Precision, Recall, F1).
    """
    logging.info(f"Evaluating: {model_name}")
    
    # Generate predictions
    y_pred = model.predict(X_test)
    
    # Compute metrics
    metrics = {
        "Model": model_name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1 Score": round(f1_score(y_test, y_pred), 4)
    }
    
    # Log results in a clean, structured format
    logging.info(f"Accuracy: {metrics['Accuracy']:.2f}")
    logging.info(f"Precision: {metrics['Precision']:.2f}")
    logging.info(f"Recall: {metrics['Recall']:.2f}")
    logging.info(f"F1 Score: {metrics['F1 Score']:.2f}")
    logging.info(f"Evaluation for {model_name} completed.\n")
    
    return metrics


def plot_confusion_matrix(y_true: pd.Series, y_pred: pd.Series, model_name: str) -> None:
    """
    Plots and displays a confusion matrix heatmap.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        model_name: Name of the model for the plot title.
    """
    logging.info(f"Generating confusion matrix plot for: {model_name}")
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    
    plt.title(f"Confusion Matrix: {model_name}")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.show()


def compare_models(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Compiles a list of model evaluation results into a pandas DataFrame.

    Args:
        results: List of dictionaries containing model metrics.

    Returns:
        A pandas DataFrame comparing all models cleanly.
    """
    logging.info("Compiling model comparison table...")
    comparison_df = pd.DataFrame(results)
    
    if "Model" in comparison_df.columns:
        comparison_df.set_index("Model", inplace=True)
        
    return comparison_df
