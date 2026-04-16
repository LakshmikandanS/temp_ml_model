# Production-Grade Adaptive Movie Revenue Prediction Architecture (v4)

## Overview
This document outlines the v4 architecture for the movie revenue prediction pipeline. Transitioning from rigid, manual budget-based segmentation (v3) to a fully adaptive, data-driven system, this upgrade addresses the extreme heteroscedasticity of Hollywood revenues. The goal is to aggressively reduce the ~100M RMSE baseline by employing learned segmentation, out-of-fold smoothed encoding, meta-learning (stacking), residual correction, and uncertainty estimation.

---

## 1. Advanced Feature Engineering & Encodings (Anti-Overfitting layer)
Before modeling, we must construct robust features that capture the complex relationships in movie data without overfitting to rare occurrences.

*   **Temporal & Structural Features:**
    *   `release_month` and `release_day_of_week`.
    *   Boolean flags: `is_summer_release` (May-August), `is_holiday_release` (Nov-Dec).
*   **Logarithmic Transformations:**
    *   `log_budget = np.log1p(budget)`
    *   `log_votes = np.log1p(votes)`
*   **Interaction Features:**
    *   `hype_index = log_budget * log_votes`
    *   `quality_momentum = score * log_votes`
*   **Experience Tracking:**
    *   `director_movie_count`, `star_movie_count`, `writer_movie_count` (calculated prior to target encoding).
*   **Smoothed Out-of-Fold (OOF) Target Encoding:**
    *   Replaces naive target mapping. 
    *   Formula: `(mean_target * count + global_mean * smoothing) / (count + smoothing)`
    *   Implementation uses strict K-Fold splits during training to prevent data leakage from rare directors/stars.

---

## 2. Learned Segmentation (Data-Driven Tiering)
Manual budget thresholds are replaced with data-driven **Target Quantile Segmentation** to group movies with similar revenue behaviors.

*   **Segment Definitions (based on historical training revenue):**
    *   **Low Tier:** Bottom 30% of revenue.
    *   **Mid Tier:** Middle 40% of revenue.
    *   **High Tier:** Top 30% of revenue.
*   **The Stage 1 Classifier:**
    *   A robust classifier (e.g., LightGBM) is trained to predict which of these 3 quantiles a movie belongs to.
    *   Outputs soft probabilities: $[P(Low), P(Mid), P(High)]$.
    *   *Leakage Prevention:* Target boundaries are calculated *only* on the training set.

---

## 3. Base Models & Blockbuster Specialist (Stage 2)
The regression layer trains independent estimators optimized for specific data distributions.

*   **Segment-Specific Regressors:**
    *   **Low_Model:** Trained only on the bottom 30% slice.
    *   **Mid_Model:** Trained only on the middle 40% slice.
    *   **High_Model:** Trained only on the top 30% slice.
*   **Blockbuster Specialist Model:**
    *   An auxiliary model trained strictly on the **Top 10%** highest-grossing movies.
    *   Activates/Overrides when the classifier predicts "High" with extreme confidence, addressing the historical underestimation of massive outliers (e.g., Marvel movies, Avatar).

---

## 4. Meta-Learner Layer (Stacking)
We abandon simple linear soft-routing in favor of a Level-2 Stacking Meta-Learner.

*   **Inputs to Meta-Learner:**
    *   Base predictions: `[pred_low, pred_mid, pred_high]`
    *   Routing probabilities: `[P_low, P_mid, P_high]`
    *   (Optional) Blockbuster prediction: `[pred_blockbuster]`
*   **Model:** A robust meta-model (Ridge Regression or shallow LightGBM/XGBoost) learns the optimal non-linear combination of these predictions.
*   **Output:** `meta_prediction` (A highly stabilized revenue estimate).

---

## 5. Residual Correction & Uncertainty Estimation (Stage 3)
A post-processing layer to correct bias and quantify variance.

*   **Residual Correction Model:**
    *   Calculate absolute residuals on validation folds: `residual = actual - meta_prediction`
    *   Train a separate LightGBM/RandomForest to predict the `residual` based on the original features.
    *   Final Point Estimate: `y_final = meta_prediction + predicted_residual`
*   **Uncertainty Estimation Model:**
    *   Train another model to predict the *absolute error*: `abs_error = |actual - y_final|`
    *   Final Output format: `$120M ± $40M` (Providing confidence bounds for the Studio).

---

## 6. Evaluation Framework
Robust performance metrics tracked continuously.

*   **Metrics Tracked:** 
    *   Global RMSE, MAE, R²
    *   Segment-wise RMSE (to ensure we actually improved blockbusters).
    *   **Weighted RMSE:** Weights applied using `log(budget)` to penalize errors on high-stakes movies appropriately.
    *   Baseline vs. Pipeline V4 comparison matrix.

---

## 7. "Lucy" Hardware Optimizations (Strict Constraints)
To ensure everything executes locally without freezing the 16GB i5 machine:

1.  **Sequential Execution:** Models (Classifier -> Regressors -> Meta -> Residual -> Uncertainty) are trained one by one, not concurrently.
2.  **Explicit Garbage Collection:** `gc.collect()` will be invoked after each model stage.
3.  **Hyperparameter Bounds:** 
    *   `RandomizedSearchCV` limited to `n_iter <= 10`.
    *   Trees: `n_estimators <= 400` max.
4.  **Parallelism Limit:** Enforce `n_jobs=2` globally across sub-models.
5.  **Early Stopping:** Used heavily on LightGBM/XGBoost to prevent unnecessary tree building.
