# Production ML Plan: Segmented Movie Revenue Prediction (v3)

## 📌 Executive Summary
The current monolithic Random Forest model (R² ≈ 0.73, RMSE ≈ 101M) struggles with the extreme heteroscedasticity of movie revenues. A $5M indie horror film does not follow the same financial physics as a $250M Marvel blockbuster. 

To drastically reduce RMSE and improve prediction stability across all scales, we are transitioning to a **Soft-Routing Segmented Ensemble**. This architecture will use a classifier to compute the probability of a movie belonging to a specific financial tier, and then linearly combine the predictions of distinct, specialized regression models tuned for those exact tiers.

---

## 🛠️ Phase 1: Robust Feature Engineering
Before segmentation, we need features that isolate quality, hype, and historical performance.

1. **Target Transformation:** Log-transform the target `y_train_log = np.log1p(gross)` to stabilize variance during training.
2. **Derived Metrics:**
   - `budget_per_min` = `budget` / `runtime`
   - `votes_per_year` = `votes` / (current_year - `year`)
3. **Advanced Target Encoding (Crucial for Segmenting):**
   - Compute out-of-fold `director_avg_gross`, `star_avg_gross`, and `company_avg_gross`.
   - *Alternative:* Target encode based on win-rate (probability of a movie grossing > 3x its budget).
4. **Interactions:**
   - `hype_index` = `log(budget)` × `log(votes)`
   - `quality_momentum` = `score` × `log(votes)`

---

## 🗂️ Phase 2: Meaningful Segment Definition & Classification (Stage 1)
Instead of forcing the model to learn boundary edges, we define distinct regimes based primarily on `budget` (and optionally filtered by historical studio tier).

### The 3 Regimes:
* **Segment 0 (Low Budget):** Budget < $10M (High ROI potential, relies on score/votes).
* **Segment 1 (Mid Budget):** Budget $10M – $100M (The hardest to predict, standard studio releases).
* **Segment 2 (High Budget / Blockbuster):** Budget > $100M (Global theatrical plays, relies on franchise, CGI, and star power).

### The Router (Classifier):
* **Model:** A fast, calibrated `LightGBMClassifier` or `LogisticRegression`.
* **Input:** The engineered features.
* **Output:** Soft probabilities `[P_low, P_med, P_high]` for every movie.

---

## 🧠 Phase 3: Segment-Specific Regressors (Stage 2)
We train three isolated regression models. Each model *only* sees training data from its assigned segment.

* **Model_Low `< $10M`:** 
  * *Algorithm:* `GradientBoostingRegressor` or `Ridge Regression`.
  * *Reasoning:* Low-budget data is noisy and prone to extreme outliers (e.g., *Paranormal Activity*). Simpler models often prevent extreme overfitting here.
* **Model_Med `$10M - $100M`:**
  * *Algorithm:* `RandomForestRegressor`.
  * *Reasoning:* Handles complex non-linear relationships well without deep tuning.
* **Model_High `> $100M`:**
  * *Algorithm:* `XGBoost` or `LightGBMRegressor`.
  * *Reasoning:* Needs to precisely map the upper ceiling of box office returns based on franchise momentum and studio scale.

---

## 🔀 Phase 4: Soft-Routing Assembly
Hard boundaries (e.g., exactly at $9.99M vs $10.01M) create massive prediction cliffs. We use the classifier's probabilities to blend the outputs.

For a new movie `X`:
1. Get routing probabilities: `P_low, P_med, P_high = Classifier.predict_proba(X)`
2. Get raw log-predictions: `Pred_low`, `Pred_med`, `Pred_high`
3. Exponentiate back to USD: `y_hat_low = expm1(Pred_low)`, etc.
4. **Final Prediction:** 
   `y_final = (P_low * y_hat_low) + (P_med * y_hat_med) + (P_high * y_hat_high)`

---

## 💻 Phase 5: Hardware-Aware Execution (The "Lucy" Constraints)
Since the system has 16GB RAM and a mid-range CPU with thermal limits:
* **No massive parallel grids:** Use `RandomizedSearchCV(n_iter=5, cv=3)` or sequential `Optuna` trials.
* **CPU limits:** Global `n_jobs=2` to leave overhead and prevent thermal throttling.
* **Model sizes:** Limit `n_estimators <= 400` with `early_stopping_rounds=20` for boosting models.
* **Memory Management:** Use `gc.collect()` after training each segment's model.

---

## 📊 Phase 6: Evaluation & Error Analysis
To prove this architecture is superior to the v2 pipeline, the script will output:

1. **Global Metrics Match-up:** Overall R², RMSE, and MAE compared directly against the single RandomForest baseline.
2. **Regime-Specific Accuracy:** RMSE isolated for Low, Med, and High budget movies.
3. **Error Distribution (Before vs After):** 
   - A dataframe explicitly showing MAE in ranges (`<10M`, `10M-50M`, `50M+`).
   - Expectation: We should see a massive drop in RMSE because the `Model_High` is no longer being conservative to appease the `Model_Low` data points, and `Model_Low` isn't wildly over-predicting due to blockbuster skew. 

---
*Ready to commence coding when authorized.*