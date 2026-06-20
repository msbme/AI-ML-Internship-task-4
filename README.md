# End-to-End Telco Customer Churn Prediction Pipeline

## Objective
The goal of this project is to build a reusable, robust, and production-ready machine learning pipeline using Scikit-Learn's `Pipeline` and `ColumnTransformer` APIs to predict customer churn. 

Customer retention is vital for telecom operators. This system acts as an early-warning mechanism to identify high-risk customers, allowing business retention teams to take proactive mitigation measures before a customer switches to a competitor.

---

## Methodology & Engineering Approach

The solution is entirely automated within a single, unified execution block to enforce clean code architecture and avoid data leakage.

### 1. Data Cleaning & Feature Engineering
* **Identifier Removal:** Dropped non-predictive columns (`customerID`) to avoid noise.
* **Type Conversion Safety:** Formatted the `TotalCharges` field by converting missing/empty string spaces (`" "`) safely into numeric `NaN` values.
* **Target Encoding:** Mapped the target column `Churn` (`Yes`/`No`) into a binary representation (`1`/`0`).

### 2. Preprocessing Architecture
Using Scikit-Learn `Pipeline` wrappers, the dataset features are handled explicitly based on statistical types:
* **Numerical Features** (`SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges`): Imputed missing entries using the **Median** strategy to protect against outliers, followed by **StandardScaler** to normalize value distributions.
* **Categorical Features** (15 columns including `Contract`, `InternetService`, etc.): Imputed missing parameters using the **Most Frequent (Mode)** strategy, followed by **OneHotEncoder** configured to gracefully handle unknown categories during inference (`handle_unknown='ignore'`).
* **ColumnTransformer Union:** Encapsulated both workflows together to form a clean upstream preprocessing matrix.

### 3. Model Development & Hyperparameter Optimization
To balance speed and complexity, an exhaustive grid search (`GridSearchCV`) with 5-fold cross-validation was run across two distinct algorithmic architectures:
* **Logistic Regression** (optimized with L1/L2 penalties via `liblinear`)
* **Random Forest Classifier** (optimized across depths and estimator counts)

**Handling Class Imbalance:** Because the dataset inherently features fewer churning customers than loyal ones, `class_weight='balanced'` was passed dynamically to prevent the models from defaulting to the majority class.

---

## Key Results & Observations

### Model Selection
* **Winning Architecture:** `LogisticRegression`
* **Best Cross-Validation ROC-AUC:** **0.8455**
* **Test Set Verification ROC-AUC:** **0.8406**

The negligible performance gap (~0.005) between Cross-Validation and Test phases demonstrates that the preprocessing pipeline successfully neutralized overfitting.

### Evaluation Metrics (Test Dataset)

```text
Classification Report:
              precision    recall  f1-score   support

           0       0.90      0.72      0.80      1035
           1       0.51      0.78      0.61       374

    accuracy                           0.74      1409
   macro avg       0.70      0.75      0.71      1409
weighted avg       0.80      0.74      0.75      1409
