# Explainable Loan Default Risk Prediction using Machine Learning and SHAP


This project builds an explainable machine learning system that predicts loan default risk before credit is issued, using the Home Credit Default Risk dataset combined with external credit bureau and payment history data. It is built for Credit Risk Managers at Kenyan lending institutions — including M-KOPA, Tala, Branch, and Equity Bank — who need both accurate default predictions and transparent, auditable explanations for every lending decision.

---

## Table of Contents

- [Business Problem](#business-problem)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Key Results](#key-results)
- [SHAP Explainability](#shap-explainability)
- [How to Reproduce](#how-to-reproduce)
- [Business Recommendations](#business-recommendations)
- [Technologies Used](#technologies-used)
- [Limitations](#limitations)
- [Author](#author)

---

## Business Problem

The Credit Risk Manager at M-KOPA Kenya makes daily decisions on whether to approve or reject loan applications — decisions that directly determine the financial health of the institution. With over 2 million active customers and a default rate of approximately 8%, approving even a small fraction of high-risk borrowers results in millions of shillings in unrecoverable losses annually. Manual credit assessment is slow, inconsistent, and fails to account for the complex interactions between income, debt burden, credit history, and repayment behaviour that predict default.

Machine learning solves this by learning patterns across 307,511 historical loan applications — including bureau records, payment history, and credit card behaviour — to produce a default probability score for each new applicant before a single shilling is disbursed. Kenyan digital lenders such as Tala, Branch, and Equity Bank already use data-driven credit scoring; this project builds a fully explainable version that meets both predictive and regulatory requirements, including the ability to justify every rejection to the applicant, the loan officer, and the regulator.

---

## Dataset

**Source:** Home Credit Default Risk — Kaggle Competition
**URL:** https://www.kaggle.com/competitions/home-credit-default-risk/data
**Size:** 307,511 loan applications across 5 linked files

| File | Description | Join Key |
|---|---|---|
| `application_train.csv` | Primary dataset — one row per loan application, includes the TARGET variable (1=default, 0=repaid) | `SK_ID_CURR` |
| `bureau.csv` | Credit history from other financial institutions reported to the credit bureau — multiple records per applicant | `SK_ID_CURR` |
| `previous_application.csv` | All previous loan applications submitted to Home Credit — approved, refused, or cancelled | `SK_ID_CURR`, `SK_ID_PREV` |
| `installments_payments.csv` | Repayment history for previous Home Credit loans — payment timing, amounts, and lateness | `SK_ID_PREV` |
| `credit_card_balance.csv` | Monthly credit card balance snapshots — credit utilisation and days past due | `SK_ID_CURR` |

> **Note:** The data is not included in this repository due to Kaggle's terms of use. See the [How to Reproduce](#how-to-reproduce) section for download instructions.

---

## Project Structure

```
loan-default-prediction/
│
├── data/   
│   ├── application_train.csv
│   ├── bureau.csv
│   ├── previous_application.csv
│   ├── installments_payments.csv
│   └── credit_card_balance.csv
│
├── images/                              
│   ├── target_distribution.png
│   ├── feature_distributions.png
│   ├── age_default_rate.png
│   ├── correlation_target.png
│   ├── payment_lateness.png
│   ├── model_comparison.png
│   ├── confusion_matrix.png
│   ├── roc_curves.png
│   ├── shap_summary.png
│   ├── shap_waterfall.png
│   └── permutation_importance.png
│
├── saved_models/                           # Trained model files
│   ├── xgb_loan_default_pipeline.joblib
│   └── preprocessor.joblib
│
├── loan_default_risk_prediction.ipynb      # Main analysis notebook
├── app.py                                  # Flask API for deployment
├── requirements.txt                        # All dependencies
├── .gitignore                              # Excludes data/ and saved_models/
└── README.md
```

---

## Methodology

The project follows the full data science workflow from raw data to deployed model.

### 1. Data Merging

All five datasets contain multiple records per applicant. Before merging, each external dataset was aggregated into a single summary row per applicant using group-level statistics. All aggregated tables were then joined onto `application_train` using left joins on `SK_ID_CURR`, preserving all 307,511 applicants including those with no external credit history.

**Final merged dataset:** 307,511 rows × 147 features

### 2. Data Cleaning

- Columns with more than 60% missing values were dropped — insufficient data to impute meaningfully
- The `DAYS_EMPLOYED` anomaly value of 365,243 (meaning unemployed or pensioner) was flagged as a new binary feature `EMPLOYED_ANOM` and replaced with NaN
- Negative day columns (`DAYS_BIRTH`, `DAYS_EMPLOYED`, `DAYS_REGISTRATION`, `DAYS_ID_PUBLISH`) were converted to positive year-based features
- `AMT_INCOME_TOTAL` was capped at the 99th percentile to remove extreme outlier values

### 3. Feature Engineering

Six domain-knowledge features were created to capture credit risk signals not present in raw columns:

| Feature | Formula | Why It Predicts Default |
|---|---|---|
| `DEBT_TO_INCOME` | `AMT_CREDIT / AMT_INCOME_TOTAL` | High ratio = borrower is financially stretched |
| `ANNUITY_TO_INCOME` | `AMT_ANNUITY / AMT_INCOME_TOTAL` | Monthly burden above 30% of income signals stress |
| `CREDIT_TO_GOODS` | `AMT_CREDIT / AMT_GOODS_PRICE` | Ratio above 1.0 means borrowing beyond the purchase value |
| `PAYMENT_RATE` | `AMT_ANNUITY / AMT_CREDIT` | Higher rate = shorter loan term = less accumulated risk |
| `INCOME_PER_PERSON` | `AMT_INCOME_TOTAL / CNT_FAM_MEMBERS` | Adjusts income for household size and dependents |
| `INST_PCT_LATE` | Mean of late payment flags per applicant | Past lateness is the strongest behavioural predictor |

### 4. Preprocessing Pipelines

scikit-learn `Pipeline` and `ColumnTransformer` were used to chain all preprocessing steps and prevent data leakage:

- **Numerical:** `SimpleImputer(strategy='median')` → `StandardScaler()`
- **Categorical:** `SimpleImputer(strategy='most_frequent')` → `OneHotEncoder(handle_unknown='ignore')`

The train/test split was performed **before** any preprocessing. All transformers were fitted on training data only, then applied identically to the test set.

### 5. Class Imbalance

With 91.9% repaid and 8.1% defaulted, a naive model achieves 92% accuracy by predicting everyone repays — catching zero actual defaulters. Two strategies were combined:

- **SMOTE** (Synthetic Minority Oversampling Technique) — applied inside the pipeline to training data only
- **`scale_pos_weight`** in XGBoost — set to `n_repaid / n_defaulted ≈ 11.4` to weight minority class appropriately

Primary evaluation metrics: **Recall** and **ROC-AUC** — not accuracy.

---

## Key Results

The champion model is **XGBoost Balanced** (`scale_pos_weight=15`), selected for achieving the best combination of Recall and ROC-AUC while meeting the business target of ROC-AUC ≥ 0.75.

| Model | Recall | Precision | ROC-AUC | Notes |
|---|---|---|---|---|
| Logistic Regression | ~60.0% | ~20.0% | ~0.680 | Linear baseline |
| Random Forest | ~65.0% | ~18.0% | ~0.730 | Ensemble baseline |
| XGBoost Baseline | 65.8% | 17.6% | 0.7607 | scale_pos_weight = 11.39 |
| **XGBoost Balanced** | **76.5%** | **15.0%** | **0.7609** | **CHAMPION — scale_pos_weight = 15** |
| XGBoost Cost-Optimised | 83.8% | 12.9% | 0.7605 | scale_pos_weight = 22.77 |

**Business translation of Recall 76.5%:**
The champion model correctly identifies 76 out of every 100 real defaulters before a loan is issued. At M-KOPA's scale, this prevents tens of millions of shillings in annual losses from unpaid loans.

**5-Fold Cross-Validation:**
Mean ROC-AUC across 5 stratified folds is consistent with the test set result, confirming the model is genuinely learning patterns that generalise to new applicants — not overfitting to one particular data split.

### Business Requirements Check

| Requirement | Target | Achieved | Status |
|---|---|---|---|
| Recall | ≥ 70% | 76.5% | PASS |
| Precision | ≥ 60% | 15.0% | BELOW TARGET* |
| ROC-AUC | ≥ 0.75 | 0.7609 | PASS |

*Precision of 15% is expected given the severe 8% default rate. In imbalanced classification, prioritising Recall necessarily reduces Precision. The trade-off is acceptable because the cost of missing a defaulter (full loan loss) far exceeds the cost of reviewing an additional flagged applicant.

### Confusion Matrix — Champion Model

| | Predicted Repaid | Predicted Default |
|---|---|---|
| **Actual Repaid** | True Negatives — correctly approved | False Positives — good borrowers flagged |
| **Actual Default** | False Negatives — missed defaults (most costly) | True Positives — defaults caught |

### Model Comparison Chart

![Model Comparison]
*Grouped bar chart comparing Recall, Precision, F1 Score, and ROC-AUC across all four models*

### SHAP Summary Plot

![SHAP Summary]
*SHAP dot plot showing global feature importance and direction — red dots push toward default, blue dots reduce default risk*

### Confusion Matrix

![Confusion Matrix]
*Confusion matrix for the XGBoost Balanced champion model on the held-out test set*

---

## SHAP Explainability

SHAP (SHapley Additive exPlanations) answers the question that regulators and applicants need answered: **why** did the model flag this specific person as high risk?

### Two Levels of Explanation

**Global explanation** — which features matter most across all predictions:

| Rank | Feature | Plain Language | Risk Direction |
|---|---|---|---|
| 1 | `EXT_SOURCE_2` | External credit score from bureau | Low score → increases default risk |
| 2 | `EXT_SOURCE_3` | Second external credit score | Low score → increases default risk |
| 3 | `EXT_SOURCE_1` | Third external credit score | Low score → increases default risk |
| 4 | Education Type | Applicant's highest education level | Lower education → increases default risk |
| 5 | Income / Employment Type | Employment stability | Unemployed or irregular income → increases default risk |

**Local explanation** — why one specific applicant was flagged:
The SHAP waterfall plot shows the model's step-by-step reasoning. Starting from the average portfolio default rate, each feature either pushes the prediction up (red, toward default) or down (blue, away from default). The result is an auditable, feature-level explanation that can be shown to the applicant, the loan officer, and the regulator.

### Key Threshold Discovered

SHAP dependence analysis revealed a non-linear threshold effect in `EXT_SOURCE_2`:

- Below 0.35: SHAP values are predominantly positive — high default risk zone
- Between 0.35 and 0.60: Standard risk range — normal underwriting
- Above 0.60: SHAP values plateau — consistently safe zone

Simple correlations and linear models miss this threshold effect entirely. XGBoost captures it; SHAP makes it visible and actionable.

### Validation

SHAP and Permutation Importance were compared as two independent feature importance methods. Strong rank correlation between both methods confirms the findings are robust and not an artefact of one particular technique.

---

## How to Reproduce

Follow these steps exactly. Every command is tested and ready to run.

**1. Clone the repository**

```bash
git clone https://github.com/KimutaiHub/loan-default-prediction.git
cd loan-default-prediction
```

**2. Create a virtual environment**

python -m venv venv
venv\Scripts\activate
```

**3. Install all dependencies**

```bash
pip install -r requirements.txt
```

**4. Download the dataset from Kaggle**

Go to: https://www.kaggle.com/competitions/home-credit-default-risk/data

You will need a free Kaggle account. Accept the competition rules, then download the following files:
- `application_train.csv`
- `bureau.csv`
- `previous_application.csv`
- `installments_payments.csv`
- `credit_card_balance.csv`
- `HomeCredit_columns_description.csv`

**5. Place all CSV files in the `data/` folder**

```
loan-default-prediction/
└── data/
    ├── application_train.csv
    ├── bureau.csv
    ├── previous_application.csv
    ├── installments_payments.csv
    ├── credit_card_balance.csv
    └── HomeCredit_columns_description.csv
```

**6. Open and run the notebook**

```bash
jupyter notebook loan_default_risk_prediction.ipynb
```

Run all cells from top to bottom using **Kernel → Restart & Run All**. Expected total runtime: 20–40 minutes depending on hardware, primarily due to XGBoost training and SHAP value computation.

> All random operations use `random_state=42` for full reproducibility. Results will match those reported in this README exactly.

**Optional — Run the Flask API**

```bash
python app.py
```

The API will start at `http://localhost:5000`. Send a POST request to `/predict` with applicant features as JSON to receive a default probability score and risk tier.

---

## Business Recommendations

The following recommendations come directly from SHAP analysis of the champion model. They are written for credit risk managers and lending executives — no data science background required.

- **Implement a three-tier credit score screening system.** External credit scores are the single most powerful predictor of default in this model. Applicants with very low credit scores should be flagged automatically for manual review and additional documentation. Those with strong credit scores should be eligible for fast-track approval. A tiered system reduces reviewer workload while concentrating scrutiny where risk is highest.

- **Cap monthly loan repayments at 28% of verified monthly income.** When a borrower's scheduled monthly repayment exceeds 30% of their income, default risk rises sharply. Setting a firm cap at 28% prevents the institution from approving loans that borrowers are structurally unable to service, regardless of how creditworthy they appear on other measures.

- **Apply additional scrutiny to applicants with a history of late payments.** Borrowers who paid more than 20% of their previous loan installments late are significantly more likely to default on new loans. Past payment behaviour is the most reliable indicator of future payment behaviour. This check should be a standard part of every credit assessment process.

- **Set conservative initial loan limits for first-time borrowers under 27 years of age.** Applicants in the 20–25 age group default at nearly double the overall average rate. Rather than excluding young borrowers, offer smaller initial loans that build credit history. Increase limits progressively as repayment is demonstrated.

- **Provide every applicant with a specific, data-driven explanation for their credit decision.** The model produces a feature-level explanation for every prediction. When an application is declined, the loan officer can tell the applicant exactly which factors drove the decision — credit score, debt burden, payment history — and what they can improve. This reduces disputes, builds trust, and meets the transparency requirements expected by Kenya's financial regulators.

---

## Technologies Used

**Data Processing**

| Library | Purpose |
|---|---|
| `pandas` | Data loading, merging, aggregation, and transformation |
| `numpy` | Numerical operations, array manipulation, and feature calculations |

**Machine Learning**

| Library | Purpose |
|---|---|
| `scikit-learn` | Preprocessing pipelines, Logistic Regression, Random Forest, train/test split, cross-validation, evaluation metrics |
| `xgboost` | Gradient boosting champion model with built-in class imbalance handling |

**Class Imbalance**

| Library | Purpose |
|---|---|
| `imbalanced-learn` | SMOTE for synthetic minority oversampling within training pipelines |

**Explainability**

| Library | Purpose |
|---|---|
| `shap` | SHAP values for global and local model interpretability |
| `scikit-learn` | Permutation importance for independent validation of SHAP findings |

**Visualisation**

| Library | Purpose |
|---|---|
| `matplotlib` | Base plotting for all charts |
| `seaborn` | Statistical visualisations — distributions, heatmaps, bar charts |

**Deployment**

| Library | Purpose |
|---|---|
| `flask` | REST API for serving real-time predictions |
| `joblib` | Saving and loading trained model pipelines to disk |

---

## Limitations

**Dataset origin:** The training data comes from a Central and Eastern European lending company. Kenyan borrowers differ in income structure, mobile money behaviour, informal economy participation, and credit bureau coverage. The model must be retrained on local Kenyan data before production deployment at M-KOPA, Tala, or any Kenyan institution.

**Precision trade-off:** Precision of 15% means that of every 100 applicants flagged as high risk, approximately 85 will actually repay. This is an expected consequence of the 8% default rate combined with prioritising Recall. In the lending context, the cost of missing a real defaulter exceeds the cost of reviewing an additional application — this trade-off is intentional and appropriate.

**Model drift:** Economic conditions, employment patterns, and borrower behaviour change over time. A model trained on historical data will degrade in performance as the world changes. The model should be retrained every 6 months and monitored monthly — if ROC-AUC drops below 0.70 on recent approvals, retraining should be triggered immediately.

**Feature dependency:** The three most important features — `EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3` — require integration with third-party credit bureaux. Lenders without access to bureau data will see materially lower model performance and should not deploy this model without a plan to source these features.

**Fairness:** The model identified gender and education level as predictive features. These may encode historical lending biases rather than true creditworthiness signals. A fairness audit examining approval rates and false negative rates across demographic groups is strongly recommended before any production deployment.

---

## Author

### Kimutai Kevine

Data Scientist and TVET Educator with 7+ years of professional experience in national data systems, curriculum development, and applied research — pursuing MSc Data Science, University of East London, UK.

- LinkedIn: [linkedin.com/in/kimutaikevine](https://linkedin.com/in/kimutaikevine)
- GitHub: [github.com/KimutaiHub](https://github.com/KimutaiHub)
- Email: kevinekimutai@gmail.com

This project was built to demonstrate applied machine learning in the context of financial services — where explainable, data-driven credit risk assessment is not just a technical achievement but a practical tool for promoting responsible lending and financial inclusion.


