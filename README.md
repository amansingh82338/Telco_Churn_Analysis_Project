# Telco Customer Churn Prediction — ML Classification Engine

> **Production-grade churn prediction pipeline** | Logistic Regression · Random Forest · Gradient Boosting | AUC: 0.8536 | 7,011 customer records

---

## Executive Summary

Telecom customer attrition is a silent revenue hemorrhage. Every churned subscriber directly erodes **Monthly Recurring Revenue (MRR)** while simultaneously inflating **Customer Acquisition Costs (CAC)** — the dual cost pressure that compresses operating margins and destabilizes long-range forecasting. Reactive retention strategies, deployed only after churn materializes, are both operationally inefficient and commercially damaging.

This machine learning engine delivers **early-warning churn signals** at the individual subscriber level, enabling proactive intervention before revenue is lost. Built on a rigorously cleaned dataset of **7,011 customer records** with a baseline churn rate of **26.6%**, the pipeline encompasses the complete Data Science lifecycle: programmatic SQL auditing, Python-based preprocessing, domain-driven feature engineering, stratified cross-validated model training, and deterministic pipeline serialization for downstream batch inference. The winning production model — **Logistic Regression** — achieves a **Test AUC of 0.8536** while preserving full coefficient interpretability, making it suitable for both automated scoring and analytical stakeholder communication.

---

## Repository File Structure

```
customer_churn_prediction/
├── Data/
│   ├── telco_raw_data.csv           # Source dataset: 7,033 raw customer records
│   └── telco_cleaned_data.csv       # Cleaned dataset: 7,011 records post-deduplication
├── Models/
│   ├── churn_model.pkl              # Serialized Logistic Regression classifier
│   ├── feature_columns.pkl          # Ordered feature column manifest for inference matching
│   ├── feature_importance.pkl       # Coefficient weight registry for interpretability audit
│   ├── label_encoders.pkl           # Fitted categorical encoders (LabelEncoder per feature)
│   ├── model_metrics.pkl            # Frozen evaluation metrics (AUC, F1, precision, recall)
│   └── scaler.pkl                   # Fitted StandardScaler for numerical normalization
├── Python/
│   ├── Python_churn_output.pdf      # Rendered execution output with inline diagnostics
│   ├── model_evaluation.png         # ROC curves, confusion matrix, and classification report
│   ├── eda_plots.png                # EDA distribution plots and churn-stratified visuals
│   └── telco_churn_python.py        # Master pipeline script (cleaning → training → serialization)
├── sql/
│   └── telco sql queries .pdf       # SQL audit log: data profiling and anomaly detection queries
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## Data Auditing & Cleaning Protocol

Raw ingestion from `telco_raw_data.csv` triggers a multi-stage programmatic cleaning protocol before any analytical or modelling operations are executed.

**Stage 1 — High-Cardinality Column Excision**
Variables `Zip_Code` and `City` are unconditionally dropped. These fields carry zero predictive signal for churn classification while introducing high-cardinality noise that degrades regularization efficiency in downstream models. Their removal also eliminates implicit geographic proxy leakage that could introduce unintended demographic bias into the scoring layer.

**Stage 2 — Duplicate Record Deduplication**
The raw frame of **7,033 rows** was subjected to full-row duplicate detection using a deterministic `df.drop_duplicates()` pass. **22 duplicate records** were identified and excised, yielding the canonical **7,011-record analytical dataset** used across all modelling stages.

**Stage 3 — Type Coercion & Numerical Anomaly Resolution**
The `Total_Charges` column was ingested as an object type due to whitespace-padded null representations (entries containing `" "`). These were coerced to `NaN` via `pd.to_numeric(..., errors='coerce')` and subsequently imputed using **median substitution** — a statistically robust imputation strategy that preserves distributional shape while respecting skewness bounds in a right-tailed billing variable. Mean imputation was explicitly ruled out due to sensitivity to high-value outlier accounts.

---

## Feature Engineering Canvas

Beyond raw telco attributes, five domain-derived features were constructed to surface latent behavioral and financial signals that the base schema does not expose directly.

**`Avg_Monthly_Spend`** — *Billing Efficiency Index*
Computed as `Total_Charges / Tenure_Months`. This ratio normalizes cumulative billing against customer longevity, isolating whether a customer's effective monthly burden has escalated, stabilized, or declined relative to their cohort. Elevated values relative to tenure signal billing optimization friction — a structural churn precursor.

**`Addon_Count`** — *Product Stickiness Quantifier*
A summation of active auxiliary services (Online Security, Online Backup, Device Protection, Tech Support, Streaming TV, Streaming Movies). A higher addon count implies deeper platform integration and cross-sell penetration, which empirically correlates with reduced churn probability. Low counts expose customers to single-service dependency risk.

**`Is_New_Customer`** — *Early-Stage Attrition Flag*
Binary indicator activated when `Tenure_Months ≤ 6`. New subscribers exhibit disproportionately high churn volatility during the onboarding window, before habituating to the service ecosystem. This flag enables the model to apply elevated attrition sensitivity to the cohort most at risk of dropout before the first loyalty threshold.

**`High_Bill`** — *Billing Stress Sentinel*
A threshold-based binary flag isolating customers whose `Monthly_Charges` exceed the dataset's 75th percentile. High billing burden, particularly when uncorrelated with perceived service value, is a primary churn accelerant.

**`No_Addons`** — *Minimal Engagement Detector*
A binary flag set when `Addon_Count == 0`. Customers with zero auxiliary services represent the lowest-engagement, highest-disengagement-risk cohort. The interaction between `High_Bill` and `No_Addons` is particularly diagnostic: subscribers paying premium rates without consuming supplementary services exhibit the behavioral profile most predictive of voluntary cancellation.

---

## Model Training & Cross-Validation Matrix

All models were trained on an 80/20 **stratified train-test split**, preserving the 26.6% baseline churn ratio across both partitions. Model selection was governed by **5-Fold Stratified Cross-Validation** on the training partition, with held-out Test AUC serving as the independent generalization benchmark. No hyperparameter leakage from the test fold was permitted into the cross-validation selection process.

| Model | CV Mean AUC (5-Fold) | Test AUC | Selection Status |
|---|---|---|---|
| **Logistic Regression** | **0.8600** | **0.8536** | ✅ **Production Model** |
| Gradient Boosting | 0.8622 | 0.8503 | Evaluated — Not Selected |
| Random Forest | 0.8410 | 0.8299 | Evaluated — Not Selected |

**Engineering Rationale — Why Logistic Regression?**

Gradient Boosting achieves a marginally superior CV Mean AUC (0.8622 vs. 0.8600), but its Test AUC (0.8503) falls **0.0033 points below** Logistic Regression's (0.8536), indicating a slight tendency toward overfitting the training distribution. Random Forest underperforms on both metrics across the stratified splitting matrices.

Beyond raw discriminative power, Logistic Regression provides three production-critical advantages that ensemble methods cannot match:

1. **Coefficient Interpretability** — Absolute logit weights produce directly auditable feature importance rankings, enabling compliance-grade explanations for retention decisions without post-hoc approximation methods (e.g., SHAP/LIME).
2. **Inference Latency** — Linear scoring operations execute at sub-millisecond throughput for batch pipelines, with no ensemble aggregation overhead.
3. **Overfitting Robustness** — L2 regularization (Ridge penalty) constrains coefficient magnitudes under high-collinearity feature sets, ensuring stable AUC performance across unseen population samples.

**Final Production Metrics (Test Set — Logistic Regression)**

| Metric | Score |
|---|---|
| Test AUC | 0.8536 |
| Overall Accuracy | 81% |
| Precision (Churn Class) | 67% |
| Recall (Churn Class) | 55% |
| F1-Score (Churn Class) | 60% |

---

## Operational Feature Weights & Diagnostics

Diagnostic visualizations are stored in the `Python/` directory:

- **`Python/model_evaluation.png`** — ROC curve with AUC annotation, confusion matrix heatmap, and per-class classification report rendering.
- **`Python/eda_plots.png`** — Churn-stratified distribution plots across contract type, payment method, internet service, and demographic variables.

**Logistic Regression Coefficient Analysis**

The model's classification boundary is governed by signed logit weights. The following directional patterns were extracted from the serialized `feature_importance.pkl` registry:

*Risk-Suppressing Coefficients (Negative Logit Weight — Churn Probability ↓)*
- **Two-Year Contract** — The single most powerful churn-suppressing signal. Long-term contractual commitment dramatically anchors retention, producing a 2.8% churn rate vs. 42.7% for month-to-month subscribers. Negative coefficient magnitude is dominant.
- **One-Year Contract** — Intermediate suppression effect, providing structural retention buffer between month-to-month volatility and two-year commitment security.
- **Higher Addon Count** — Product depth creates switching cost inertia. Each incremental service layer contributes a compounding negative coefficient contribution.
- **Extended Tenure** — Long-tenure customers encode accumulated lifecycle value and inertia; the model learns that historical persistence is strongly predictive of continued persistence.

*Risk-Amplifying Coefficients (Positive Logit Weight — Churn Probability ↑)*
- **Month-to-Month Contract** — The dominant positive coefficient. Zero switching friction and no contractual exit barrier make this the highest-risk contract class (42.7% churn rate).
- **Fiber Optic Internet Service** — Significant positive coefficient contribution. 41.9% churn rate vs. 19.0% for DSL subscribers signals either technical service quality deficits or pricing-to-value misalignment in the Fiber tier.
- **Electronic Check Payment** — 45.3% churn rate, the highest of any payment modality. May reflect lower automated commitment (vs. bank transfer or credit card auto-pay), higher payment friction, or demographic correlation with lower service engagement.
- **Senior Citizen Flag** — 41.7% churn rate among senior cohorts, with positive coefficient contribution, indicating elevated attrition volatility in this demographic segment.
- **`Is_New_Customer` Flag** — Encodes early-stage dropout risk within the 0–6 month onboarding window.
- **`High_Bill` + `No_Addons` Interaction** — Customers bearing peak billing loads with zero product engagement represent the highest-acuity at-risk cohort in the scoring distribution.

---

## Key Exploratory Findings

Pre-modelling EDA surfaced four structurally significant churn risk concentrations:

| Risk Dimension | High-Risk Segment | Churn Rate | Low-Risk Benchmark | Benchmark Rate |
|---|---|---|---|---|
| Contract Type | Month-to-Month | 42.7% | Two-Year Agreement | 2.8% |
| Payment Method | Electronic Check | 45.3% | Bank Transfer / Credit Card | ~15–18% |
| Internet Infrastructure | Fiber Optic | 41.9% | DSL | 19.0% |
| Demographics | Senior Citizens | 41.7% | Non-Senior Cohort | ~23% |

These concentrations served as the primary domain-knowledge inputs into the feature engineering stage and validate the model's learned coefficient structure.

---

## Pipeline Serialization & Production Deployment

All transformers, column schemas, and the predictive binary are frozen into deterministic serialization artifacts using `joblib`, located in the `Models/` directory. Serialization guarantees **inference deterministic matching** between training-time preprocessing and batch scoring environments — eliminating transformation drift that would otherwise corrupt predictions when the pipeline is redeployed.

| Artifact | Object Type | Function |
|---|---|---|
| `churn_model.pkl` | `LogisticRegression` | Core predictive binary; outputs churn probability scores and class labels |
| `scaler.pkl` | `StandardScaler` | Fitted normalization transformer; must be applied to all numerical features prior to inference |
| `label_encoders.pkl` | `Dict[str, LabelEncoder]` | Per-column categorical encoding registry; guarantees encoding parity with training-time encoding |
| `feature_columns.pkl` | `List[str]` | Ordered column manifest; enforces strict feature alignment during batch inference to prevent column-ordering errors |
| `feature_importance.pkl` | `Dict[str, float]` | Coefficient weight map; used for post-inference explanation and audit trail generation |
| `model_metrics.pkl` | `Dict[str, float]` | Frozen evaluation registry; enables automated metric drift detection in production monitoring layers |

**Inference Pipeline — Execution Order**

At batch scoring time, the preprocessing sequence must match the training-time transformation order exactly. The deterministic execution chain is:

1. Load raw feature frame → apply `label_encoders.pkl` for categorical columns
2. Apply `feature_columns.pkl` to enforce column selection and ordering
3. Apply `scaler.pkl` to normalize numerical features
4. Pass transformed frame into `churn_model.pkl` to generate probability scores
5. Apply business-defined threshold (default: 0.5) to produce binary churn labels

Deviation from this sequence — including column reordering or encoder substitution — will produce undefined inference behavior. Imputation bounds for `Total_Charges` must be reapplied at inference time if missing values are possible in the scoring population.

---

## Exploratory Analysis & SQL Audit Layer

Prior to Python processing, the raw dataset was subjected to systematic SQL-based profiling to establish data quality baselines. The audit output is documented in `sql/telco sql queries .pdf` and covers:

- Null count profiling across all 21 raw columns
- Distribution analysis for high-churn-correlated categorical variables
- Duplicate detection via grouped row-hash comparison
- Outlier flagging in `Monthly_Charges` and `Total_Charges` via IQR boundary analysis
- Churn rate segmentation by contract type, payment method, and internet service

SQL auditing precedes Python processing in the lifecycle to establish a transparent, reproducible data quality record that is independent of the modelling codebase — a production-grade separation of concerns between data validation and analytical transformation.

---

## Installation & Reusability

**Clone the Repository**

```bash
git clone https://github.com/<your-username>/customer_churn_prediction.git
cd customer_churn_prediction
```

**Install Dependencies**

```bash
pip install -r requirements.txt
```

**Package Requirements (`requirements.txt`)**

```
pandas
numpy
scikit-learn
plotly
joblib
```

**Execute the Pipeline**

```bash
python Python/telco_churn_python.py
```

Running this script executes the complete pipeline end-to-end: data ingestion from `Data/telco_cleaned_data.csv`, feature engineering, stratified train-test split, 5-fold cross-validated model training across all three candidate classifiers, evaluation metric computation, diagnostic plot generation (`model_evaluation.png`, `eda_plots.png`), and serialization of all production artifacts to the `Models/` directory.

**Re-run Requirements**
- Python ≥ 3.8
- Input data file present at `Data/telco_cleaned_data.csv`
- Write access to `Models/` and `Python/` output directories

---

## License

This project is licensed under the terms specified in `LICENSE`. Refer to the license file for permitted use, redistribution, and attribution requirements.

---

*Built for portfolio demonstration of production-grade Data Science workflow design, statistical validation rigor, and end-to-end ML pipeline engineering.*
