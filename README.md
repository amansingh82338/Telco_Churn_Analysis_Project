# 📡 Telco Customer Churn Prediction System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-ML%20Engine-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/SQL-Data%20Layer-4479A1?style=for-the-badge&logo=mysql&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Status-Production%20Ready-brightgreen?style=for-the-badge"/>
</p>

<p align="center">
  An end-to-end machine learning system that predicts telecom customer churn using structured behavioral and subscription data — from raw SQL queries through model training to an interactive Streamlit dashboard.
</p>

---

## 🔍 Business Problem

Customer churn is one of the most expensive challenges facing telecom providers. Acquiring a new customer costs 5–7× more than retaining an existing one, yet most companies only act after a customer has already left.

This project transforms raw customer data from 7,032 IBM Telco subscribers into a production-grade churn prediction engine. It identifies high-risk customers **before** they churn, quantifies revenue at risk, and surfaces actionable retention signals — all within an interactive dashboard built for both technical and business stakeholders.

**Key findings surfaced by this system:**
- Month-to-month customers churn at **42.7%** vs. 2.8% for two-year contract holders
- Customers with no security or support add-ons churn at nearly **49%**
- Over **$204,000/month** in revenue is at risk from high-value month-to-month churners alone
- Los Angeles, San Diego, and Modesto rank among the highest churn-risk cities

---

## ✨ Features

| Feature | Description |
|---|---|
| 📊 **Interactive Dashboard** | Real-time KPI cards, churn distribution charts, and trend visualizations powered by Plotly |
| 🔍 **Customer Explorer** | Filter, search, and drill into individual customer profiles with churn risk scores |
| 🎯 **Live Churn Prediction** | Enter customer attributes and receive an instant churn probability with risk classification |
| 💡 **Business Insights** | Revenue-at-risk analysis, CLV segmentation, geographic churn maps, and contract-type breakdowns |
| 📈 **Model Performance** | ROC curves, confusion matrices, cross-validation scores, and feature importance rankings |
| 🗄️ **SQL Integration** | Five production-grade SQL queries for churn analytics, CLV segmentation, and geographic risk mapping |
| 🤖 **Machine Learning** | Three-model comparison pipeline with stratified cross-validation and ROC-AUC model selection |
| 🖥️ **Streamlit UI** | Multi-page responsive application deployable locally or on Streamlit Cloud |
| 📥 **Downloadable Reports** | Export filtered customer data and prediction results as CSV files |

---

## 🛠️ Technologies Used

| Category | Tools |
|---|---|
| **Language** | Python 3.10+ |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-Learn |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **Web Application** | Streamlit |
| **Database** | SQL (MySQL / BigQuery compatible) |
| **Model Serialization** | Joblib (.pkl) |
| **Version Control** | Git, GitHub |

---

## 🔄 Project Workflow

```
Raw Data (IBM Telco Dataset — 7,032 records × 33 features)
         │
         ▼
  Exploratory Data Analysis
  (Churn distribution, tenure analysis, contract breakdown,
   correlation heatmap, service-level churn rates)
         │
         ▼
     Data Cleaning
  (Type coercion, duplicate removal, median imputation,
   column pruning — Zip_Code, City dropped)
         │
         ▼
   Feature Engineering
  (Avg_Monthly_Spend, Addon_Count, Is_New_Customer,
   High_Bill flag, No_Addons flag)
         │
         ▼
  SQL Analytics Layer
  (Churn by contract, CLV segmentation, revenue at risk,
   geographic risk mapping, add-on impact analysis)
         │
         ▼
Preprocessing & Encoding
  (Label Encoding for binary columns,
   One-Hot Encoding for multi-category columns,
   StandardScaler on numerical features)
         │
         ▼
   Model Training & Evaluation
  (Logistic Regression · Random Forest · Gradient Boosting
   5-Fold Stratified Cross-Validation · ROC-AUC scoring)
         │
         ▼
   Model Serialization (.pkl)
  (churn_model · scaler · label_encoders ·
   feature_columns · model_metrics · feature_importance)
         │
         ▼
  Streamlit Dashboard
  (Dashboard · Customer Explorer · Predict Churn ·
   Business Insights · Model Performance · About)
         │
         ▼
    Business Insights
  (Actionable retention recommendations for operations teams)
```

---

## 📁 Folder Structure

```
telco-churn-prediction/
│
├── data/
│   ├── telco_raw_data.csv            # Original IBM Telco dataset (7,032 records)
│   └── telco_cleaned_data.csv        # Cleaned & preprocessed dataset
│
├── model/
│   ├── churn_model.pkl               # Best trained ML model
│   ├── scaler.pkl                    # Fitted StandardScaler
│   ├── label_encoders.pkl            # Binary column encoders
│   ├── feature_columns.pkl           # Ordered feature list for inference
│   ├── model_metrics.pkl             # Accuracy, AUC, F1, Precision, Recall
│   ├── feature_importance.pkl        # Feature importance scores
│   └── sample_input.pkl              # Sample preprocessed row for testing
│
├── notebooks/
│   └── churn_eda.ipynb               # Exploratory analysis notebook
│
├── sql/
│   └── telco_sql_queries.sql         # 5 production SQL analytics queries
│
├── assets/
│   ├── eda_plots.png                 # EDA visualization exports
│   └── model_evaluation.png         # ROC, confusion matrix, feature importance
│
├── telco_churn_python.py             # Full ML pipeline (EDA → train → serialize)
├── app.py                            # Streamlit multi-page application entry point
├── requirements.txt                  # Python dependencies
└── README.md
```

---

## 🤖 Machine Learning Models

Three classifiers were trained and compared using 5-fold stratified cross-validation. The best model was selected based on **ROC-AUC score** on the held-out test set (80/20 stratified split).

| Model | CV AUC | Description |
|---|---|---|
| **Logistic Regression** | Baseline | Linear decision boundary with L2 regularization. Fast, interpretable, strong baseline for binary classification. |
| **Random Forest** | Competitive | Ensemble of 100 decision trees with bootstrap aggregation. Handles non-linearity and provides native feature importance scores. |
| **Gradient Boosting** | Best | Sequential boosting ensemble (100 estimators). Minimizes classification error iteratively — strongest on tabular churn data. |

> **Model selection criterion:** `roc_auc_score` on the test set. The winning model and all preprocessing artifacts are serialized via Joblib for zero-overhead loading in the Streamlit app.

**Engineered features fed to all models:**

| Feature | Description |
|---|---|
| `Avg_Monthly_Spend` | Total Charges ÷ Tenure Months — spend efficiency proxy |
| `Addon_Count` | Number of active value-add services (0–6) |
| `Is_New_Customer` | Binary flag: Tenure ≤ 6 months |
| `High_Bill` | Binary flag: Monthly Charges above dataset median |
| `No_Addons` | Binary flag: Zero active add-ons (highest risk segment) |

---

## 🖥️ Dashboard Features

<details>
<summary><strong>📊 Dashboard (Home)</strong></summary>

The landing page displays real-time business KPIs: total customers, overall churn rate, average monthly revenue, and average customer tenure. Plotly charts show churn distribution, monthly charge comparisons between churned and retained customers, and churn rate breakdowns by contract type and internet service tier.

</details>

<details>
<summary><strong>🔍 Customer Explorer</strong></summary>

An interactive data table with multi-column filtering — filter by contract type, internet service, senior citizen status, and churn label. Each row links to a full customer profile showing service subscriptions, payment details, and model-assigned churn risk score. Supports CSV export of filtered results.

</details>

<details>
<summary><strong>🎯 Predict Churn</strong></summary>

A form-based inference page where users input any customer's attributes (demographics, services, contract type, billing details) and receive an instant churn probability from the serialized model. Output includes a risk tier (Low / Medium / High), probability gauge, and recommended retention actions.

</details>

<details>
<summary><strong>💡 Business Insights</strong></summary>

Five SQL-backed analytical views:
- **Churn by Contract Type** — Month-to-month: 42.71% · One-year: 11.27% · Two-year: 2.83%
- **CLV Segmentation** — High / Mid / Low value tiers with churn counts and avg. monthly charges
- **Revenue at Risk** — Monthly revenue exposed per contract segment from high-value churners
- **Geographic Risk Map** — Top 15 cities ranked by churn rate with lost revenue estimates
- **Add-on Impact** — Churn rate matrix across Online Security × Tech Support combinations

</details>

<details>
<summary><strong>📈 Model Performance</strong></summary>

Side-by-side ROC curves for all three models, confusion matrix for the best model, classification report (Precision / Recall / F1 by class), cross-validation AUC scores, and a horizontal bar chart of the top 15 most predictive features ranked by importance score.

</details>

<details>
<summary><strong>ℹ️ About</strong></summary>

Project overview, dataset source attribution, model architecture summary, and links to the author's GitHub and LinkedIn profiles.

</details>

---

## 🚀 Installation Guide

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/telco-churn-prediction.git
cd telco-churn-prediction
```

### 2. Create a Virtual Environment
```bash
python -m venv venv

# Activate — Windows
venv\Scripts\activate

# Activate — macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the ML Pipeline (generates .pkl files)
```bash
python telco_churn_python.py
```

### 5. Launch the Streamlit App
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 📸 Screenshots

| Page | Preview |
|---|---|
| **Dashboard** | ![Dashboard Screenshot](assets/screenshots/dashboard.png) |
| **Predict Churn** | ![Prediction Page Screenshot](assets/screenshots/predict_churn.png) |
| **Business Insights** | ![Business Insights Screenshot](assets/screenshots/business_insights.png) |
| **Model Performance** | ![Model Performance Screenshot](assets/screenshots/model_performance.png) |

> *Add screenshots to `assets/screenshots/` after running the app locally.*

---

## 💼 Business Impact

This system gives telecom retention teams a data-driven advantage across four dimensions:

**Reduce Churn**
Proactively identify customers with high churn probability before they cancel. The model flags at-risk customers days or weeks ahead, giving retention teams time to intervene.

**Improve Customer Retention**
Targeted outreach — upgrade offers, loyalty discounts, or proactive tech support — can be prioritized to the customers the model ranks highest risk, maximizing retention ROI.

**Target High-Risk Segments**
SQL analytics reveal that customers on month-to-month contracts with no security or support add-ons churn at nearly 49%. These micro-segments can receive automated campaign triggers based on model score thresholds.

**Protect Revenue**
Over $204,000 in monthly recurring revenue is at risk from high-value month-to-month churners alone. The Business Insights module quantifies this exposure by contract segment, giving leadership a clear financial case for retention investment.

---

## 🔮 Future Improvements

- [ ] **SHAP Explainability** — Add SHAP value waterfall plots to explain individual predictions at the customer level
- [ ] **Live Database Integration** — Replace flat-file input with a PostgreSQL or BigQuery connection for real-time scoring
- [ ] **Cloud Deployment** — Containerize and deploy to AWS / GCP / Azure with CI/CD via GitHub Actions
- [ ] **Real-Time Prediction API** — Expose the serialized model via a FastAPI endpoint for integration with CRM systems
- [ ] **Authentication Layer** — Add role-based access control so business and technical users see appropriate views
- [ ] **Docker Support** — Publish a `Dockerfile` and `docker-compose.yml` for one-command reproducible deployment
- [ ] **Automated Retraining** — Trigger model retraining on a schedule or when data drift is detected via Evidently AI

---

## 👤 Author

| | |
|---|---|
| **Name** | Aman Singh Chauhan |
| **LinkedIn** | [linkedin.com/in/your-profile](https://linkedin.com/in/your-profile) |
| **GitHub** | [github.com/your-username](https://github.com/your-username) |
| **Email** | your.email@example.com |

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- **[IBM Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)** — The foundational dataset powering this project
- **[Scikit-Learn](https://scikit-learn.org/)** — Machine learning pipeline, cross-validation, and model evaluation utilities
- **[Streamlit](https://streamlit.io/)** — Rapid, Pythonic web application framework for data science
- **Open Source Community** — Pandas, NumPy, Plotly, Seaborn, Joblib, and every library that made this possible

---

<p align="center">
  Built with ❤️ by <strong>Aman Singh Chauhan</strong> · If this project helped you, give it a ⭐
</p>
