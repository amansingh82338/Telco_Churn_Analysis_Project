# =============================================================================
#  TELCO CUSTOMER CHURN PREDICTION — FULL PRODUCTION ML PIPELINE
#  Optimized for Streamlit Deployment
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
import logging
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score, f1_score
)

warnings.filterwarnings('ignore')

# =============================================================================
# GLOBAL CONFIGURATIONS & PATHS
# =============================================================================
DATA_PATH = Path(r"C:\Users\amans\Downloads\Codtech Projects\customer churn prediction\telco_cleaned_data.csv")
MODEL_DIR = Path("model")

# Ensure the model directory exists
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Set up basic logging for error handling
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_data(file_path: Path) -> pd.DataFrame:
    """Loads the dataset from the specified path."""
    try:
        df = pd.read_csv(file_path)
        print("=" * 65)
        print("  TELCO CUSTOMER CHURN PREDICTION PIPELINE")
        print("=" * 65)
        print(f"\n✅ Data loaded: {df.shape[0]:,} rows × {df.shape[1]} columns\n")
        return df
    except FileNotFoundError:
        logging.error(f"File not found at {file_path}. Please check the path.")
        raise
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise


def perform_eda(df: pd.DataFrame) -> None:
    """Performs Exploratory Data Analysis and generates plots."""
    print("=" * 65)
    print("  SECTION 1 — EXPLORATORY DATA ANALYSIS")
    print("=" * 65)

    # --- 1.1 Basic Overview ---
    print(f"\n📋 Dataset Info:")
    print(f"   Rows      : {df.shape[0]:,}")
    print(f"   Columns   : {df.shape[1]}")
    print(f"   Duplicates: {df.duplicated().sum()}")
    print(f"\n📊 Column Types:")
    print(f"   Numerical : {df.select_dtypes(include='number').columns.tolist()}")
    print(f"   Categorical: {df.select_dtypes(include='object').columns.tolist()}")

    # --- 1.2 Target Variable Distribution ---
    churn_counts = df['Churn_Label'].value_counts()
    churn_pct    = df['Churn_Label'].value_counts(normalize=True) * 100
    print(f"\n🎯 Target Variable — Churn_Label:")
    for label in churn_counts.index:
        print(f"   {label}: {churn_counts[label]:,}  ({churn_pct[label]:.1f}%)")

    # --- 1.3 Missing Values ---
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    print(f"\n⚠️  Missing Values:")
    if missing.empty:
        print("   None — dataset is clean ✅")
    else:
        for col, cnt in missing.items():
            print(f"   {col}: {cnt} ({cnt/len(df)*100:.1f}%)")

    # --- 1.4 Numerical Summary ---
    # Fix Total_Charges to numeric for EDA purposes
    df['Total_Charges'] = pd.to_numeric(df['Total_Charges'], errors='coerce')
    num_cols = ['Tenure_Months', 'Monthly_Charges', 'Total_Charges']
    print(f"\n📈 Numerical Summary:")
    print(df[num_cols].describe().round(2).to_string())

    # --- 1.5 Churn Rate by Key Categories ---
    df['Churn_Value'] = (df['Churn_Label'] == 'Yes').astype(int)
    print(f"\n📊 Churn Rate by Key Categories:")
    cat_analysis = {
        'Contract'        : 'Contract',
        'Internet_Service': 'Internet Service',
        'Payment_Method'  : 'Payment Method',
        'Senior_Citizen'  : 'Senior Citizen'
    }
    for col, label in cat_analysis.items():
        grp = df.groupby(col)['Churn_Value'].mean() * 100
        print(f"\n   {label}:")
        for k, v in grp.items():
            print(f"      {str(k):<40} → {v:.1f}% churn")

    # --- 1.6 EDA Plots ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('EDA — Telco Customer Churn', fontsize=15, fontweight='bold')

    # Churn distribution
    colors = ['#2ecc71', '#e74c3c']
    churn_counts.plot(kind='bar', ax=axes[0, 0], color=colors, edgecolor='black')
    axes[0, 0].set_title('Churn Distribution')
    axes[0, 0].set_xlabel('Churn Label')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].tick_params(axis='x', rotation=0)
    for i, v in enumerate(churn_counts):
        axes[0, 0].text(i, v + 30, str(v), ha='center', fontweight='bold')

    # Tenure by churn
    for label, color in zip(['No', 'Yes'], colors):
        df[df['Churn_Label'] == label]['Tenure_Months'].plot(
            kind='hist', bins=30, alpha=0.6, ax=axes[0, 1], color=color, edgecolor='black'
        )
    axes[0, 1].set_title('Tenure Months by Churn')
    axes[0, 1].set_xlabel('Tenure Months')
    axes[0, 1].legend(['No Churn', 'Churn'])

    # Monthly Charges by churn
    df.boxplot(column='Monthly_Charges', by='Churn_Label', ax=axes[0, 2])
    axes[0, 2].set_title('Monthly Charges by Churn')
    axes[0, 2].set_xlabel('Churn Label')
    plt.sca(axes[0, 2])
    plt.title('Monthly Charges by Churn')
    plt.suptitle('')

    # Churn rate by Contract
    contract_churn = df.groupby('Contract')['Churn_Value'].mean() * 100
    contract_churn.plot(kind='bar', ax=axes[1, 0], color='#3498db', edgecolor='black')
    axes[1, 0].set_title('Churn Rate by Contract Type')
    axes[1, 0].set_ylabel('Churn Rate (%)')
    axes[1, 0].tick_params(axis='x', rotation=15)
    for i, v in enumerate(contract_churn):
        axes[1, 0].text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=9)

    # Churn rate by Internet Service
    internet_churn = df.groupby('Internet_Service')['Churn_Value'].mean() * 100
    internet_churn.plot(kind='bar', ax=axes[1, 1], color='#9b59b6', edgecolor='black')
    axes[1, 1].set_title('Churn Rate by Internet Service')
    axes[1, 1].set_ylabel('Churn Rate (%)')
    axes[1, 1].tick_params(axis='x', rotation=15)
    for i, v in enumerate(internet_churn):
        axes[1, 1].text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=9)

    # Correlation heatmap
    num_df = df[['Tenure_Months', 'Monthly_Charges', 'Total_Charges', 'Churn_Value']].dropna()
    sns.heatmap(num_df.corr().round(2), annot=True, cmap='coolwarm',
                ax=axes[1, 2], fmt='.2f', linewidths=0.5)
    axes[1, 2].set_title('Correlation Heatmap')

    plt.tight_layout()
    plt.savefig('eda_plots.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n✅ EDA plots saved → eda_plots.png")
    
    # Cleanup temporary Churn_Value column used for EDA
    df.drop(columns=['Churn_Value'], inplace=True, errors='ignore')


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values, types, and removes unused columns."""
    print("\n" + "=" * 65)
    print("  SECTION 2 — DATA CLEANING")
    print("=" * 65)

    df_clean = df.copy()
    df_clean.drop(columns=['Churn_Value'], inplace=True, errors='ignore')

    # --- 2.1 Drop low-value columns ---
    drop_cols = ['Zip_Code', 'City']
    df_clean.drop(columns=drop_cols, inplace=True, errors='ignore')
    print(f"\n🗑️  Dropped columns: {drop_cols}")
    print(f"   Reason: High cardinality / not predictive at model level")

    # --- 2.2 Fix Total_Charges dtype ---
    df_clean['Total_Charges'] = pd.to_numeric(df_clean['Total_Charges'], errors='coerce')
    filled = df_clean['Total_Charges'].isnull().sum()
    if filled > 0:
        df_clean['Total_Charges'].fillna(df_clean['Total_Charges'].median(), inplace=True)
        print(f"🔧 Total_Charges: coerced to numeric, {filled} NaN → filled with median")
    else:
        print(f"✅ Total_Charges: clean, no issues")

    # --- 2.3 Check & remove duplicates ---
    dupes = df_clean.duplicated().sum()
    df_clean.drop_duplicates(inplace=True)
    print(f"🧹 Duplicates removed: {dupes}")

    # --- 2.4 Verify no nulls remain ---
    remaining_nulls = df_clean.isnull().sum().sum()
    print(f"✅ Null values remaining: {remaining_nulls}")
    print(f"\n✅ Clean dataset: {df_clean.shape[0]:,} rows × {df_clean.shape[1]} columns")
    print(f"   Columns: {df_clean.columns.tolist()}")
    
    return df_clean


def engineer_features(df_clean: pd.DataFrame) -> pd.DataFrame:
    """Creates new domain-specific features based on existing data."""
    print("\n" + "=" * 65)
    print("  SECTION 3 — FEATURE ENGINEERING & PREPROCESSING")
    print("=" * 65)

    df_feat = df_clean.copy()

    # --- 3.1 New Engineered Features ---
    # Average monthly spend efficiency
    df_feat['Avg_Monthly_Spend'] = np.where(
        df_feat['Tenure_Months'] > 0,
        df_feat['Total_Charges'] / df_feat['Tenure_Months'],
        df_feat['Monthly_Charges']
    )

    # Add-on stickiness score
    addon_cols = ['Online_Security', 'Online_Backup', 'Device_Protection',
                  'Tech_Support', 'Streaming_TV', 'Streaming_Movies']
    df_feat['Addon_Count'] = df_feat[addon_cols].apply(
        lambda row: sum(1 for v in row if v == 'Yes'), axis=1
    )

    # New customer flag (tenure ≤ 6 months)
    df_feat['Is_New_Customer'] = (df_feat['Tenure_Months'] <= 6).astype(int)

    # High bill flag (above median)
    df_feat['High_Bill'] = (
        df_feat['Monthly_Charges'] > df_feat['Monthly_Charges'].median()
    ).astype(int)

    # No add-ons at all (most at-risk)
    df_feat['No_Addons'] = (df_feat['Addon_Count'] == 0).astype(int)

    print("\n🔧 Engineered Features:")
    print("   ✅ Avg_Monthly_Spend  — Total Charges / Tenure")
    print("   ✅ Addon_Count        — Count of active add-on services")
    print("   ✅ Is_New_Customer    — 1 if Tenure ≤ 6 months")
    print("   ✅ High_Bill          — 1 if Monthly Charges > median")
    print("   ✅ No_Addons          — 1 if zero add-on services")

    # --- 3.2 Encode Target ---
    df_feat['Churn'] = (df_feat['Churn_Label'] == 'Yes').astype(int)
    df_feat.drop(columns=['Churn_Label'], inplace=True)
    
    return df_feat


def preprocess_data(df_feat: pd.DataFrame):
    """Encodes categoricals and scales numerical features. Extracts and saves transformers."""
    # --- 3.3 Binary Encode (Yes/No and two-category columns) ---
    binary_cols = ['Gender', 'Senior_Citizen', 'Partner', 'Dependents',
                   'Phone_Service', 'Paperless_Billing']
    
    label_encoders_dict = {}
    for col in binary_cols:
        le = LabelEncoder()
        df_feat[col] = le.fit_transform(df_feat[col].astype(str))
        label_encoders_dict[col] = le
        
    print(f"\n🔢 Label Encoded (binary): {binary_cols}")

    # --- 3.4 One-Hot Encode multi-category columns ---
    ohe_cols = ['Multiple_Lines', 'Internet_Service', 'Online_Security',
                'Online_Backup', 'Device_Protection', 'Tech_Support',
                'Streaming_TV', 'Streaming_Movies', 'Contract', 'Payment_Method']
    
    df_feat = pd.get_dummies(df_feat, columns=ohe_cols, drop_first=True)
    print(f"🔢 One-Hot Encoded: {ohe_cols}")
    print(f"\n📐 Dataset shape after encoding: {df_feat.shape}")

    # Extract target and features before scaling to maintain column integrity
    X = df_feat.drop(columns=['Churn'])
    y = df_feat['Churn']
    
    feature_columns = X.columns.tolist()

    # --- 3.5 Scale Numerical Features ---
    scale_cols = ['Tenure_Months', 'Monthly_Charges', 'Total_Charges', 'Avg_Monthly_Spend', 'Addon_Count']
    scaler = StandardScaler()
    X[scale_cols] = scaler.fit_transform(X[scale_cols])
    print(f"📏 StandardScaler applied to: {scale_cols}")

    return X, y, scaler, label_encoders_dict, feature_columns


def train_and_evaluate(X: pd.DataFrame, y: pd.Series):
    """Splits data, trains models, evaluates them, and extracts key metrics."""
    print("\n" + "=" * 65)
    print("  SECTION 4 — SPLITTING THE DATA")
    print("=" * 65)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y       # preserve churn ratio in both splits
    )

    print(f"\n📦 Split: 80% Train / 20% Test  (stratified)")
    print(f"   Training set : {X_train.shape[0]:,} rows  |  Churn rate: {y_train.mean()*100:.1f}%")
    print(f"   Test set     : {X_test.shape[0]:,}  rows  |  Churn rate: {y_test.mean()*100:.1f}%")
    print(f"   Features     : {X_train.shape[1]}")

    print("\n" + "=" * 65)
    print("  SECTION 5 — MODEL TRAINING & EVALUATION")
    print("=" * 65)

    models = {
        "Logistic Regression" : LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest"       : RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting"   : GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    results = {}
    cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # --- 5.1 Cross-Validation ---
    print(f"\n📊 5-Fold Stratified Cross-Validation (AUC):")
    print(f"{'Model':<25} {'Mean AUC':>10} {'Std AUC':>10}")
    print("-" * 48)
    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
        model.fit(X_train, y_train)
        y_pred   = model.predict(X_test)
        y_prob   = model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, y_prob)
        results[name] = {
            'model'   : model,
            'y_pred'  : y_pred,
            'y_prob'  : y_prob,
            'cv_mean' : cv_scores.mean(),
            'cv_std'  : cv_scores.std(),
            'test_auc': test_auc
        }
        print(f"{name:<25} {cv_scores.mean():>10.4f} {cv_scores.std():>10.4f}")

    # --- 5.2 Classification Reports ---
    print("\n" + "-" * 65)
    print("📋 Test Set — Classification Reports:")
    print("-" * 65)
    for name, res in results.items():
        print(f"\n🔷 {name}  (Test AUC: {res['test_auc']:.4f})")
        print(classification_report(
            y_test, res['y_pred'],
            target_names=['No Churn', 'Churn']
        ))

    # --- 5.3 Best Model ---
    best_name = max(results, key=lambda k: results[k]['test_auc'])
    best      = results[best_name]
    best_model = best['model']
    print(f"\n🏆 Best Model: {best_name}  (Test AUC = {best['test_auc']:.4f})")

    # --- 5.4 Evaluation Plots ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Model Evaluation — Telco Churn', fontsize=14, fontweight='bold')

    # ROC Curves
    line_styles = ['-', '--', ':']
    for (name, res), ls in zip(results.items(), line_styles):
        fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
        axes[0].plot(fpr, tpr, lw=2, ls=ls, label=f"{name} (AUC={res['test_auc']:.3f})")
    axes[0].plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
    axes[0].set_title('ROC Curves — All Models')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # Confusion Matrix — best model
    cm   = confusion_matrix(y_test, best['y_pred'])
    disp = ConfusionMatrixDisplay(cm, display_labels=['No Churn', 'Churn'])
    disp.plot(ax=axes[1], colorbar=False, cmap='Blues')
    axes[1].set_title(f'Confusion Matrix\n({best_name})')

    # Feature Importance
    feature_importances_dict = None
    if hasattr(best_model, 'feature_importances_'):
        feat_imp = pd.Series(best_model.feature_importances_, index=X.columns)
        feature_importances_dict = dict(zip(X.columns, best_model.feature_importances_))
        
        top15    = feat_imp.nlargest(15).sort_values()
        top15.plot(kind='barh', ax=axes[2], color='#3498db', edgecolor='black')
        axes[2].set_title(f'Top 15 Feature Importances\n({best_name})')
        axes[2].set_xlabel('Importance Score')
    elif hasattr(best_model, 'coef_'):
        # Fallback for Logistic Regression
        feature_importances_dict = dict(zip(X.columns, best_model.coef_[0]))
        feat_imp = pd.Series(np.abs(best_model.coef_[0]), index=X.columns)
        top15 = feat_imp.nlargest(15).sort_values()
        top15.plot(kind='barh', ax=axes[2], color='#3498db', edgecolor='black')
        axes[2].set_title('Top 15 Absolute Coefficients\n(Logistic Regression)')
        axes[2].set_xlabel('Coefficient Magnitude')
    else:
        axes[2].axis('off')
        axes[2].set_title('Feature Importance Not Supported')

    plt.tight_layout()
    plt.savefig('model_evaluation.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n✅ Evaluation plots saved → model_evaluation.png")

    # --- 5.5 Final Summary Table ---
    print("\n" + "=" * 65)
    print("  FINAL MODEL COMPARISON")
    print("=" * 65)
    print(f"\n{'Model':<25} {'CV AUC':>10} {'Test AUC':>10}")
    print("-" * 48)
    for name, res in results.items():
        marker = "  🏆" if name == best_name else ""
        print(f"{name:<25} {res['cv_mean']:>10.4f} {res['test_auc']:>10.4f}{marker}")

    # Compile Best Model Metrics
    model_metrics = {
        'Best Model': best_name,
        'Accuracy': accuracy_score(y_test, best['y_pred']),
        'Precision': precision_score(y_test, best['y_pred']),
        'Recall': recall_score(y_test, best['y_pred']),
        'F1 Score': f1_score(y_test, best['y_pred']),
        'ROC AUC': best['test_auc'],
        'Cross Validation Score': best['cv_mean']
    }

    return best_model, model_metrics, feature_importances_dict


def save_artifacts(model, scaler, label_encoders, feature_columns, metrics, feature_importance, sample_input):
    """Saves all preprocessing objects, metadata, and models to the model directory."""
    joblib.dump(model, MODEL_DIR / 'churn_model.pkl')
    joblib.dump(scaler, MODEL_DIR / 'scaler.pkl')
    joblib.dump(label_encoders, MODEL_DIR / 'label_encoders.pkl')
    joblib.dump(feature_columns, MODEL_DIR / 'feature_columns.pkl')
    joblib.dump(metrics, MODEL_DIR / 'model_metrics.pkl')
    
    if feature_importance:
        joblib.dump(feature_importance, MODEL_DIR / 'feature_importance.pkl')
        
    joblib.dump(sample_input, MODEL_DIR / 'sample_input.pkl')


# =============================================================================
# MAIN EXECUTION PIPELINE
# =============================================================================
def main():
    try:
        # 1. Load Data
        df = load_data(DATA_PATH)

        # 2. Exploratory Data Analysis
        perform_eda(df)

        # 3. Data Cleaning
        df_clean = clean_data(df)

        # 4. Feature Engineering
        df_feat = engineer_features(df_clean)

        # 5. Preprocessing & Encoding
        X, y, scaler, label_encoders_dict, feature_columns = preprocess_data(df_feat)

        # 6. Model Training & Evaluation
        best_model, model_metrics, feature_importance = train_and_evaluate(X, y)

        # Extract a sample processed row for testing the Streamlit app later
        sample_input = X.iloc[[0]]

        # 7. Save Artifacts for Streamlit Deployment
        save_artifacts(
            model=best_model,
            scaler=scaler,
            label_encoders=label_encoders_dict,
            feature_columns=feature_columns,
            metrics=model_metrics,
            feature_importance=feature_importance,
            sample_input=sample_input
        )

        print("\nPipeline completed successfully.")
        print("\nGenerated PKL files:")
        for pkl_file in MODEL_DIR.glob('*.pkl'):
            print(f"  - {pkl_file.name}")

    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()