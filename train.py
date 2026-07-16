"""
train.py
Trains the Employee Attrition prediction model (Logistic Regression - best
performer from the original notebook: 89.9% accuracy, 0.83 precision) and
saves all artifacts needed for the Streamlit app.

Run:
    python train.py
Requires:
    Human_Resources.csv in the same directory (or pass --data path)
"""

import argparse
import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

CAT_COLS = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus']
NUM_COLS = [
    'Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction',
    'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome',
    'MonthlyRate', 'NumCompaniesWorked', 'OverTime', 'PercentSalaryHike',
    'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel',
    'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
    'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
    'YearsWithCurrManager'
]
DROP_COLS = ['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber']


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['Attrition'] = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
    df['OverTime'] = df['OverTime'].apply(lambda x: 1 if x == 'Yes' else 0)
    df['Over18'] = df['Over18'].apply(lambda x: 1 if x == 'Y' else 0)
    df.drop(DROP_COLS, axis=1, inplace=True)
    return df


def build_features(df: pd.DataFrame, encoder: OneHotEncoder = None, scaler: MinMaxScaler = None):
    X_cat_raw = df[CAT_COLS]

    if encoder is None:
        encoder = OneHotEncoder(handle_unknown='ignore')
        X_cat = encoder.fit_transform(X_cat_raw).toarray()
    else:
        X_cat = encoder.transform(X_cat_raw).toarray()

    X_cat = pd.DataFrame(X_cat, columns=encoder.get_feature_names_out(CAT_COLS), index=df.index)
    X_num = df[NUM_COLS]
    X_all = pd.concat([X_cat, X_num], axis=1)

    if scaler is None:
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X_all)
    else:
        X = scaler.transform(X_all)

    return X, encoder, scaler, X_all.columns.tolist()


def main(data_path: str, out_dir: str):
    df = load_and_clean(data_path)
    y = df['Attrition']

    X, encoder, scaler, feature_names = build_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # --- Logistic Regression (best performer in the original notebook) ---
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Logistic Regression Accuracy: {acc * 100:.2f}%")
    print(report)
    print("Confusion Matrix:\n", cm)

    # Save all artifacts the Streamlit app needs
    joblib.dump(lr, f"{out_dir}/model.pkl")
    joblib.dump(encoder, f"{out_dir}/encoder.pkl")
    joblib.dump(scaler, f"{out_dir}/scaler.pkl")
    joblib.dump(feature_names, f"{out_dir}/feature_names.pkl")
    joblib.dump(CAT_COLS, f"{out_dir}/cat_cols.pkl")
    joblib.dump(NUM_COLS, f"{out_dir}/num_cols.pkl")

    # Save unique category values for building Streamlit dropdowns
    cat_options = {col: sorted(df[col].astype(str).unique().tolist())
                   if col in df.columns else [] for col in CAT_COLS}
    # BusinessTravel/Department/etc are still strings in df pre-encoding
    joblib.dump(cat_options, f"{out_dir}/cat_options.pkl")

    print(f"\nArtifacts saved to '{out_dir}/'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="Human_Resources.csv")
    parser.add_argument("--out", default="models")
    args = parser.parse_args()
    main(args.data, args.out)
