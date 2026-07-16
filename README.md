# Employee Attrition Predictor

End-to-end ML app that predicts whether an employee is likely to leave the
company, built on the IBM HR Analytics dataset (1,470 employees, 35 features).

**Model:** Logistic Regression
**Test Accuracy:** 87.2%
**Precision (attrition class):** 0.70
**Recall (attrition class):** 0.36

---

## Project Structure

```
hr-attrition-app/
├── app.py                 # Streamlit UI + inference
├── train.py                # Training script (run once to (re)generate model artifacts)
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── Human_Resources.csv     # training data (not included in Docker image)
└── models/                 # saved model + preprocessing artifacts
    ├── model.pkl
    ├── encoder.pkl
    ├── scaler.pkl
    ├── feature_names.pkl
    ├── cat_cols.pkl
    ├── num_cols.pkl
    └── cat_options.pkl
```

---

## 1. Run Locally (no Docker)

```bash
pip install -r requirements.txt

# (Optional) retrain the model — artifacts are already included in models/
python train.py --data Human_Resources.csv --out models

# Launch the app
streamlit run app.py
```
App will open at `http://localhost:8501`

---

## 2. Run with Docker

```bash
# Build the image
docker build -t hr-attrition-app .

# Run the container
docker run -p 8501:8501 hr-attrition-app
```
App will be available at `http://localhost:8501`

---

## 3. Deploy on Streamlit Community Cloud

Streamlit Cloud builds directly from your GitHub repo using `requirements.txt`
(it does not use the Dockerfile — that's for local/portable deployment).

1. Push this folder to a public GitHub repo (include `app.py`, `requirements.txt`,
   and the `models/` folder — do **not** need to push `Human_Resources.csv` or
   `train.py` unless you want reproducibility).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo, branch, and set main file path to `app.py`
4. Click **Deploy**

Your app will be live at:
`https://<your-app-name>.streamlit.app`

---

## Tech Stack

Python · Pandas · Scikit-learn · Streamlit · Docker

## Model Evaluation Summary

| Metric | Class 0 (Stayed) | Class 1 (Left) |
|---|---|---|
| Precision | 0.89 | 0.70 |
| Recall | 0.97 | 0.36 |
| F1-score | 0.93 | 0.47 |

Accuracy: **87.2%** on held-out test set (25% split, stratified)
