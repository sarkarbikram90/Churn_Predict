# Customer Churn Prediction (MLflow + Streamlit)

This project demonstrates an **end-to-end customer churn prediction workflow**, covering **model training, experiment tracking, and live inference deployment**.

The model is trained locally using **MLflow** for experiment tracking and then deployed as a lightweight **Streamlit web app** for real-time predictions.

---

## What this project shows

- 📊 Customer churn prediction using a supervised ML model  
- 🧪 MLflow for experiment tracking during training  
- 🚀 Streamlit app for interactive model inference  
- 🧱 Clear separation between training and serving  

---

## Tech Stack

- Python  
- scikit-learn  
- MLflow (training & tracking only)  
- Streamlit (deployment)  
- Pandas, NumPy  

---

## How it works

1. The model is trained locally on the Telco Customer Churn dataset  
2. Experiments and metrics are tracked using MLflow  
3. The trained model is exported as a serialized artifact  
4. Streamlit loads the exported model and serves predictions via a web UI  

---

## Running the app locally

```bash
pip install -r requirements.txt
streamlit run app.py

## Notes

MLflow is not required at inference time
The Streamlit app loads a pre-trained model artifact
Designed as a clean, minimal demo of an ML lifecycle

## Use case

This project is intended as a demo-quality reference for showcasing:
ML lifecycle understanding
MLOps fundamentals
Clean deployment practices