from flask import Flask, render_template, request
import joblib
import pandas as pd
import os 

app = Flask(__name__)

# Load model, scaler, and encoders
model = joblib.load("models/churn_model.pkl")
scaler = joblib.load("models/scaler.pkl")
encoders = joblib.load("models/encoders.pkl")

# Define features in same order used during training
feature_names = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges'
]

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        form_data=[request.form.get(f) for f in feature_names]
        df=pd.DataFrame([form_data],columns=feature_names)

        numeric_cols=['SeniorCitizen','tenure','MonthlyCharges','TotalCharges']
        df[numeric_cols]=df[numeric_cols].apply(pd.to_numeric)

        for col,le in encoders.items():
            if col in df.columns:
                df[col]=le.transform(df[col])
        
        df_scaled=scaler.transform(df)

        prediction=model.predict(df_scaled)[0]
        result="Customer will NOT Churn" if prediction==0 else "Customer will Churn"

        return render_template('index.html', prediction=result)

    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
