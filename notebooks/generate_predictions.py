import pandas as pd 
import joblib
import os 

model=joblib.load("models/churn_model.pkl")
scaler=joblib.load("models/scaler.pkl")
encoders=joblib.load("models/encoders.pkl")

df=pd.read_csv(r"C:\Users\Raghav\OneDrive\Desktop\Customer Churn Prediction\data\Telco-Customer-Churn.csv")

df['TotalCharges']=pd.to_numeric(df['TotalCharges'],errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].mean(),inplace=True)
df.drop(columns=['customerID'],inplace=True)

for col,le in encoders.items():
    df[col]=le.transform(df[col])

X=df.drop(columns=['Churn'])
X_scaled=scaler.transform(X)

df["PredictedChurn"]=model.predict(X_scaled)
df["ChurnProbability"]=model.predict_proba(X_scaled)[:,1]

os.makedirs("../output", exist_ok=True)
output_path="../output/churn_predictions.csv"
df.to_csv(output_path, index=False)
