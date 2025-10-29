# Customer Churn Prediction

An **end-to-end Machine Learning project** to predict customer churn using **Python, Flask, and Power BI**.  
This project analyzes telecom customer data to identify customers who are likely to leave the service (churn) and helps the business improve retention strategies.

---

## Project Overview

Customer churn is a major challenge for telecom companies.  
This project:
- Analyzes customer behavior and demographic data.
- Builds multiple ML models to predict churn likelihood.
- Deploys the best model using a **Flask web app** for real-time predictions.
- Visualizes insights and churn patterns using **Power BI** dashboards.

---

## Tech Stack

| Category | Tools |
|-----------|-------|
| **Language** | Python |
| **Framework** | Flask |
| **Libraries** | Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn, Joblib |
| **Visualization** | Power BI |
| **Frontend** | HTML, Bootstrap |
| **Environment** | VS Code (Virtual Environment) |

---
## Dashboard Preview
![Customer Churn Dashboard](https://github.com/Raghav1703/Customer_Churn_Prediction/blob/main/images/Dashboard.png?raw=true)
## Project Structure

```bash
Customer_Churn_Prediction/
│
├── data/
│   └── Telco-Customer-Churn.csv           # Raw dataset
│
├── models/
│   ├── churn_model.pkl                    # Trained model
│   ├── scaler.pkl                         # Scaler used for normalization
│   └── encoders.pkl                       # Label encoders for categorical features
│
├── notebooks/
│   ├── churn_analysis.ipynb               # EDA & preprocessing
│   ├── train_model.py                     # Model training & selection
│   └── generate_predictions.py            # Generate churn_predictions.csv for Power BI
│
├── templates/
│   └── index.html                         # Web app frontend (Flask UI)
│
├── app.py                                 # Flask backend
├── requirements.txt                       # All dependencies
├── Customer_Churn_Prediction.pbix         # Power BI dashboard
└── README.md                              # Project documentation
└── images/
    └── dashboard.png


