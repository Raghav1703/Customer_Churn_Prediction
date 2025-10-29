import pandas as pd 
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

#Load the dataset
df=pd.read_csv(r"C:\Users\Raghav\OneDrive\Desktop\Customer Churn Prediction\data\Telco-Customer-Churn.csv")

#Convert 'TotalCharges' to numeric, forcing errors to NaN
df['TotalCharges']=pd.to_numeric(df['TotalCharges'],errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].mean(),inplace=True)

df.drop(columns=['customerID'],inplace=True) #Dropping the CustomerID column because it's not useful for prediction

object_col=df.select_dtypes(include=['object']).columns #Selecting the categorical columns with object datatype

encoders={}
#Encode categorical features 
for col in object_col:
    le=LabelEncoder()
    df[col]=le.fit_transform(df[col])
    encoders[col]=le

#Splitting the dataset into X and y
X=df.drop(columns=['Churn'])
y=df['Churn']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#Handle class imbalance
smote=SMOTE(random_state=42)
X_train_smote,y_train_smote=smote.fit_resample(X_train,y_train)

#Scale numeric features
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train_smote)
X_test_scaled=scaler.transform(X_test)

models={
    "LogisticRegression": LogisticRegression(max_iter=5000,solver='lbfgs'),
    "RandomForest": RandomForestClassifier(random_state=42),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "SVM": SVC(kernel='rbf', probability=True, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42,use_label_encoder=False, eval_metric='logloss')
}

best_model_name=None 
best_cv_score=0
for model_name,model in models.items():
    print(f"Training {model_name}...")
    scores=cross_val_score(model,X_train_scaled,y_train_smote,cv=5,scoring='accuracy')
    print(f"{model_name} CV Accuracy: {scores.mean():.4f}")
    if scores.mean()>best_cv_score:
        best_cv_score=scores.mean()
        best_model_name=model_name

best_model=models[best_model_name]
best_model.fit(X_train_scaled,y_train_smote)

y_test_pred=best_model.predict(X_test_scaled)
print(f"Best Model: {best_model_name}")
print("Accuracy Score: ", accuracy_score(y_test,y_test_pred))
print("Classification Report: \n", classification_report(y_test,y_test_pred))
print("Confusion Matrix: \n", confusion_matrix(y_test,y_test_pred))

joblib.dump(best_model,"models/churn_model.pkl")
joblib.dump(scaler,"models/scaler.pkl")
joblib.dump(encoders,"models/encoders.pkl")

