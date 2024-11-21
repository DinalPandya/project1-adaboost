import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from joblib import dump
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split


# Load the dataset
telecom_cust = pd.read_csv('Telco_Customer_Churn.csv')

# Data preprocessing
# Fill missing values in 'TotalCharges' and convert to numeric
telecom_cust['TotalCharges'] = pd.to_numeric(telecom_cust['TotalCharges'], errors='coerce')
telecom_cust['TotalCharges'].fillna(0, inplace=True)

# Convert 'Churn' to binary labels
label_encoder = LabelEncoder()
telecom_cust['Churn'] = label_encoder.fit_transform(telecom_cust['Churn'])

# Use Label Encoding for 'InternetService' and 'Contract'
telecom_cust['InternetService'] = label_encoder.fit_transform(telecom_cust['InternetService'])
telecom_cust['Contract'] = label_encoder.fit_transform(telecom_cust['Contract'])

# Select features
selected_features = ['tenure', 'InternetService', 'Contract', 'MonthlyCharges', 'TotalCharges']
X = telecom_cust[selected_features]
y = telecom_cust['Churn']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=99)
# Train the AdaBoost model
model_ada = AdaBoostClassifier(n_estimators=200)
model_ada.fit(X_train, y_train)

# Save the trained model to a file
dump(model_ada, 'ada_boost_model.joblib')
