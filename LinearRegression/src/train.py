import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Reading CSV data
salary = pd.read_csv('../input/salary.csv')

# Target and Features
X = salary[['YearsExperience']]
y = salary['Salary']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

# Select model and train
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, '../models/linear_model.pkl')