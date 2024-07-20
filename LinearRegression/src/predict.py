import pandas as pd
import joblib

# Load model
model = joblib.load('../models/linear_model.pkl')

# Create example new_data
new_salary_data = pd.DataFrame({
    'YearsExperience': [1, 2, 3, 4, 5]
})

# Predict
predictions = model.predict(new_salary_data)
new_salary_data['Predicted_Salary'] = predictions

# Save predictions
new_salary_data.to_csv('../output/predictions.csv', index=False)

print(new_salary_data)