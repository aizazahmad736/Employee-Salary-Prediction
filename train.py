import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

def train_model():
    # 1. Load the generated dataset
    data_path = os.path.join("data", "employee_salaries.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please run generate_data.py first.")
    
    df = pd.read_csv(data_path)
    print("Dataset Loaded Successfully!")
    print(df.head())
    print("\n--- Encoding Categorical Variables ---")
    
    # 2. ENCODING CATEGORICAL VARIABLES
    # A. Ordinal Encoding for 'Education_Level' (since education has an inherent order)
    education_order = {
        "High School": 0,
        "Bachelor's": 1,
        "Master's": 2,
        "PhD": 3
    }
    df['Education_Encoded'] = df['Education_Level'].map(education_order)
    print("Ordinal Encoding applied to 'Education_Level':")
    print(df[['Education_Level', 'Education_Encoded']].drop_duplicates().sort_values('Education_Encoded'))
    
    # B. One-Hot Encoding for Nominal variables ('Job_Role' and 'Location')
    # We will use pandas get_dummies, keeping track of column structures for future predictions
    df_encoded = pd.get_dummies(df, columns=['Job_Role', 'Location'], drop_first=True)
    
    print("\nColumns after One-Hot Encoding (with drop_first=True to avoid dummy variable trap):")
    print(list(df_encoded.columns))
    
    # 3. Define Features (X) and Target (y)
    # We drop the original categorical text columns and the target variable
    X = df_encoded.drop(columns=['Education_Level', 'Salary'])
    y = df_encoded['Salary']
    
    # Save the feature column names to ensure consistency in predictions later
    feature_columns = list(X.columns)
    
    # 4. Split data into Training and Testing sets (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")
    
    # 5. LINEAR REGRESSION TRAINING
    print("\n--- Training Linear Regression Model ---")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 6. Evaluation
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Training Complete!")
    print(f"Mean Absolute Error (MAE): ${mae:.2f}")
    print(f"R-squared Score (R2): {r2:.4f} (Model explains {r2*100:.2f}% of the salary variance)")
    
    # Output the coefficients to show linear regression interpretation
    print("\n--- Model Coefficients & Intercept ---")
    print(f"Base Intercept: ${model.intercept_:.2f}")
    for col, coef in zip(X.columns, model.coef_):
        print(f"  {col}: {'+' if coef >= 0 else ''}${coef:.2f}")
        
    # 7. Save model and encoding metadata
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)
    
    artifacts = {
        "model": model,
        "feature_columns": feature_columns,
        "education_order": education_order
    }
    
    with open(os.path.join(model_dir, "salary_model_artifacts.pkl"), "wb") as f:
        pickle.dump(artifacts, f)
        
    print("\nModel and preprocessing artifacts saved successfully to model/salary_model_artifacts.pkl")

if __name__ == "__main__":
    train_model()
