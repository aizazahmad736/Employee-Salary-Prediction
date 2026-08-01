import pandas as pd
import numpy as np
import pickle
import os

def load_artifacts():
    path = os.path.join("model", "salary_model_artifacts.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model artifacts not found at {path}. Please run train.py first.")
    
    with open(path, "rb") as f:
        return pickle.load(f)

def predict_salary(years_exp, education, role, location, artifacts):
    model = artifacts["model"]
    feature_columns = artifacts["feature_columns"]
    education_order = artifacts["education_order"]
    
    # 1. Encode Education (Ordinal)
    edu_encoded = education_order.get(education, 0)
    
    # 2. Prepare user input as a single-row DataFrame matching training columns
    # Start with all columns set to 0
    input_data = {col: 0 for col in feature_columns}
    
    # Set experience and encoded education
    input_data["Years_Experience"] = years_exp
    input_data["Education_Encoded"] = edu_encoded
    
    # Set job role (One-hot encoded)
    # The columns look like 'Job_Role_Senior Developer'
    role_col = f"Job_Role_{role}"
    if role_col in input_data:
        input_data[role_col] = 1
        
    # Set location (One-hot encoded)
    # The columns look like 'Location_Suburban'
    loc_col = f"Location_{location}"
    if loc_col in input_data:
        input_data[loc_col] = 1
        
    # Create DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Reorder columns to match feature_columns exactly
    input_df = input_df[feature_columns]
    
    # Predict
    predicted_salary = model.predict(input_df)[0]
    return predicted_salary

def main():
    try:
        artifacts = load_artifacts()
    except FileNotFoundError as e:
        print(e)
        return
    
    print("=========================================")
    print("      Employee Salary Predictor App      ")
    print("=========================================\n")
    
    try:
        years_exp = float(input("Enter Years of Experience (e.g., 5): "))
        
        print("\nSelect Education Level:")
        print("1. High School")
        print("2. Bachelor's")
        print("3. Master's")
        print("4. PhD")
        edu_choice = int(input("Choice (1-4): "))
        edu_map = {1: "High School", 2: "Bachelor's", 3: "Master's", 4: "PhD"}
        education = edu_map.get(edu_choice, "Bachelor's")
        
        print("\nSelect Job Role:")
        print("1. Junior Developer")
        print("2. Senior Developer")
        print("3. Technical Lead")
        print("4. Manager")
        print("5. Director")
        role_choice = int(input("Choice (1-5): "))
        role_map = {
            1: "Junior Developer", 
            2: "Senior Developer", 
            3: "Technical Lead", 
            4: "Manager", 
            5: "Director"
        }
        role = role_map.get(role_choice, "Junior Developer")
        
        print("\nSelect Location:")
        print("1. Rural")
        print("2. Suburban")
        print("3. Urban")
        print("4. Metro")
        loc_choice = int(input("Choice (1-4): "))
        loc_map = {1: "Rural", 2: "Suburban", 3: "Urban", 4: "Metro"}
        location = loc_map.get(loc_choice, "Urban")
        
        predicted = predict_salary(years_exp, education, role, location, artifacts)
        
        print("\n" + "="*45)
        print(f" Predicted Salary: ${predicted:,.2f}")
        print("="*45)
        
    except ValueError:
        print("Invalid input. Please enter numbers where prompted.")

if __name__ == "__main__":
    main()
