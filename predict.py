import os
import pickle

import pandas as pd


def load_artifacts():
    path = os.path.join("model", "salary_model_artifacts.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model artifacts not found at {path}. Please run train.py first."
        )

    with open(path, "rb") as f:
        return pickle.load(f)


def normalize_education(education):
    if education is None:
        raise ValueError("Education is required.")

    education_map = {
        "high school": "High School",
        "bachelor's": "Bachelor's",
        "bachelors": "Bachelor's",
        "master's": "Master's",
        "masters": "Master's",
        "phd": "PhD",
    }

    normalized = str(education).strip()
    match = education_map.get(normalized.lower(), normalized)
    if match not in {"High School", "Bachelor's", "Master's", "PhD"}:
        raise ValueError(
            f"Invalid education level '{education}'. Allowed values: High School, Bachelor's, Master's, PhD."
        )
    return match


def normalize_role(role):
    if role is None:
        raise ValueError("Job role is required.")

    role_map = {
        "junior developer": "Junior Developer",
        "senior developer": "Senior Developer",
        "technical lead": "Technical Lead",
        "manager": "Manager",
        "director": "Director",
    }

    normalized = str(role).strip()
    match = role_map.get(normalized.lower(), normalized)
    if match not in {
        "Junior Developer",
        "Senior Developer",
        "Technical Lead",
        "Manager",
        "Director",
    }:
        raise ValueError(
            f"Invalid job role '{role}'. Allowed values: Junior Developer, Senior Developer, Technical Lead, Manager, Director."
        )
    return match


def normalize_location(location):
    if location is None:
        raise ValueError("Location is required.")

    location_map = {
        "rural": "Rural",
        "suburban": "Suburban",
        "urban": "Urban",
        "metro": "Metro",
    }

    normalized = str(location).strip()
    match = location_map.get(normalized.lower(), normalized)
    if match not in {"Rural", "Suburban", "Urban", "Metro"}:
        raise ValueError(
            f"Invalid location '{location}'. Allowed values: Rural, Suburban, Urban, Metro."
        )
    return match


def prepare_feature_row(years_exp, education, role, location, artifacts):
    feature_columns = artifacts["feature_columns"]
    education_order = artifacts["education_order"]

    education = normalize_education(education)
    role = normalize_role(role)
    location = normalize_location(location)

    try:
        years_exp = float(years_exp)
    except (TypeError, ValueError):
        raise ValueError(f"Years of experience must be numeric. Received: {years_exp}")

    input_data = {col: 0 for col in feature_columns}
    input_data["Years_Experience"] = years_exp
    input_data["Education_Encoded"] = education_order.get(education, 0)

    role_col = f"Job_Role_{role}"
    if role_col in input_data:
        input_data[role_col] = 1

    loc_col = f"Location_{location}"
    if loc_col in input_data:
        input_data[loc_col] = 1

    input_df = pd.DataFrame([input_data])
    return input_df[feature_columns]


def predict_salary(years_exp, education, role, location, artifacts):
    model = artifacts["model"]
    input_df = prepare_feature_row(years_exp, education, role, location, artifacts)
    predicted_salary = model.predict(input_df)[0]
    return predicted_salary


def predict_salary_batch(records, artifacts):
    """records: list of dicts with Years_Experience, Education_Level, Job_Role, Location"""
    predictions = []

    for index, record in enumerate(records, start=1):
        required_fields = ["Years_Experience", "Education_Level", "Job_Role", "Location"]
        missing = [field for field in required_fields if field not in record]
        if missing:
            raise ValueError(f"Record {index} is missing fields: {missing}")

        predicted = predict_salary(
            record["Years_Experience"],
            record["Education_Level"],
            record["Job_Role"],
            record["Location"],
            artifacts,
        )

        predictions.append(
            {
                "Years_Experience": record["Years_Experience"],
                "Education_Level": normalize_education(record["Education_Level"]),
                "Job_Role": normalize_role(record["Job_Role"]),
                "Location": normalize_location(record["Location"]),
                "Predicted_Salary": float(predicted),
            }
        )

    return pd.DataFrame(predictions)


def load_csv_predictions(csv_path, artifacts):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_columns = ["Years_Experience", "Education_Level", "Job_Role", "Location"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"CSV file is missing required columns: {missing_columns}. "
            "Expected: Years_Experience, Education_Level, Job_Role, Location."
        )

    records = df.to_dict(orient="records")
    return predict_salary_batch(records, artifacts)


def print_prediction_summary(predictions_df):
    if predictions_df.empty:
        print("No predictions to display.")
        return

    print("\n===== Batch Prediction Summary =====")
    print(predictions_df.to_string(index=False, formatters={"Predicted_Salary": lambda x: f"${x:,.2f}"}))

    avg_salary = predictions_df["Predicted_Salary"].mean()
    min_salary = predictions_df["Predicted_Salary"].min()
    max_salary = predictions_df["Predicted_Salary"].max()

    print("\nSummary Statistics:")
    print(f"Average Salary: ${avg_salary:,.2f}")
    print(f"Lowest Salary:  ${min_salary:,.2f}")
    print(f"Highest Salary: ${max_salary:,.2f}")
    print("====================================\n")


def prompt_manual_batch(artifacts):
    try:
        count = int(input("How many employees do you want to evaluate? "))
    except ValueError:
        raise ValueError("Please enter a valid number of employees.")

    if count <= 0:
        raise ValueError("Number of employees must be greater than zero.")

    records = []
    for i in range(1, count + 1):
        print(f"\nEmployee {i}")
        record = {
            "Years_Experience": float(input("Years of Experience: ")),
            "Education_Level": input("Education Level (High School / Bachelor's / Master's / PhD): "),
            "Job_Role": input("Job Role (Junior Developer / Senior Developer / Technical Lead / Manager / Director): "),
            "Location": input("Location (Rural / Suburban / Urban / Metro): "),
        }
        records.append(record)

    return predict_salary_batch(records, artifacts)


def run_single_prediction(artifacts):
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
            5: "Director",
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

        print("\n" + "=" * 45)
        print(f" Predicted Salary: ${predicted:,.2f}")
        print("=" * 45)

    except ValueError as exc:
        print(f"Invalid input: {exc}")


def run_batch_prediction(artifacts):
    print("\nChoose batch input mode:")
    print("1. Add employees manually")
    print("2. Load from CSV file")
    mode = input("Select option (1-2): ")

    try:
        if mode == "1":
            predictions = prompt_manual_batch(artifacts)
        elif mode == "2":
            csv_path = input("Enter CSV file path: ").strip()
            predictions = load_csv_predictions(csv_path, artifacts)
        else:
            raise ValueError("Please select either 1 or 2.")

        print_prediction_summary(predictions)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Batch prediction error: {exc}")


def main():
    try:
        artifacts = load_artifacts()
    except FileNotFoundError as e:
        print(e)
        return

    print("=========================================")
    print("      Employee Salary Predictor App      ")
    print("=========================================\n")

    print("Choose a mode:")
    print("1. Single Employee Prediction")
    print("2. Batch Prediction")
    choice = input("Select option (1-2): ")

    if choice == "1":
        run_single_prediction(artifacts)
    elif choice == "2":
        run_batch_prediction(artifacts)
    else:
        print("Invalid choice. Please select 1 or 2.")


if __name__ == "__main__":
    main()
