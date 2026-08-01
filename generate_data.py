import pandas as pd
import numpy as np
import os

def generate_salary_data(num_samples=1000, seed=42):
    np.random.seed(seed)
    
    # 1. Features
    # Experience (0 to 20 years)
    experience = np.random.randint(0, 21, size=num_samples)
    
    # Education Level (High School, Bachelor's, Master's, PhD)
    education_levels = ["High School", "Bachelor's", "Master's", "PhD"]
    education = np.random.choice(education_levels, size=num_samples, p=[0.15, 0.55, 0.22, 0.08])
    
    # Job Role (Junior Dev, Senior Dev, Tech Lead, Manager, Director)
    job_roles = ["Junior Developer", "Senior Developer", "Technical Lead", "Manager", "Director"]
    roles = np.random.choice(job_roles, size=num_samples, p=[0.35, 0.30, 0.15, 0.15, 0.05])
    
    # Location (Rural, Suburban, Urban, Metro)
    locations = ["Rural", "Suburban", "Urban", "Metro"]
    locs = np.random.choice(locations, size=num_samples, p=[0.15, 0.35, 0.35, 0.15])
    
    # 2. Define salary coefficients (base and effect of each factor)
    base_salary = 35000  # Base salary in USD
    
    # Linear effect of experience
    exp_effect = experience * 3200
    
    # Effect of education (ordinal mapping)
    edu_map = {
        "High School": 0,
        "Bachelor's": 12000,
        "Master's": 25000,
        "PhD": 45000
    }
    edu_effect = np.array([edu_map[e] for e in education])
    
    # Effect of job role
    role_map = {
        "Junior Developer": 5000,
        "Senior Developer": 25000,
        "Technical Lead": 45000,
        "Manager": 55000,
        "Director": 85000
    }
    role_effect = np.array([role_map[r] for r in roles])
    
    # Effect of location
    loc_map = {
        "Rural": -5000,
        "Suburban": 5000,
        "Urban": 15000,
        "Metro": 28000
    }
    loc_effect = np.array([loc_map[l] for l in locs])
    
    # Add some random noise (normally distributed)
    noise = np.random.normal(0, 4000, size=num_samples)
    
    # Calculate Salary
    salary = base_salary + exp_effect + edu_effect + role_effect + loc_effect + noise
    
    # Ensure no salaries are below a realistic minimum
    salary = np.clip(salary, 25000, None)
    
    # Create DataFrame
    df = pd.DataFrame({
        "Years_Experience": experience,
        "Education_Level": education,
        "Job_Role": roles,
        "Location": locs,
        "Salary": np.round(salary, 2)
    })
    
    return df

if __name__ == "__main__":
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    
    df = generate_salary_data()
    output_path = os.path.join(output_dir, "employee_salaries.csv")
    df.to_csv(output_path, index=False)
    print(f"Dataset generated and saved to: {output_path}")
    print(df.head())
