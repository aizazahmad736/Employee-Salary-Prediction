# Employee Salary Prediction Project

This project predicts an employee's salary based on four key features:
1. **Years of Experience** (Numerical)
2. **Education Level** (Categorical / Ordinal)
3. **Job Role** (Categorical / Nominal)
4. **Location Type** (Categorical / Nominal)

---

## What You Will Learn Here

### 1. Encoding Categorical Variables
Machine Learning models (like Linear Regression) only work with numbers. We cannot feed text strings like `"Bachelor's"` or `"Manager"` directly into the model. We must convert them to numbers.

There are two primary ways to do this:

*   **Ordinal Encoding**:
    *   **When to use**: When the categories have a natural sequence or ordering (e.g., Education: High School < Bachelor's < Master's < PhD).
    *   **How it works**: We assign a sequential integer to each category (e.g., High School = 0, Bachelor's = 1, Master's = 2, PhD = 3). The linear model learns that as the encoded number increases, the target variable (salary) increases/decreases proportionally.
*   **One-Hot Encoding**:
    *   **When to use**: When the categories have no inherent order (e.g., Job Role: Developer, Manager, Director; or Location: Rural, Urban). If we ordinal-encoded these (e.g., Rural=0, Urban=1, Metro=2), the model would think Metro is twice "something" of Urban, which doesn't make logical sense.
    *   **How it works**: We create a new binary column (0 or 1) for each category. For example, if we have location categories: `Rural`, `Suburban`, `Urban`. If a row is `Urban`, the columns will be: `Rural = 0`, `Suburban = 0`, `Urban = 1`.
    *   **Dummy Variable Trap**: When One-Hot Encoding, if we keep all category columns, we introduce perfect multicollinearity (since if `Rural=0` and `Suburban=0`, then `Urban` *must* be 1). To avoid this, we drop the first column using `drop_first=True`. The dropped category becomes the baseline.

---

### 2. Linear Regression
Linear Regression assumes a linear relationship between the input features (\(X\)) and the target variable (\(y\)):

\[y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_n X_n + \epsilon\]

Where:
*   \(y\) is the predicted **Salary**.
*   \(\beta_0\) is the **Intercept** (the baseline salary if all features are zero).
*   \(\beta_i\) are the **Coefficients** (weights representing how much the salary changes for every unit increase in that feature).
*   \(X_i\) are the features (Experience, Ordinal Education, and One-Hot Encoded variables).

For example, if the coefficient for `Years_Experience` is `+$3,200`, it means that for every additional year of experience, the predicted salary increases by exactly `$3,200`, holding all other factors constant.

---

<img width="946" height="439" alt="Screenshot 2026-07-30 162957" src="https://github.com/user-attachments/assets/30220a07-19f3-443f-afbc-5d0769881215" />


## Project Structure
*   [generate_data.py](file:///c:/Users/Aizaz%20Ahmad/Desktop/Antigravity-Projects/employee-salary-predictor/generate_data.py): Synthesizes a realistic dataset of 1,000 employees.
*   [train.py](file:///c:/Users/Aizaz%20Ahmad/Desktop/Antigravity-Projects/employee-salary-predictor/train.py): Performs preprocessing, applies categorical encoding, trains the model, and saves artifacts.
*   [predict.py](file:///c:/Users/Aizaz%20Ahmad/Desktop/Antigravity-Projects/employee-salary-predictor/predict.py): Interactive CLI application to predict salary for custom employee profiles.

---

## How to Run

1. **Install Dependencies**:
   ```bash
   pip install pandas numpy scikit-learn
   ```

2. **Generate the Dataset**:
   ```bash
   python generate_data.py
   ```

3. **Train the Model**:
   ```bash
   python train.py
   ```

4. **Predict Salaries**:
   ```bash
   python predict.py
   ```
