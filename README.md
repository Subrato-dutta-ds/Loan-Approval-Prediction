# 🏦 Loan Approval Prediction App

A Machine Learning web application that predicts whether a loan will be approved or not based on applicant details.

## 🚀 Features
- Clean Streamlit UI
- Real-time prediction
- Data preprocessing & feature engineering
- Interactive graph visualization

## 🛠 Tech Stack
- Python
- Pandas
- Scikit-learn
- Streamlit
- Matplotlib

## ▶️ Run Locally
```bash
pip install streamlit pandas scikit-learn matplotlib
streamlit run app.py

## 🤖 Model Details

- Algorithm: Random Forest Classifier
- Problem Type: Classification
- Target Variable: Loan_Status
- Key Features:
  - Credit History
  - Applicant Income
  - Loan Amount
  - Property Area

### 📌 Why Random Forest?
Random Forest was chosen because it handles both categorical and numerical data well and provides better accuracy by combining multiple decision trees.

### 📈 Model Workflow
1. Data Cleaning & Preprocessing  
2. Encoding categorical variables  
3. Handling missing values  
4. Feature selection  
5. Model training  
6. Prediction using Streamlit UI  

