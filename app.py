import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# --- PAGE CONFIG ---
st.set_page_config(page_title="Loan Predictor Pro", layout="wide", page_icon="🏦")

# --- CUSTOM CSS (HIGH CONTRAST DARK THEME) ---
st.markdown("""
<style>
    /* --- HIDE STREAMLIT HEADER --- */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}

    /* Global Reset & Background */
    .stApp {
        background: radial-gradient(ellipse at 50% 50%, #0f172a, #020617);
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    .main > div {
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
    }
    .glass-card {
        background: rgba(15, 23, 42, 0.85);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 212, 255, 0.15);
        border-radius: 24px;
        padding: 24px 30px;
        margin-bottom: 16px;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.8);
    }
    .hero-title {
        font-size: 2.8rem;
        font-weight: 900;
        background: linear-gradient(135deg, #00d4ff, #7b2ffc, #00d4ff);
        background-size: 200% 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradientShift 4s ease infinite;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 0px;
        letter-spacing: -0.5px;
    }
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .sub-title {
        text-align: center;
        color: #94a3b8;
        font-size: 0.95rem;
        margin-bottom: 20px;
        border-bottom: 1px solid rgba(255,255,255,0.05);
        padding-bottom: 15px;
    }
    .stSelectbox label, .stNumberInput label {
        color: #e2e8f0 !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.3px;
        margin-bottom: 2px !important;
    }
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
        color: white !important;
    }
    div[data-baseweb="select"] input, div[data-baseweb="input"] input {
        color: white !important;
    }
    div[data-baseweb="select"]:hover > div, div[data-baseweb="input"]:hover > div {
        border-color: #00d4ff !important;
        box-shadow: 0 0 20px -5px rgba(0, 212, 255, 0.3);
    }
    div[data-testid="metric-container"] {
        background: rgba(0, 212, 255, 0.05);
        border: 1px solid #1e293b;
        border-radius: 16px;
        padding: 12px 16px;
        backdrop-filter: blur(4px);
        transition: 0.3s;
    }
    div[data-testid="metric-container"]:hover {
        border-color: #00d4ff;
        box-shadow: 0 0 30px -8px rgba(0, 212, 255, 0.2);
    }
    div[data-testid="metric-container"] label {
        color: #94a3b8 !important;
        font-weight: 500 !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="metric-container"] div[data-testid="metric-value"] {
        color: #ffffff !important;
        font-weight: 800 !important;
        font-size: 2rem !important;
    }
    .stButton button {
        width: 100%;
        background: linear-gradient(135deg, #00d4ff, #7b2ffc) !important;
        color: white !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        padding: 0.6rem 0rem !important;
        border: none !important;
        border-radius: 14px !important;
        box-shadow: 0 8px 25px -8px rgba(0, 212, 255, 0.4);
        transition: all 0.3s ease !important;
        margin-top: 8px !important;
    }
    .stButton button:hover {
        transform: scale(1.02) !important;
        box-shadow: 0 12px 35px -5px rgba(123, 47, 252, 0.6) !important;
    }
    .custom-success {
        background: rgba(34, 197, 94, 0.15) !important;
        border-left: 6px solid #22c55e !important;
        border-radius: 16px !important;
        padding: 18px 22px !important;
        border: 1px solid rgba(34, 197, 94, 0.2);
        color: #e2e8f0 !important;
    }
    .custom-error {
        background: rgba(239, 68, 68, 0.15) !important;
        border-left: 6px solid #ef4444 !important;
        border-radius: 16px !important;
        padding: 18px 22px !important;
        border: 1px solid rgba(239, 68, 68, 0.2);
        color: #e2e8f0 !important;
    }
    .custom-success b, .custom-error b { color: #ffffff !important; }
    .streamlit-expanderHeader {
        background: rgba(15, 23, 42, 0.8) !important;
        border-radius: 16px !important;
        color: #e2e8f0 !important;
        font-weight: 700 !important;
        border: 1px solid #1e293b !important;
    }
    .streamlit-expanderContent {
        background: rgba(15, 23, 42, 0.6) !important;
        border-radius: 0 0 16px 16px !important;
        padding: 20px !important;
        border: 1px solid #1e293b !important;
        border-top: none !important;
    }
    .block-container {
        padding-top: 0.5rem !important;
        padding-bottom: 0.5rem !important;
        max-width: 1200px !important;
    }
    hr {
        margin: 0.5rem 0 !important;
        border-color: rgba(255,255,255,0.05) !important;
    }
</style>
""", unsafe_allow_html=True)

# --- DATA LOADING & TRAINING ---
@st.cache_data
def load_data():
    try:
        return pd.read_csv("train_data.csv")
    except FileNotFoundError:
        st.error("❌ 'train_data.csv' not found. Please upload it to the same folder.")
        return None

@st.cache_resource
def train_model(df):
    df_clean = df.copy()
    
    df_clean['LoanAmount'].fillna(df_clean['LoanAmount'].median(), inplace=True)
    df_clean['Credit_History'].fillna(df_clean['Credit_History'].mode()[0], inplace=True)
    df_clean['Loan_Amount_Term'].fillna(df_clean['Loan_Amount_Term'].mode()[0], inplace=True)
    df_clean['Self_Employed'].fillna('No', inplace=True)
    df_clean['Gender'].fillna(df_clean['Gender'].mode()[0], inplace=True)
    df_clean['Married'].fillna(df_clean['Married'].mode()[0], inplace=True)
# Convert Dependents: replace '3+' with '3', then coerce to numeric, fill NaN with 0, and convert to int
    df_clean['Dependents'] = pd.to_numeric(
    df_clean['Dependents'].replace('3+', '3'),
    errors='coerce'
).fillna(0).astype(int)
    
    df_clean['Total_Income'] = df_clean['ApplicantIncome'] + df_clean['CoapplicantIncome']
    df_clean['EMI'] = df_clean['LoanAmount'] / df_clean['Loan_Amount_Term']
    df_clean['Debt_to_Income_Ratio'] = df_clean['EMI'] / (df_clean['Total_Income'] + 1e-5)
    
    le = LabelEncoder()
    cat_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
    for col in cat_cols:
        df_clean[col] = le.fit_transform(df_clean[col].astype(str))
    
    features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 
                'Credit_History', 'Property_Area', 'Total_Income', 'EMI', 'Debt_to_Income_Ratio']
    
    X = df_clean[features]
    y = df_clean['Loan_Status']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    feature_importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    
    return model, feature_importance, acc, cm, features

# --- MAIN APP ---
df = load_data()
if df is None:
    st.stop()

model, feature_importance, acc, cm, feature_cols = train_model(df)

st.markdown('<div class="hero-title">🏦 Smart Loan Predictor Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Advanced AI · Feature Engineering · <span style="color:#00d4ff;">Glass UI</span></div>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="medium")
    
    with col1:
        st.markdown("##### 👤 Personal Details")
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox("Dependents", [0, 1, 2, 3])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["No", "Yes"])
        
    with col2:
        st.markdown("##### 💰 Financial Details")
        app_income = st.number_input("Applicant Income (₹)", min_value=0, value=5000, step=500)
        co_income = st.number_input("Coapplicant Income (₹)", min_value=0, value=2000, step=500)
        loan_amount = st.number_input("Loan Amount (₹)", min_value=0, value=100000, step=10000)
        loan_term = st.number_input("Loan Term (Months)", min_value=12, max_value=480, value=360, step=12)
        credit = st.selectbox("Credit History", ["Good (1)", "Bad (0)"])
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
    st.markdown('</div>', unsafe_allow_html=True)

col_eng1, col_eng2, col_eng3 = st.columns(3)
total_income = app_income + co_income
emi = loan_amount / loan_term if loan_term > 0 else 0
debt_ratio = emi / (total_income + 1e-5)

with col_eng1:
    st.metric(label="🧮 Total Income", value=f"₹ {total_income:,.0f}")
with col_eng2:
    st.metric(label="📆 Monthly EMI", value=f"₹ {emi:,.0f}")
with col_eng3:
    st.metric(label="⚖️ Debt-to-Income", value=f"{debt_ratio:.4f}")

col_btn, col_result = st.columns([1, 2])

with col_btn:
    predict_btn = st.button("🚀 Predict Approval", use_container_width=True)

if predict_btn:
    input_gender = 1 if gender == "Male" else 0
    input_married = 1 if married == "Yes" else 0
    input_education = 1 if education == "Graduate" else 0
    input_self = 1 if self_employed == "Yes" else 0
    input_credit = 1 if credit == "Good (1)" else 0
    input_property = {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area]

    input_data = [[
        input_gender, input_married, dependents, input_education, input_self,
        input_credit, input_property, total_income, emi, debt_ratio
    ]]
    
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1] * 100

    with col_result:
        st.markdown("##### 📊 Prediction Result")
        if prediction == 1:
            st.markdown(f'<div class="custom-success">✅ <b>Loan Approved</b> with <span style="color:#22c55e; font-weight:800;">{prob:.1f}%</span> confidence.</div>', unsafe_allow_html=True)
            st.progress(int(prob)/100, text=f"Approval Probability: {prob:.1f}%")
        else:
            st.markdown(f'<div class="custom-error">❌ <b>Loan Rejected</b> with <span style="color:#ef4444; font-weight:800;">{100-prob:.1f}%</span> rejection probability.</div>', unsafe_allow_html=True)
            st.progress(int(prob)/100, text=f"Approval Probability: {prob:.1f}%")

st.divider()
with st.expander("📈 Model Performance Dashboard (Feature Importance & Confusion Matrix)", expanded=False):
    st.caption("💡 Screenshot this for your Data Science Portfolio!")
    
    col_d1, col_d2 = st.columns(2)
    
    with col_d1:
        st.subheader("🔑 Feature Importance")
        fig, ax = plt.subplots(figsize=(8, 5))
        feature_importance.plot(kind='barh', color='#00d4ff', ax=ax)
        ax.set_title("Top Factors Driving Loan Approval", fontsize=14, color='white')
        ax.set_xlabel("Importance Score", color='#94a3b8')
        ax.tick_params(colors='#94a3b8')
        for spine in ax.spines.values():
            spine.set_edgecolor('#1e293b')
        st.pyplot(fig)
        
    with col_d2:
        st.subheader("🎯 Confusion Matrix")
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', ax=ax2, cbar=False,
                    xticklabels=['Rejected', 'Approved'], yticklabels=['Rejected', 'Approved'])
        ax2.set_title("Model Performance", fontsize=14, color='white')
        ax2.tick_params(colors='#94a3b8')
        for spine in ax2.spines.values():
            spine.set_edgecolor('#1e293b')
        st.pyplot(fig2)

    st.metric("📊 Overall Model Accuracy", f"{acc*100:.2f}%")

st.caption("🚀 Built with Streamlit, Random Forest & Feature Engineering | High-Contrast Dark UI")