import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Modern CSS Styling - High Contrast & Easy to Read
st.markdown("""
<style>
    /* Main background with medical gradient pattern */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #4facfe 75%, #00f2fe 100%);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Custom container for content */
    .main .block-container {
        padding: 2rem;
        background: rgba(255, 255, 255, 0.98);
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        max-width: 1400px;
        margin: 1rem auto;
    }
    
    /* Title styling - Ultra bold and visible */
    h1 {
        color: #1a202c !important;
        text-align: center;
        font-weight: 900 !important;
        padding: 1.5rem 0;
        font-size: 3rem !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
        letter-spacing: 1px;
    }
    
    /* Subheader styling */
    h2, h3 {
        color: #1a202c !important;
        font-weight: 900 !important;
        font-size: 1.5rem !important;
    }
    
    /* Input labels - Ultra bold and dark */
    .stSlider label, .stSelectbox label {
        color: #000000 !important;
        font-weight: 900 !important;
        font-size: 1.15rem !important;
        margin-bottom: 0.5rem !important;
        text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.5);
    }
    
    /* Slider styling - Vibrant color */
    .stSlider > div > div > div > div {
        background: #667eea !important;
    }
    
    .stSlider [data-baseweb="slider"] {
        background: #cbd5e0;
    }
    
    /* Select box styling - White background, clear border */
    .stSelectbox > div > div {
        background-color: white !important;
        border-radius: 8px !important;
        border: 3px solid #667eea !important;
        font-weight: 900 !important;
        color: #000000 !important;
        font-size: 1.1rem !important;
    }
    
    /* Button styling - Bold and visible */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        font-weight: 900 !important;
        font-size: 1.4rem !important;
        padding: 1.2rem 2rem !important;
        border-radius: 12px !important;
        border: none !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
        margin-top: 1.5rem !important;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.8) !important;
    }
    
    /* Result box styling */
    .result-box {
        padding: 3rem;
        border-radius: 12px;
        text-align: center;
        margin-top: 2rem;
        font-size: 2.2rem;
        font-weight: 900;
        animation: fadeIn 0.5s ease-in;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    
    .positive {
        background: #ff4757;
        color: white;
        box-shadow: 0 6px 20px rgba(255, 71, 87, 0.5);
        border: 4px solid #ee5a6f;
    }
    
    .negative {
        background: #2ed573;
        color: white;
        box-shadow: 0 6px 20px rgba(46, 213, 115, 0.5);
        border: 4px solid #26de81;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Info box - Clear and visible */
    .info-box {
        background: #fff3cd;
        padding: 1.8rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 6px solid #ffc107;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        color: #000000;
        font-weight: 900;
        font-size: 1.1rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Section title - Ultra bold and clear */
    .section-title {
        color: white !important;
        font-size: 1.4rem !important;
        font-weight: 900 !important;
        margin-bottom: 1rem;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 8px;
        text-align: center;
        letter-spacing: 1px;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        text-transform: uppercase;
    }
    
    /* Column backgrounds */
    div[data-testid="column"] {
        background: rgba(248, 249, 250, 0.95);
        padding: 1.5rem;
        border-radius: 12px;
        border: 3px solid #667eea;
        margin: 0.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    /* Slider value display - Bold */
    .stSlider > div > div > div > div[data-testid="stTickBar"] > div {
        color: #000000 !important;
        font-weight: 900 !important;
        font-size: 1.1rem !important;
    }
    
    /* Make all text bold and visible */
    p, span, div {
        font-weight: 700 !important;
    }
    
    /* Dropdown text */
    .stSelectbox div[data-baseweb="select"] span {
        font-weight: 900 !important;
        color: #000000 !important;
    }

</style>
""", unsafe_allow_html=True)

# Load dataset
@st.cache_data
def load_data():
    try:
        data = pd.read_csv("C:\\Users\\aaron\\Downloads\\clg project\\heart .csv")
        return data
    except FileNotFoundError:
        st.error("❌ Error: 'heart.csv' file not found!")
        st.info("Please ensure 'heart.csv' is in the same directory as app.py")
        st.stop()

data = load_data()
X = data.drop(columns=['target'])
y = data['target']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and save models
@st.cache_resource
def train_models():
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(probability=True),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(),
        "Logistic Regression": LogisticRegression(),
        "Naive Bayes": GaussianNB(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
    
    # ANN Model
    ann = Sequential([
        Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    ann.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    ann.fit(X_train, y_train, epochs=50, batch_size=10, verbose=0)
    
    models["Artificial Neural Network"] = ann
    return models

models = train_models()

# Streamlit UI
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="wide")

# Header
st.markdown("<h1>❤️ Heart Disease Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<div class='info-box'>🏥 <b>AI-Powered Cardiovascular Health Assessment</b><br>Enter your health parameters below to get an instant prediction using advanced machine learning algorithms.</div>", unsafe_allow_html=True)

# User input features
def user_input_features():
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<div class='section-title'>👤 Personal Information</div>", unsafe_allow_html=True)
        age = st.slider("Age", 20, 100, 50)
        sex = st.selectbox("Sex", ["Male", "Female"])
        
    with col2:
        st.markdown("<div class='section-title'>💓 Cardiac Indicators</div>", unsafe_allow_html=True)
        cp = st.slider("Chest Pain Type (0-3)", 0, 3, 1)
        trestbps = st.slider("Resting Blood Pressure", 80, 200, 120)
        chol = st.slider("Cholesterol Level", 100, 600, 250)
        thalach = st.slider("Max Heart Rate Achieved", 60, 220, 150)
        
    with col3:
        st.markdown("<div class='section-title'>🔬 Clinical Tests</div>", unsafe_allow_html=True)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
        restecg = st.slider("Resting ECG (0-2)", 0, 2, 1)
        exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
    
    col4, col5 = st.columns(2)
    
    with col4:
        st.markdown("<div class='section-title'>📊 Additional Metrics</div>", unsafe_allow_html=True)
        oldpeak = st.slider("ST Depression", 0.0, 6.2, 1.0)
        slope = st.slider("Slope of Peak Exercise", 0, 2, 1)
        
    with col5:
        st.markdown("<div class='section-title'>🩺 Advanced Parameters</div>", unsafe_allow_html=True)
        ca = st.slider("Number of Major Vessels (0-4)", 0, 4, 0)
        thal = st.slider("Thalassemia (0-3)", 0, 3, 2)
    
    sex = 1 if sex == "Male" else 0
    fbs = 1 if fbs == "Yes" else 0
    exang = 1 if exang == "Yes" else 0
    
    return np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

input_data = user_input_features()

# Model selection
st.markdown("<div class='section-title'>🤖 Select Prediction Model</div>", unsafe_allow_html=True)
model_type = st.selectbox("Choose your preferred machine learning algorithm", list(models.keys()))
model = models[model_type]

# Predict
if st.button("🔍 Predict Heart Disease Risk"):
    with st.spinner('Analyzing your health data...'):
        if model_type == "Artificial Neural Network":
            prediction = (model.predict(input_data) > 0.5).astype(int)
        else:
            prediction = model.predict(input_data)
        
        if prediction[0] == 1:
            st.markdown("""
                <div class='result-box positive'>
                    ⚠️ POSITIVE PREDICTION<br>
                    <span style='font-size: 1rem;'>Higher risk detected. Please consult a healthcare professional.</span>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class='result-box negative'>
                    ✅ NEGATIVE PREDICTION<br>
                    <span style='font-size: 1rem;'>Lower risk detected. Maintain a healthy lifestyle!</span>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<div style='text-align: center; margin-top: 1rem; color: #718096; font-size: 0.9rem;'>⚕️ <i>This is a screening tool only. Always consult with medical professionals for proper diagnosis.</i></div>", unsafe_allow_html=True)
