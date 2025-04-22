# titanic_streamlit_app.py
import streamlit as st
import joblib
import numpy as np

# Load model and scaler
@st.cache_resource
def load_model():
    model = joblib.load("titanic_survival_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()

# Configure page
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stNumberInput, .stSelectbox {
        margin-bottom: 15px;
    }
    .prediction-header {
        color: #1f77b4;
        font-size: 24px !important;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.title("🚢 Titanic Survival Predictor")
st.markdown("Predict your chances of surviving the Titanic disaster based on passenger details!")

# Input sections
with st.form("passenger_details"):
    # Personal Information
    st.header("🧑 Passenger Details")
    col1, col2 = st.columns(2)
    with col1:
        pclass = st.selectbox("Ticket Class", [1, 2, 3], 
                            help="1 = 1st Class, 2 = 2nd Class, 3 = 3rd Class")
        age = st.number_input("Age", min_value=0.0, max_value=100.0, value=25.0,
                            step=1.0, format="%.1f")
        sex = st.selectbox("Gender", ["Female", "Male"])
        
    with col2:
        fare = st.number_input("Fare (£)", min_value=0.0, max_value=600.0, 
                             value=32.20 if pclass == 1 else 15.75 if pclass == 2 else 8.05,
                             step=1.0, format="%.2f",
                             help="Typical fares: 1st (£30-50), 2nd (£13-15), 3rd (£7-10)")
        title = st.selectbox("Title", ["Mr", "Mrs", "Miss", "Master", "Other"],
                           help="Social title from passenger name")

    # Family Information
    st.header("👨👩👧👦 Family Members")
    col3, col4 = st.columns(2)
    with col3:
        sibsp = st.number_input("Siblings/Spouses Aboard", 
                              min_value=0, max_value=10, value=0)
    with col4:
        parch = st.number_input("Parents/Children Aboard", 
                              min_value=0, max_value=10, value=0)

    # Embarkation Details
    st.header("⚓ Embarkation Information")
    embarked = st.selectbox("Port of Embarkation", 
                          ["Southampton (S)", "Cherbourg (C)", "Queenstown (Q)"],
                          index=0,
                          help="S = Southampton, C = Cherbourg, Q = Queenstown")

    # Prediction button
    submitted = st.form_submit_button("Predict Survival")

# Process inputs and predict
if submitted:
    try:
        # Validate inputs
        if fare <= 0:
            st.warning("⚠️ Fare must be greater than 0!")
            st.stop()
            
        if age > 100:
            st.warning("⚠️ Please enter a valid age (0-100)")
            st.stop()

        # Feature engineering
        family_size = sibsp + parch + 1
        
        # Encode categorical features
        sex_male = 1 if sex == "Male" else 0
        embarked_q = 1 if "Queenstown" in embarked else 0
        embarked_s = 1 if "Southampton" in embarked else 0
        
        # Encode title
        title_mr = 1 if title == "Mr" else 0
        title_mrs = 1 if title == "Mrs" else 0
        title_miss = 1 if title == "Miss" else 0
        title_master = 1 if title == "Master" else 0
        title_other = 1 if title == "Other" else 0

        # Create feature array
        features = np.array([[
            pclass, age, sibsp, parch, fare, sex_male, 
            embarked_q, embarked_s, title_mr, title_mrs, 
            title_miss, title_master, title_other
        ]])
        
        # Scale numerical features
        features[:, [1, 4]] = scaler.transform(features[:, [1, 4]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        # Display results
        st.markdown(f'<p class="prediction-header">Prediction Result</p>', 
                   unsafe_allow_html=True)
        
        if prediction == 1:
            st.success(f"✅ Survived (Probability: {probability:.1%})")
            st.markdown("Higher chances likely due to:")
            st.markdown("- 🚺 Female gender" if sex == "Female" else "")
            st.markdown("- 🎫 Higher class ticket" if pclass == 1 else "")
            st.markdown("- 👶 Younger age" if age < 18 else "")
        else:
            st.error(f"❌ Did Not Survive (Probability: {1-probability:.1%})")
            st.markdown("Lower chances likely due to:")
            st.markdown("- 🚹 Male gender" if sex == "Male" else "")
            st.markdown("- 🎟️ Lower class ticket" if pclass == 3 else "")
            st.markdown("- 👴 Older age" if age > 50 else "")

    except Exception as e:
        st.error(f"❌ Error in prediction: {str(e)}")

 
