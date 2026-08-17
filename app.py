import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Lung Cancer Prediction",
    page_icon="🫁",
    layout="centered"
)

# Load trained model
model_data = joblib.load("lung_cancer_model.pkl")
model = model_data["model"]
features = model_data["features"]

st.title("🫁 Lung Cancer Prediction System")

st.write(
    "Enter the patient's information below "
    "to get a machine-learning prediction."
)

st.warning(
    "⚠️ This application is for educational purposes only "
    "and is not a medical diagnosis."
)

st.divider()

st.subheader("👤 Patient Information")

# Age and Gender
col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=40
    )

with col2:
    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

# Smoking and Yellow Fingers
col1, col2 = st.columns(2)

with col1:
    smoking = st.selectbox(
        "Smoking",
        ["No", "Yes"]
    )

with col2:
    yellow_fingers = st.selectbox(
        "Yellow Fingers",
        ["No", "Yes"]
    )

# Anxiety and Peer Pressure
col1, col2 = st.columns(2)

with col1:
    anxiety = st.selectbox(
        "Anxiety",
        ["No", "Yes"]
    )

with col2:
    peer_pressure = st.selectbox(
        "Peer Pressure",
        ["No", "Yes"]
    )

# Chronic Disease and Fatigue
col1, col2 = st.columns(2)

with col1:
    chronic_disease = st.selectbox(
        "Chronic Disease",
        ["No", "Yes"]
    )

with col2:
    fatigue = st.selectbox(
        "Fatigue",
        ["No", "Yes"]
    )

# Allergy and Wheezing
col1, col2 = st.columns(2)

with col1:
    allergy = st.selectbox(
        "Allergy",
        ["No", "Yes"]
    )

with col2:
    wheezing = st.selectbox(
        "Wheezing",
        ["No", "Yes"]
    )

# Alcohol and Coughing
col1, col2 = st.columns(2)

with col1:
    alcohol_consuming = st.selectbox(
        "Alcohol Consuming",
        ["No", "Yes"]
    )

with col2:
    coughing = st.selectbox(
        "Coughing",
        ["No", "Yes"]
    )

# Breathing and Swallowing
col1, col2 = st.columns(2)

with col1:
    shortness_of_breath = st.selectbox(
        "Shortness of Breath",
        ["No", "Yes"]
    )

with col2:
    swallowing_difficulty = st.selectbox(
        "Swallowing Difficulty",
        ["No", "Yes"]
    )

# Chest Pain
chest_pain = st.selectbox(
    "Chest Pain",
    ["No", "Yes"]
)

st.divider()

# Prediction
if st.button(
    "🔍 Predict Lung Cancer",
    use_container_width=True
):

    # Convert inputs to numbers
    gender_value = 1 if gender == "Male" else 0

    smoking_value = 1 if smoking == "Yes" else 0
    yellow_fingers_value = 1 if yellow_fingers == "Yes" else 0
    anxiety_value = 1 if anxiety == "Yes" else 0
    peer_pressure_value = 1 if peer_pressure == "Yes" else 0
    chronic_disease_value = 1 if chronic_disease == "Yes" else 0
    fatigue_value = 1 if fatigue == "Yes" else 0
    allergy_value = 1 if allergy == "Yes" else 0
    wheezing_value = 1 if wheezing == "Yes" else 0
    alcohol_value = 1 if alcohol_consuming == "Yes" else 0
    coughing_value = 1 if coughing == "Yes" else 0
    breathing_value = 1 if shortness_of_breath == "Yes" else 0
    swallowing_value = 1 if swallowing_difficulty == "Yes" else 0
    chest_pain_value = 1 if chest_pain == "Yes" else 0

    # Create input
    input_data = {
        "GENDER": gender_value,
        "AGE": age,
        "SMOKING": smoking_value,
        "YELLOW_FINGERS": yellow_fingers_value,
        "ANXIETY": anxiety_value,
        "PEER_PRESSURE": peer_pressure_value,
        "CHRONIC_DISEASE": chronic_disease_value,
        "FATIGUE": fatigue_value,
        "ALLERGY": allergy_value,
        "WHEEZING": wheezing_value,
        "ALCOHOL_CONSUMING": alcohol_value,
        "COUGHING": coughing_value,
        "SHORTNESS_OF_BREATH": breathing_value,
        "SWALLOWING_DIFFICULTY": swallowing_value,
        "CHEST_PAIN": chest_pain_value
    }

    input_df = pd.DataFrame([input_data])

    # Make prediction
    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0]

    confidence = max(probability) * 100

    st.subheader("📊 Prediction Result")

    if prediction == "YES":
        st.error("🔴 Lung Cancer Risk Detected")
    else:
        st.success("🟢 Lung Cancer Risk Not Detected")

    st.info(
        f"Model Confidence: {confidence:.2f}%"
    )

    with st.expander("View Submitted Information"):
        st.dataframe(input_df)