import streamlit as st
import pandas as pd
import joblib


st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide"
)


model = joblib.load("logistic_heart_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("x_columns.pkl")

if hasattr(columns, "tolist"):
    columns = columns.tolist()


st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(
        135deg,
        #0f172a,
        #1e293b,
        #334155
    );
}

/* Heart Animation */
@keyframes heartbeat {
    0% {transform: scale(1);}
    25% {transform: scale(1.1);}
    50% {transform: scale(1);}
    75% {transform: scale(1.1);}
    100% {transform: scale(1);}
}

/* Title */
.title {
    text-align:center;
    color:#ff4d6d;
    font-size:48px;
    font-weight:bold;
    animation:heartbeat 1.5s infinite;
}

/* Glass Card */
.card {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
    padding: 25px;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.2);
    margin-top: 15px;
}

/* Button */
.stButton > button {
    width:100%;
    background:linear-gradient(90deg,#ff416c,#ff4b2b);
    color:white;
    font-size:20px;
    font-weight:bold;
    border:none;
    border-radius:12px;
    padding:12px;
    transition:0.3s;
}

.stButton > button:hover {
    transform:scale(1.05);
    box-shadow:0px 0px 20px rgba(255,75,75,0.8);
}

/* Labels */
label {
    color:white !important;
    font-weight:bold !important;
}

/* Hide Streamlit Header */
header {
    visibility:hidden;
}

/* Footer */
footer {
    visibility:hidden;
}

</style>
""", unsafe_allow_html=True)


st.markdown(
    '<div class="title"> Heart Disease Prediction System </div>',
    unsafe_allow_html=True
)

st.image(
    r"C:\Users\ASUS\OneDrive\Desktop\ML\unsupervised\ChatGPT Image May 31, 2026, 12_54_42 PM.png",
    use_container_width=True
)

st.markdown("<br>", unsafe_allow_html=True)


st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("🩺 Enter Patient Details")

col1, col2 = st.columns(2)

with col1:

    age = st.slider("Age", 15, 100, 30)

    sex = st.selectbox(
        "Sex",
        ["Male", "Female"]
    )

    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["ATA", "NAP", "TA", "ASY"]
    )

    resting_bp = st.number_input(
        "Resting Blood Pressure (mm Hg)",
        min_value=80,
        max_value=250,
        value=120
    )

    cholesterol = st.number_input(
        "Cholesterol (mg/dL)",
        min_value=0,
        max_value=700,
        value=200
    )

with col2:

    fasting_bs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dL",
        [0, 1]
    )

    resting_ecg = st.selectbox(
        "Resting ECG",
        ["Normal", "ST", "LVH"]
    )

    max_hr = st.slider(
        "Maximum Heart Rate",
        60,
        220,
        150
    )

    exercise_angina = st.selectbox(
        "Exercise-Induced Angina",
        ["Y", "N"]
    )

    oldpeak = st.slider(
        "Oldpeak (ST Depression)",
        0.0,
        6.0,
        1.0
    )

    st_slope = st.selectbox(
        "ST Slope",
        ["Up", "Flat", "Down"]
    )

st.markdown('</div>', unsafe_allow_html=True)


if st.button("🔍 Predict Heart Disease Risk"):

    try:

        input_df = pd.DataFrame(
            [[0] * len(columns)],
            columns=columns
        )

        # Numerical Features
        if "Age" in columns:
            input_df.loc[0, "Age"] = age

        if "RestingBP" in columns:
            input_df.loc[0, "RestingBP"] = resting_bp

        if "Cholesterol" in columns:
            input_df.loc[0, "Cholesterol"] = cholesterol

        if "FastingBS" in columns:
            input_df.loc[0, "FastingBS"] = fasting_bs

        if "MaxHR" in columns:
            input_df.loc[0, "MaxHR"] = max_hr

        if "Oldpeak" in columns:
            input_df.loc[0, "Oldpeak"] = oldpeak

        sex_col = f"Sex_{sex}"
        if sex_col in columns:
            input_df.loc[0, sex_col] = 1

        cp_col = f"ChestPainType_{chest_pain}"
        if cp_col in columns:
            input_df.loc[0, cp_col] = 1

        ecg_col = f"RestingECG_{resting_ecg}"
        if ecg_col in columns:
            input_df.loc[0, ecg_col] = 1

        angina_col = f"ExerciseAngina_{exercise_angina}"
        if angina_col in columns:
            input_df.loc[0, angina_col] = 1

        slope_col = f"ST_Slope_{st_slope}"
        if slope_col in columns:
            input_df.loc[0, slope_col] = 1

        scaled_input = scaler.transform(input_df)

        prediction = model.predict(scaled_input)[0]

        probability = model.predict_proba(
            scaled_input
        )[0][1]

        st.markdown("---")

        st.subheader("📊 Prediction Result")

        if prediction == 1:

            st.error(
                "⚠️ High Risk of Heart Disease"
            )

            st.progress(float(probability))

            st.markdown(
                f"""
                <div style="
                background:#7f1d1d;
                padding:20px;
                border-radius:15px;
                color:white;
                text-align:center;
                font-size:24px;">
                ❤️ Risk Probability: <b>{probability:.2%}</b>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.success(
                "✅ Low Risk of Heart Disease"
            )

            st.progress(float(probability))

            st.markdown(
                f"""
                <div style="
                background:#14532d;
                padding:20px;
                border-radius:15px;
                color:white;
                text-align:center;
                font-size:24px;">
                💚 Risk Probability: <b>{probability:.2%}</b>
                </div>
                """,
                unsafe_allow_html=True
            )

    except Exception as e:
        st.error(f"Prediction Error: {e}")

st.markdown("---")
st.markdown(
    "<center><h4 style='color:white;'>❤️ Powered by Machine Learning & Streamlit ❤️</h4></center>",
    unsafe_allow_html=True
)