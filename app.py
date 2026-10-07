
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

st.set_page_config(
    page_title="TB Resistance Predictor",
    page_icon="🧬",
    layout="wide"
)

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"

st.title("🧬 TB Antibiotic Resistance Prediction")
st.write(
    "Machine-learning interface for exploring "
    "tuberculosis antibiotic resistance."
)

st.info(
    "Research prototype only. Predictions are not "
    "a substitute for laboratory drug-susceptibility testing."
)

drug = st.selectbox(
    "Select antibiotic",
    ["Rifampicin", "Isoniazid", "Ethambutol"]
)

model_type = st.selectbox(
    "Select machine-learning model",
    ["Random Forest", "Logistic Regression", "XGBoost"]
)

slug = drug.lower()
model_slug = model_type.lower().replace(" ", "_")

model_path = MODEL_DIR / f"{slug}_{model_slug}.joblib"
features_path = MODEL_DIR / f"{slug}_features.json"

st.subheader("Mutation features")

if not features_path.exists():
    st.warning(
        f"Feature manifest not found: {features_path.name}. "
        "Export the exact feature list used during training."
    )
    st.stop()

with open(features_path, "r", encoding="utf-8") as file:
    features = json.load(file)

values = {}

with st.form("prediction_form"):
    st.write("Enter the mutation indicators for this isolate.")

    columns = st.columns(3)

    for i, feature in enumerate(features):
        with columns[i % 3]:
            values[feature] = st.selectbox(
                feature,
                options=[0, 1],
                format_func=lambda x: (
                    "Absent (0)" if x == 0 else "Present (1)"
                ),
                key=feature
            )

    submitted = st.form_submit_button("Predict resistance")

if submitted:
    if not model_path.exists():
        st.error(
            f"Model file not found: {model_path.name}. "
            "Save your trained model in the models folder."
        )
    else:
        model = joblib.load(model_path)

        input_df = pd.DataFrame(
            [[values[f] for f in features]],
            columns=features
        )

        try:
            prediction = model.predict(input_df)[0]

            st.subheader("Prediction result")
            st.write(f"**Antibiotic:** {drug}")
            st.write(f"**Model:** {model_type}")
            st.write(f"**Predicted class:** {prediction}")

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_df)[0]
                classes = model.classes_

                st.write("**Model probability estimates**")
                probability_df = pd.DataFrame({
                    "Class": classes,
                    "Probability (%)": probabilities * 100
                })
                st.dataframe(
                    probability_df,
                    use_container_width=True,
                    hide_index=True
                )

        except Exception as error:
            st.error(f"Prediction failed: {error}")
