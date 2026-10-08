import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# 1. PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="TB-ResistAI",
    page_icon="🧬",
    layout="wide"
)


# ============================================================
# 2. PROJECT PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"


# ============================================================
# 3. MODEL LOCATIONS
# ============================================================

MODEL_PATHS = {
    "Rifampicin": MODEL_DIR / "rifampicin" / "model.joblib",
    "Isoniazid": MODEL_DIR / "isoniazid" / "model.joblib",
    "Ethambutol": MODEL_DIR / "ethambutol" / "model.joblib",
}


# ============================================================
# 4. LOAD SAVED MODEL BUNDLE
# ============================================================

@st.cache_resource
def load_model_bundle(drug):

    model_path = MODEL_PATHS[drug]

    if not model_path.exists():
        raise FileNotFoundError(
            f"{drug} model not found:\n{model_path}"
        )

    bundle = joblib.load(model_path)

    if not isinstance(bundle, dict):
        raise ValueError(
            f"{drug}: saved file is not a model bundle."
        )

    if "estimator" not in bundle:
        raise ValueError(
            f"{drug}: 'estimator' not found in saved model."
        )

    if "feature_names" not in bundle:
        raise ValueError(
            f"{drug}: 'feature_names' not found in saved model."
        )

    return bundle


# ============================================================
# 5. CREATE FEATURE VECTOR
# ============================================================

def create_feature_vector(selected_mutations, feature_names):

    selected = set(selected_mutations)

    values = []

    for feature in feature_names:

        if feature in selected:
            values.append(1)
        else:
            values.append(0)

    return np.array(values, dtype=np.int64).reshape(1, -1)


# ============================================================
# 6. GET RESISTANCE PROBABILITY
# ============================================================

def get_resistance_probability(bundle, X):

    estimator = bundle["estimator"]

    if not hasattr(estimator, "predict_proba"):

        raise ValueError(
            f"The saved estimator "
            f"{type(estimator).__name__} "
            f"does not support predict_proba()."
        )

    probabilities = estimator.predict_proba(X)[0]

    # Your saved bundle contains this value
    positive_column = bundle.get(
        "positive_probability_column",
        1
    )

    return float(probabilities[positive_column])


# ============================================================
# 7. HEADER
# ============================================================

st.title("🧬 TB-ResistAI")

st.subheader(
    "Genomic Mutation-Based Tuberculosis "
    "Drug-Resistance Prediction"
)

st.write(
    """
    Enter the detected mutation profile. The application
    automatically evaluates the selected mutation profile
    using the finalized model for each evaluated drug.
    """
)

st.info(
    """
    Research prototype only. Predictions are not a substitute
    for laboratory drug-susceptibility testing or clinical
    decision-making.
    """
)


# ============================================================
# 8. LOAD ALL THREE MODEL BUNDLES
# ============================================================

try:

    rif_bundle = load_model_bundle("Rifampicin")
    inh_bundle = load_model_bundle("Isoniazid")
    emb_bundle = load_model_bundle("Ethambutol")

except Exception as e:

    st.error(str(e))
    st.stop()


# ============================================================
# 9. GET EXACT FEATURES FROM SAVED MODELS
# ============================================================

rif_features = rif_bundle["feature_names"]
inh_features = inh_bundle["feature_names"]
emb_features = emb_bundle["feature_names"]


# ============================================================
# 10. COMBINE FEATURES FOR USER INPUT
# ============================================================

all_features = sorted(
    set(
        rif_features
        + inh_features
        + emb_features
    )
)


# ============================================================
# 11. MODEL INFORMATION
# ============================================================

st.header("🤖 Final Model Configuration")

model_information = []

for drug, bundle in [
    ("Rifampicin", rif_bundle),
    ("Isoniazid", inh_bundle),
    ("Ethambutol", emb_bundle)
]:

    model_information.append({

        "Drug": drug,

        "Final Model": bundle.get(
            "model_name",
            type(bundle["estimator"]).__name__
        ),

        "Features": len(
            bundle["feature_names"]
        ),

        "Decision Threshold": float(
            bundle["threshold"]
        )

    })


model_table = pd.DataFrame(
    model_information
)

st.dataframe(
    model_table,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# 12. MUTATION INPUT
# ============================================================

st.header("🧬 Mutation Input")

st.write(
    f"""
    Select the mutations detected in the sample.

    Available model features: {len(all_features)}
    """
)

selected_mutations = st.multiselect(
    "Select detected mutations",
    options=all_features,
    help="Select the mutations detected in the sample."
)


# ============================================================
# 13. SHOW SELECTED MUTATIONS
# ============================================================

if selected_mutations:

    st.success(
        f"{len(selected_mutations)} mutation(s) selected."
    )

    with st.expander("View selected mutations"):

        for mutation in selected_mutations:

            st.write(
                f"✓ {mutation}"
            )

else:

    st.warning(
        "Select at least one mutation before analysis."
    )


# ============================================================
# 14. ANALYZE BUTTON
# ============================================================

analyze = st.button(
    "🔬 Analyze Mutation Profile",
    type="primary",
    use_container_width=True
)


# ============================================================
# 15. RUN PREDICTIONS
# ============================================================

if analyze:

    if not selected_mutations:

        st.error(
            "Please select at least one mutation."
        )

        st.stop()


    results = []


    drug_bundles = [
        ("Rifampicin", rif_bundle),
        ("Isoniazid", inh_bundle),
        ("Ethambutol", emb_bundle)
    ]


    for drug, bundle in drug_bundles:

        try:

            # ------------------------------------------------
            # Exact estimator saved during training
            # ------------------------------------------------

            estimator = bundle["estimator"]


            # ------------------------------------------------
            # Exact feature order used during training
            # ------------------------------------------------

            feature_names = bundle["feature_names"]


            # ------------------------------------------------
            # Exact threshold selected during research
            # ------------------------------------------------

            threshold = float(
                bundle["threshold"]
            )


            # ------------------------------------------------
            # Create model input
            # ------------------------------------------------

            X = create_feature_vector(
                selected_mutations,
                feature_names
            )


            # ------------------------------------------------
            # Safety check
            # ------------------------------------------------

            expected_features = len(
                feature_names
            )

            actual_features = X.shape[1]

            if expected_features != actual_features:

                raise ValueError(
                    f"Feature mismatch. "
                    f"Expected {expected_features}, "
                    f"got {actual_features}."
                )


            # ------------------------------------------------
            # Probability
            # ------------------------------------------------

            probability = (
                get_resistance_probability(
                    bundle,
                    X
                )
            )


            # ------------------------------------------------
            # Prediction using saved threshold
            # ------------------------------------------------

            prediction = (

                "Resistant"

                if probability >= threshold

                else "Predicted susceptible"

            )


            # ------------------------------------------------
            # Model name
            # ------------------------------------------------

            model_name = bundle.get(
                "model_name",
                type(estimator).__name__
            )


            results.append({

                "Drug": drug,

                "Model": model_name,

                "Resistance Probability":
                    probability,

                "Threshold":
                    threshold,

                "Prediction":
                    prediction

            })


        except Exception as e:

            st.error(
                f"{drug}: {str(e)}"
            )


    # ========================================================
    # 16. DISPLAY RESULTS
    # ========================================================

    if results:

        result_df = pd.DataFrame(
            results
        )


        st.header(
            "📊 Prediction Summary"
        )


        display_df = result_df.copy()


        display_df[
            "Resistance Probability"
        ] = (

            display_df[
                "Resistance Probability"
            ] * 100

        ).round(2).astype(str) + "%"


        display_df[
            "Threshold"
        ] = (

            display_df[
                "Threshold"
            ] * 100

        ).round(0).astype(int).astype(str) + "%"


        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )


        # ====================================================
        # 17. INDIVIDUAL DRUG RESULTS
        # ====================================================

        st.subheader(
            "Drug-wise Prediction"
        )


        columns = st.columns(3)


        for column, (_, row) in zip(
            columns,
            result_df.iterrows()
        ):

            with column:

                st.markdown(
                    f"### {row['Drug']}"
                )

                st.metric(
                    "Resistance probability",
                    f"{row['Resistance Probability'] * 100:.1f}%"
                )

                st.write(
                    f"**Model:** {row['Model']}"
                )

                st.write(
                    f"**Threshold:** "
                    f"{row['Threshold']:.2f}"
                )

                if row["Prediction"] == "Resistant":

                    st.error(
                        "Predicted resistant"
                    )

                else:

                    st.success(
                        "Predicted susceptible"
                    )


        # ====================================================
        # 18. COMPARISON
        # ====================================================

        st.subheader(
            "📈 Resistance Probability Comparison"
        )


        chart_df = result_df[
            [
                "Drug",
                "Resistance Probability"
            ]
        ].copy()


        chart_df = chart_df.set_index(
            "Drug"
        )


        chart_df[
            "Resistance Probability"
        ] *= 100


        st.bar_chart(
            chart_df
        )


        # ====================================================
        # 19. LOWEST PREDICTED RESISTANCE
        # ====================================================

        lowest = result_df.loc[
            result_df[
                "Resistance Probability"
            ].idxmin()
        ]


        st.subheader(
            "🔎 Model Interpretation"
        )


        st.write(
            f"""
            Among the evaluated drugs, the model predicts the
            lowest resistance probability for **{lowest["Drug"]}**
            ({lowest["Resistance Probability"] * 100:.1f}%).
            """
        )


        st.warning(
            """
            This comparison is a model-based research result,
            not a recommendation for clinical treatment.
            Laboratory susceptibility testing and clinical
            assessment are required for treatment decisions.
            """ 
        )


# ============================================================
# 20. VALIDATION INFORMATION
# ============================================================

st.header(
    "📈 Model Validation"
)

st.write(
    """
    The final models were selected based on model evaluation
    and external/generalization analysis.
    """
)


validation_table = pd.DataFrame({

    "Drug": [
        "Rifampicin",
        "Isoniazid",
        "Ethambutol"
    ],

    "Selected Model": [

        rif_bundle.get(
            "model_name",
            type(
                rif_bundle["estimator"]
            ).__name__
        ),

        inh_bundle.get(
            "model_name",
            type(
                inh_bundle["estimator"]
            ).__name__
        ),

        emb_bundle.get(
            "model_name",
            type(
                emb_bundle["estimator"]
            ).__name__
        )
    ],

    "Feature Count": [

        len(
            rif_bundle["feature_names"]
        ),

        len(
            inh_bundle["feature_names"]
        ),

        len(
            emb_bundle["feature_names"]
        )
    ],

    "Decision Threshold": [

        rif_bundle["threshold"],
        inh_bundle["threshold"],
        emb_bundle["threshold"]
    ]

})


st.dataframe(
    validation_table,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# 21. METRIC EXPLANATION
# ============================================================

with st.expander(
    "📚 What do ROC-AUC, F1, sensitivity and specificity mean?"
):

    st.markdown(
        """
        **ROC-AUC** measures how well the model separates
        resistant and susceptible samples across thresholds.

        **F1-score** balances precision and recall.

        **Sensitivity** measures how well resistant samples
        are identified.

        **Specificity** measures how well susceptible samples
        are identified.

        **External validation** evaluates model performance
        on data independent from model development.

        **Generalization gap** describes the difference between
        internal and external performance.

        These metrics describe model performance. They are not
        individual-patient treatment recommendations.
        """
    )


# ============================================================
# 22. DISCLAIMER
# ============================================================

st.divider()

st.caption(
    """
    TB-ResistAI is a research prototype for exploring
    mutation-based tuberculosis drug-resistance prediction.

    It is not intended for clinical diagnosis,
    prescription, or treatment selection.
    """
)
