# TB-ResistAI Model Export

Research prototype for predicting tuberculosis drug resistance.

Artifacts:
- RIF_model.joblib: Rifampicin Random Forest
- INH_model.joblib: Isoniazid Logistic Regression
- EMB_model.joblib: Ethambutol Random Forest
- model_manifest.json: model and feature metadata

Decision thresholds:
- RIF: 0.72
- INH: 0.50
- EMB: 0.32

Use the saved feature names and their exact order when preparing inputs.
These artifacts require compatible Python and ML-library versions.
This is a research prototype, not a clinical diagnostic tool.
