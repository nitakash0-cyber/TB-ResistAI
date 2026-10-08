# TB-Drug-Resistance-ML (TB-ResistAI)

## Machine Learning for Tuberculosis Drug-Resistance Prediction

TB-ResistAI is a machine-learning research project focused on predicting
tuberculosis (TB) drug resistance from genomic mutation features.

The project investigates how machine-learning models can learn
resistance patterns from genomic information and, importantly, evaluates
how well the developed models generalize to an independent external
dataset.

------------------------------------------------------------------------

## Research Objective

The primary objective is to investigate whether genomic mutation
features can be used to predict resistance to selected first-line
tuberculosis drugs using supervised machine-learning approaches.

A major focus of the project is **generalization**: evaluating whether
patterns learned during model development remain effective when tested
on an independent external dataset.

------------------------------------------------------------------------

## Research / Machine-Learning Workflow

``` text
Raw Data
    ↓
Preprocessing & Cleaning
    ↓
Initial Genomic Feature Setup
    ↓
Multiple Genomic Features
    ↓
ML Model Development
    ↓
Model Comparison
    ↓
Unseen Data Evaluation
    ↓
Independent External Dataset
    ↓
Cross-Dataset Evaluation
    ↓
Generalization Analysis
```

------------------------------------------------------------------------

## Machine-Learning Models

The project investigates three supervised machine-learning approaches:

  -----------------------------------------------------------------------
  Model                               Role
  ----------------------------------- -----------------------------------
  Logistic Regression                 Linear classification baseline and
                                      final model for Isoniazid

  Random Forest                       Ensemble classification model and
                                      final model for Rifampicin and
                                      Ethambutol

  XGBoost                             Gradient-boosted tree model
                                      evaluated during model comparison
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## Genomic Feature Approach

The project progressed from an initial **single-mutation
representation** toward an expanded representation using multiple
genomic mutation features.

This progression allowed the study to evaluate whether a broader genomic
feature representation could provide a more useful predictive signal.

The feature representation was kept consistent between model development
and external evaluation so that external testing represented an
independent assessment of model generalization.

> The project does not make biological claims about individual mutations
> beyond what is supported by the underlying data and feature-processing
> pipeline.

------------------------------------------------------------------------

## Final Selected Models
Drug	Model	Features	Threshold
Rifampicin	Random Forest	30	0.72
Isoniazid	Logistic Regression	2	0.50
Ethambutol	Random Forest	8	0.32
The final fitted models are loaded by the Streamlit application for
inference. The application does not retrain the models.

------------------------------------------------------------------------

# Independent External Validation

## Why External Validation Matters

High performance on an internal test set does not necessarily mean that
a model will generalize to data collected from a different source.

Therefore, an independent external dataset was used to evaluate
generalization.

The external dataset was treated as an **independent evaluation
dataset** and was not used for:

-   Model training
-   Hyperparameter tuning
-   Feature selection
-   Model selection

This separation provides a more realistic assessment of performance on
previously unseen data.

------------------------------------------------------------------------

## Final External Validation Results

  ------------------------------------------------------------------------------------------
  Drug	ROC-AUC	95% CI	Accuracy	Precision	Recall	F1
Rifampicin	94.76%	93.16–96.17%	93.20%	98.34%	87.18%	92.43%
Isoniazid	87.91%	85.93–89.84%	87.90%	97.02%	78.20%	86.60%
Ethambutol	87.10%	84.47–89.60%	85.40%	70.06%	86.11%	77.26%
  ------------------------------------------------------------------------------------------

### Rifampicin External ROC-AUC

The detailed external-results table records the Rifampicin ROC-AUC as
**94.758925%**, which is reported above as **94.76%**.

The reported 95% confidence interval is **93.16%--96.17%**.

------------------------------------------------------------------------

## Generalization Analysis

External validation is a central component of TB-ResistAI.

The analysis compares model performance between the development/internal
evaluation setting and an independent external dataset.

The analysis investigates:

-   Transfer of learned genomic patterns to unseen data
-   Performance changes across datasets
-   Differences in ROC-AUC and classification metrics
-   Potential generalization gaps
-   Drug-specific differences in predictive performance

The objective is not simply to maximize performance on one dataset, but
to evaluate how robust the learned patterns are on independent data.

------------------------------------------------------------------------

# Prediction Application

TB-ResistAI includes a Streamlit-based prediction interface.

The application is designed around the finalized models rather than
allowing users to arbitrarily select a machine-learning algorithm.

### Application workflow

``` text
User enters detected mutation profile
                ↓
Mutation features are converted
to the model's training representation
                ↓
Final model for each drug is loaded automatically
                ↓
Resistance probability is calculated
                ↓
Drug-wise predictions are compared
                ↓
Model-based results are displayed
```

### Automatic model selection

  Drug         Automatically Used Model     Feature Count   Threshold
  ------------ -------------------------- --------------- -----------
  Rifampicin   Random Forest                           30        0.72
  Isoniazid    Logistic Regression                      2        0.50
  Ethambutol   Random Forest                            8        0.32

The user does **not** select the machine-learning algorithm. The
application uses the finalized model associated with each drug.

------------------------------------------------------------------------

## Interpretation of Model Metrics

  -----------------------------------------------------------------------
  Metric                              Meaning
  ----------------------------------- -----------------------------------
  **ROC-AUC**                         Measures the ability to distinguish
                                      resistant and susceptible samples
                                      across classification thresholds

  **Accuracy**                        Proportion of samples classified
                                      correctly overall

  **Precision**                       Proportion of predicted resistant
                                      samples that are actually resistant

  **Recall / Sensitivity**            Proportion of resistant samples
                                      correctly identified

  **F1-score**                        Harmonic balance between precision
                                      and recall

  **External ROC-AUC**                Discrimination performance on the
                                      independent external dataset
  -----------------------------------------------------------------------

These metrics describe **model performance**. They are not
individual-patient treatment recommendations.

------------------------------------------------------------------------

# Project Structure

``` text
TB-Drug-Resistance-ML/
│
├── README.md
├── requirements.txt
├── app.py
│
├── models/
│   ├── rifampicin/
│   │   └── model.joblib
│   │
│   ├── isoniazid/
│   │   └── model.joblib
│   │
│   └── ethambutol/
│       └── model.joblib
│
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_initial_single_mutation.ipynb
│   ├── 03_multiple_genomic_features.ipynb
│   ├── 04_model_training.ipynb
│   ├── 05_model_comparison.ipynb
│   └── 06_external_validation.ipynb
│
├── results/
│   ├── internal/
│   ├── external/
│   └── figures/
│
├── src/
│   ├── preprocessing/
│   ├── feature_engineering/
│   ├── modeling/
│   └── evaluation/
│
└── data/
    └── README.md
```

------------------------------------------------------------------------

## Reproducibility and Data

Raw/source datasets are kept out of this public-ready repository until
redistribution permissions are verified.

The complete working backup containing raw data and intermediate
artifacts is maintained separately as the private master copy.

The notebooks document the major stages of preprocessing, feature
development, model training, model comparison, and external validation.

The public repository contains the code, fitted model artifacts,
documentation, and appropriate results needed to understand the project
without exposing restricted source data.

------------------------------------------------------------------------

## Running the Application

### 1. Clone the repository

``` bash
git clone <YOUR_REPOSITORY_URL>
cd TB-Drug-Resistance-ML
```

### 2. Install dependencies

``` bash
pip install -r requirements.txt
```

### 3. Run the Streamlit application

``` bash
streamlit run app.py
```

The application loads the already-fitted models stored in the `models/`
directory.

It does **not** retrain the models when the application is launched.

------------------------------------------------------------------------

## Technologies Used

  Technology               Purpose
  ------------------------ ----------------------------------------
  Python                   Core programming language
  Pandas                   Data processing
  NumPy                    Numerical computation
  Scikit-learn             Machine-learning models and evaluation
  XGBoost                  Gradient-boosted machine learning
  Joblib                   Saving and loading fitted models
  Streamlit                Interactive prediction application
  Matplotlib               Visualization
  Jupyter / Google Colab   Research and experimentation



------------------------------------------------------------------------

# Limitations

-   Model performance may vary across datasets and populations.
-   External validation performance does not guarantee clinical
    performance.
-   Genomic feature representation is dependent on the available data
    and preprocessing pipeline.
-   The current models should not be interpreted as replacements for
    laboratory drug-susceptibility testing.
-   Further validation on additional independent datasets would be
    required before any clinical application could be considered.

------------------------------------------------------------------------

# Disclaimer

> **TB-ResistAI is a research and educational machine-learning
> prototype.** It is not intended for clinical diagnosis, prescription,
> or treatment selection. Predictions should not be used to make
> clinical decisions or replace laboratory drug-susceptibility testing.

------------------------------------------------------------------------

# Project Status

**Research prototype --- external validation completed.**
