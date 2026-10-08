# TB-Drug-Resistance-ML (TB-ResistAI)

## Machine Learning for Tuberculosis Drug-Resistance Prediction

TB-ResistAI is a machine-learning research project focused on predicting
tuberculosis (TB) drug resistance from genomic mutation features.

The project investigates how machine-learning models can learn
resistance patterns from genomic information and, importantly, evaluates
how well the developed models generalize to independent data from a
different dataset.

The workflow progresses from initial genomic feature analysis through
multiple-feature representation, model development, model comparison,
unseen-data evaluation, independent external validation, cross-dataset
evaluation, and generalization analysis.

> **Research prototype only:** This project is intended for research and
> educational purposes. Model predictions must not be used as a
> substitute for laboratory drug-susceptibility testing or clinical
> decision-making.

------------------------------------------------------------------------

## Research Objective

The primary objective is to investigate whether genomic mutation
features can be used to predict resistance to selected first-line
tuberculosis drugs using supervised machine-learning approaches.

A major focus of the project is **generalization**.

Rather than evaluating models only on data derived from the same dataset
used during development, the project includes evaluation on an
independent external dataset to examine whether the learned patterns
remain effective under a different data source.

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

This workflow separates model development from independent evaluation
and places particular emphasis on understanding the performance gap
between internal and external datasets.

------------------------------------------------------------------------

## Machine-Learning Models

Three supervised machine-learning approaches were investigated:

-   **Logistic Regression**
-   **Random Forest**
-   **XGBoost**

The models were evaluated using classification performance metrics and
compared during the model-development stage.

The final deployed model configuration was selected separately for each
drug based on the project's model-evaluation process.

------------------------------------------------------------------------

## Genomic Feature Approach

The project evolved from an initial **single-mutation representation**
toward an expanded representation using multiple genomic mutation
features.

This progression allowed the study to examine whether using a broader
genomic feature representation could provide a more useful signal for
predicting drug resistance.

The feature representation was kept consistent between model development
and external evaluation to ensure that the external validation
represented a genuine test of model generalization.

> The project does not make biological claims about individual mutations
> beyond what is supported by the underlying data and feature-processing
> pipeline.

------------------------------------------------------------------------

## Final Selected Models

  Drug             Final Model             Features   Decision Threshold
  ---------------- --------------------- ---------- --------------------
  **Rifampicin**   Random Forest                 30                 0.72
  **Isoniazid**    Logistic Regression            2                 0.50
  **Ethambutol**   Random Forest                  8                 0.32

The models stored in the repository are already fitted models. The
Streamlit application loads these models for inference and does not
retrain them during application use.

------------------------------------------------------------------------

## Independent External Validation

### Why External Validation Matters

A high score on an internal test set does not necessarily mean that a
machine-learning model will generalize well to data collected under
different conditions.

Therefore, an independent external dataset was used to evaluate the
generalization of the developed models.

The external dataset was treated as an **independent evaluation
dataset** and was not used for:

-   Model training
-   Hyperparameter tuning
-   Feature selection
-   Model selection

This separation is important because it provides a more realistic
assessment of how the developed models perform on previously unseen
data.

------------------------------------------------------------------------

## Final External Validation Results

  ------------------------------------------------------------------------------------------
  Drug                  ROC-AUC          95% CI   Accuracy   Precision     Recall   F1-score
  ---------------- ------------ --------------- ---------- ----------- ---------- ----------
  **Rifampicin**     **94.76%**   93.16--96.17%     93.20%      98.34%     87.18%     92.43%

  **Isoniazid**          87.91%   85.93--89.84%     87.90%      97.02%     78.20%     86.60%

  **Ethambutol**         87.10%   84.47--89.60%     85.40%      70.06%     86.11%     77.26%
  ------------------------------------------------------------------------------------------

### Rifampicin External ROC-AUC

The detailed external-results table records the Rifampicin ROC-AUC as:

``` text
94.758925%
```

which is reported here as:

``` text
94.76%
```

The reported 95% confidence interval is:

``` text
93.16% – 96.17%
```

------------------------------------------------------------------------

## Generalization Analysis

The external validation stage is a central component of TB-ResistAI.

The analysis compares model performance between the development/internal
evaluation setting and an independent external dataset.

This makes it possible to investigate:

-   Whether the learned genomic patterns transfer to unseen data
-   How performance changes across datasets
-   Differences in ROC-AUC and classification metrics
-   Potential generalization gaps
-   Drug-specific differences in predictive performance

The goal is not simply to maximize performance on a single dataset, but
to understand how robust the learned patterns are when evaluated on
independent data.

------------------------------------------------------------------------

## Prediction Application

TB-ResistAI includes a Streamlit-based prediction interface.

The application is designed around the finalized models rather than
asking the user to choose an arbitrary machine-learning algorithm.

The intended workflow is:

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

The application automatically uses the finalized model associated with
each drug:

``` text
Rifampicin  → Random Forest
Isoniazid   → Logistic Regression
Ethambutol  → Random Forest
```

The decision thresholds stored with the fitted models are used for the
corresponding classification step.

------------------------------------------------------------------------

## Important Interpretation of Model Metrics

### ROC-AUC

Measures the ability of a model to distinguish between resistant and
susceptible samples across classification thresholds.

### Precision

Measures the proportion of predicted resistant samples that are actually
resistant.

### Recall / Sensitivity

Measures the proportion of resistant samples correctly identified by the
model.

### F1-score

Provides a balance between precision and recall.

### Accuracy

Measures the proportion of correctly classified samples overall.

### External ROC-AUC

Measures discrimination performance when the model is evaluated on the
independent external dataset.

These evaluation metrics describe **model performance**. They are not
individual-patient treatment recommendations.

------------------------------------------------------------------------

## Project Structure

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

The project uses genomic and phenotypic data for model development and
external evaluation.

Raw/source datasets are not included in this public-ready repository
until their redistribution permissions have been verified.

The complete project backup containing the working data and intermediate
artifacts is maintained separately as the private master copy.

The notebooks document the major stages of preprocessing, feature
development, model training, model comparison, and external validation.

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

-   Python
-   Pandas
-   NumPy
-   Scikit-learn
-   XGBoost
-   Joblib
-   Streamlit
-   Matplotlib
-   Jupyter / Google Colab


------------------------------------------------------------------------

## Limitations

This project is a machine-learning research prototype and has several
important limitations:

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

## Disclaimer

> **TB-ResistAI is a research and educational machine-learning
> prototype.** It is not intended for clinical diagnosis, prescription,
> or treatment selection. Predictions should not be used to make
> clinical decisions or replace laboratory drug-susceptibility testing.

------------------------------------------------------------------------

## Project Status

**Research prototype --- external validation completed.**
