# Diabetes Detection

An end-to-end machine learning notebook that predicts whether a patient has diabetes from basic diagnostic measurements. It follows the **OSEMN** workflow (Obtain, Scrub, Explore, Model, iNterpret) and covers data cleaning, exploratory analysis, hypothesis testing, classification, clustering, and a brief PySpark exploration.

## Dataset

`diabetes.csv` is the Pima Indians Diabetes dataset: 768 patient records, 8 numeric features and a binary target.

| Column | Description |
|---|---|
| `Pregnancies` | Number of times pregnant |
| `Glucose` | Plasma glucose concentration |
| `BloodPressure` | Diastolic blood pressure (mm Hg) |
| `SkinThickness` | Triceps skin fold thickness (mm) |
| `Insulin` | Serum insulin (mu U/ml) |
| `BMI` | Body mass index |
| `DiabetesPedigreeFunction` | Diabetes pedigree (family history) score |
| `Age` | Age in years |
| `Outcome` | **Target**: `1` = diabetic, `0` = not diabetic |

Class balance: 500 non-diabetic (65%) and 268 diabetic (35%). There are no null values, but `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin` and `BMI` contain `0`s that are physiologically impossible and effectively represent missing data.

## What's in the notebook

1. **Data preprocessing**: inspection (`head`, `describe`, `info`, `dtypes`), null checks, and replacing invalid zeros with `NaN` to count them.
2. **Exploratory data analysis**: correlation heatmap, histograms, pair plots, regression and scatter plots, box plots, and class distribution.
3. **Outlier removal**: IQR (1.5 × IQR) rule, which removes 80+ records.
4. **Statistical hypothesis testing**: two-sample z-test, independent t-tests (glucose by outcome, via `researchpy` and `scipy`), a one-sample t-test on blood pressure, and D'Agostino normality tests for every feature.
5. **Logistic regression**: accuracy, precision, recall, F1, confusion matrix, ROC/AUC, and coefficient-based feature importance.
6. **K-Nearest Neighbors**: sweep of k = 1–14, final model with k = 11, confusion matrix, classification report, ROC/AUC, and decision-region plot.
7. **K-Means clustering**: elbow plots for Age/BMI, Pregnancies, Insulin/Glucose, and Glucose.
8. **XGBoost**: classifier with feature importances and prediction probabilities (used as the basis for a simple risk-recommendation approach).
9. **Apache Spark**: schema, summary statistics, class counts, missing-value check, and `VectorAssembler` feature preparation.

## Results

Figures noted in the notebook's own comments (your numbers will vary slightly between runs because the train/test split has no fixed seed):

| Model | Reported result |
|---|---|
| Logistic Regression | AUC ≈ 0.807 |
| K-Nearest Neighbors (k = 11) | Best test accuracy in the k sweep |
| XGBoost | Accuracy ≈ 0.78 |

## Getting started

### Requirements

- Python 3.8+
- Jupyter Notebook or JupyterLab
- Java 8/11+ (only needed for the PySpark section)

### Install

```bash
pip install numpy pandas matplotlib seaborn scipy statsmodels scikit-learn \
            xgboost mlxtend researchpy pyspark jupyter
```

### Run

1. Clone or download this repository.
2. Put `diabetes.csv` in a `data/` folder next to the notebook (the notebook reads `data/diabetes.csv`).
3. Launch Jupyter and run `DiabetesDetection.ipynb` top to bottom:

```bash
jupyter notebook DiabetesDetection.ipynb
```

## Project structure

```
.
├── DiabetesDetection.ipynb   # Analysis and modeling notebook
├── data/
│   └── diabetes.csv          # Dataset
└── README.md
```

## Known issues and tips

- **Spark path:** the PySpark section reads `diabetes.csv` from the working directory, not `data/`. Adjust the path or copy the file there.
- **Spark overwrites `df`:** that section reassigns `df` to a Spark DataFrame, so run it last.
- **scikit-learn version:** `plot_confusion_matrix` was removed in scikit-learn 1.2. Use `ConfusionMatrixDisplay.from_estimator` or pin an older version.
- **Seaborn / Matplotlib versions:** `sns.distplot`, `sns.lineplot(range, ...)` positional arguments, and the `seaborn-whitegrid` style are deprecated in newer releases.
- **Confusion-matrix helper:** the `tn` scorer function compares `y_test` with `y_train` instead of `y_pred`; it is not used in the final models but should be fixed before reuse.
- **Reproducibility:** add `random_state=42` to `train_test_split` and the models for repeatable results.
- **Missing values:** the invalid zeros are counted but not imputed before modeling. Imputing them (e.g. with the median) would likely improve results.

## Disclaimer

This project is for educational purposes only and is not a medical diagnostic tool.

## Acknowledgements

Dataset originally from the National Institute of Diabetes and Digestive and Kidney Diseases, available via the UCI / Kaggle Pima Indians Diabetes Database.
