# NeoLung: ML-Based Lung Cancer Risk Prediction System

NeoLung is an academic machine-learning project that explores lung cancer risk prediction using patient health and lifestyle information. It compares multiple classification algorithms and evaluates their performance using standard machine-learning metrics.

> **Disclaimer:** NeoLung is intended for academic and research purposes only. It is not a medical diagnostic tool and must not be used to make clinical decisions.

## Project Overview

NeoLung provides a complete machine-learning workflow:

- Loading and validating a CSV dataset
- Cleaning and preparing features
- Encoding the target variable
- Splitting data into training and testing sets
- Training multiple classification models
- Comparing model performance
- Generating evaluation reports and visualisations
- Saving trained models for future use

## Objectives

1. Develop a machine-learning pipeline for lung cancer risk prediction.
2. Prepare patient health and lifestyle data for analysis.
3. Train and compare different classification algorithms.
4. Evaluate models using multiple performance metrics.
5. Generate reusable trained-model files and result reports.
6. Provide a foundation for future healthcare-oriented machine-learning research.

## Machine Learning Models

The project evaluates:

1. Logistic Regression
2. K-Nearest Neighbors
3. Decision Tree
4. Support Vector Machine
5. Gaussian Naive Bayes
6. Random Forest
7. Gradient Boosting
8. Multi-Layer Perceptron Neural Network
9. AdaBoost
10. XGBoost

XGBoost is optional and will be included when the `xgboost` package is installed.

## Project Workflow

```text
Patient Dataset
       |
       v
Data Loading
       |
       v
Data Cleaning and Validation
       |
       v
Target Encoding
       |
       v
Feature Preprocessing
       |
       v
Train/Test Split
       |
       v
Model Training
       |
       v
Model Evaluation
       |
       v
Performance Comparison
       |
       v
Saved Models and Reports
```

## Evaluation Metrics

| Metric | Meaning |
|---|---|
| Accuracy | Percentage of correctly classified samples |
| Precision | Percentage of predicted positive samples that are actually positive |
| Recall | Percentage of actual positive samples correctly identified |
| F1-Score | Harmonic mean of precision and recall |
| ROC-AUC | Measures how well the model separates the two classes |

The program also generates confusion matrices and detailed classification reports.

## Technologies Used

- **Python**
- **Pandas** — data loading and manipulation
- **NumPy** — numerical operations
- **Matplotlib** — visualisations
- **Scikit-learn** — machine-learning algorithms and evaluation
- **Joblib** — saving trained models
- **XGBoost** — optional gradient-boosting classifier

## Project Structure

```text
NeoLung/
│
├── NeoLung_enhanced.py
├── requirements.txt
├── README.md
│
├── data/
│   └── survey_lung_cancer.csv
│
├── models/
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   ├── xgboost.joblib
│   └── ...
│
└── results/
    ├── model_comparison.csv
    ├── model_comparison.png
    ├── confusion_matrix_*.png
    └── classification_report_*.txt
```

The `models/` and `results/` folders are generated after running the program.

## Dataset

The program expects a CSV file named:

```text
survey_lung_cancer.csv
```

Place it inside the `data` folder:

```text
data/survey_lung_cancer.csv
```

The dataset should contain a target column such as:

```text
LUNG_CANCER
```

The target should represent two classes, such as `YES`/`NO` or `1`/`0`.

The remaining columns are treated as input features. Depending on the dataset, these may include health, lifestyle, demographic, and symptom-related attributes.

**Do not upload private, personally identifiable, or confidential patient information to a public repository.**

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/NeoLung.git
cd NeoLung
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Create a virtual environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## How to Run

Make sure the dataset is located at:

```text
data/survey_lung_cancer.csv
```

Then run:

```bash
python NeoLung_enhanced.py
```

The program will:

1. Load the dataset.
2. Clean and preprocess the data.
3. Split the data into training and testing sets.
4. Train the available machine-learning models.
5. Calculate evaluation metrics.
6. Save model-comparison results.
7. Generate confusion-matrix images.
8. Save detailed classification reports.
9. Save trained model pipelines.

## Generated Outputs

### Results folder

- `model_comparison.csv` — numerical comparison of all models
- `model_comparison.png` — visual comparison of model metrics
- `confusion_matrix_*.png` — confusion matrix for each model
- `classification_report_*.txt` — detailed classification report for each model

### Models folder

- Trained `.joblib` files for each available model
- `training_metadata.joblib` — training configuration and dataset information

## Reproducibility

The project uses:

- A fixed random state
- A defined test-set size
- Stratified train/test splitting
- Reusable preprocessing pipelines
- Saved model artifacts

Model performance may change if the dataset, preprocessing steps, or train/test split changes.

## Results

The script calculates the actual performance of each model during execution. Use the generated `results/model_comparison.csv` file when documenting the final results.

Do not assume that a model will always achieve a particular accuracy. Performance depends on the dataset and experimental setup.

## Limitations

- The project is based on structured survey-style data.
- The dataset may be limited in size and representativeness.
- Results may not generalise to different populations.
- The system has not been clinically validated.
- Predictions should not be interpreted as confirmed medical diagnoses.
- The project does not replace professional medical evaluation.

## Future Scope

Possible future improvements include:

- Using larger and more diverse datasets
- Integrating clinically validated datasets
- Adding explainable-AI techniques
- Performing cross-validation and hyperparameter tuning
- Testing external validation datasets
- Exploring medical-image analysis using CNNs
- Combining image features with structured clinical data
- Developing a carefully validated hybrid CNN-XGBoost architecture
- Improving the frontend and deployment workflow

## Academic Purpose

NeoLung was developed as an academic machine-learning project to understand how different classification algorithms can be applied to healthcare-related structured data.

The project demonstrates the complete machine-learning workflow, from dataset preparation to model evaluation and result generation.

## License

This project is intended for educational and academic use. Add an appropriate open-source license if you decide to distribute the project under specific licensing terms.
