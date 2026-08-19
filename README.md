# Email Spam Detection Using Machine Learning

## Project Objective

This project classifies email text into two categories:

- `1` - Spam
- `0` - Ham / Legitimate

The system performs dataset validation, text cleaning, train/test splitting, TF-IDF feature extraction, model training, evaluation, comparison, best-model selection, model saving, and prediction through a Streamlit web application.

## Features

- Robust dataset validation
- Missing and empty text handling
- Duplicate email removal
- Invalid label detection
- Email, URL and HTML cleaning
- TF-IDF unigram and bigram features
- Multinomial Naive Bayes
- Logistic Regression
- Linear Support Vector Machine
- Accuracy, Precision, Recall and F1-score comparison
- Confusion matrix for every model
- Automatic best model selection using F1-score
- Streamlit web interface

## Technologies Used

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, joblib and Streamlit.

## Dataset Format

Create `data/emails.csv` with the following columns:

```csv
text,spam
"Congratulations! You won a prize. Click the link now!",1
"Hi, please attend the project meeting at 10 AM tomorrow.",0
```

The `spam` column must contain only `0` or `1`. The dataset must contain both classes and enough samples for an 80/20 stratified split.

## Project Structure

```text
email-spam-detection/
├── data/
│   └── emails.csv
├── models/
├── outputs/
├── train_model.py
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

The `models` and `outputs` folders are created automatically during training if they do not already exist.

## Installation

Open the project folder in VS Code and run:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If PowerShell blocks activation, use Command Prompt or adjust the execution policy according to your local Windows configuration.

## Train the Models

```bash
python train_model.py
```

The program uses:

```text
Dataset
  -> Cleaning
  -> Train/Test Split
  -> TF-IDF
  -> Train 3 Models
  -> Evaluate
  -> Compare Metrics
  -> Select Highest F1-score
  -> Save Best Pipeline
```

Stratification preserves the class distribution between the training and testing sets, which is especially important when spam and legitimate emails are imbalanced.

## Run Streamlit

After training:

```bash
streamlit run app.py
```

A browser page will open. Enter an email and select **Detect Spam**.

## How Prediction Works

The saved object is a scikit-learn Pipeline containing TF-IDF and the selected classifier. A new email is passed to `model.predict()`, so the same preprocessing and feature transformation used during training are applied consistently.

## Evaluation Metrics

- **Accuracy**: Overall proportion of correct predictions.
- **Precision**: Of emails predicted as spam, how many were actually spam.
- **Recall**: Of all actual spam emails, how many were detected.
- **F1-score**: Harmonic balance between precision and recall. This project uses F1-score to select the best model.

A confusion matrix contains:

- **TP**: Spam correctly predicted as spam.
- **TN**: Legitimate email correctly predicted as legitimate.
- **FP**: Legitimate email incorrectly predicted as spam.
- **FN**: Spam incorrectly predicted as legitimate.

Precision helps reduce false spam alerts, while recall helps reduce missed spam. F1-score balances both concerns.

## Generated Files

After successful training, the project creates:

```text
models/naive_bayes.pkl
models/logistic_regression.pkl
models/svm.pkl
models/best_model.pkl
outputs/model_results.csv
outputs/model_comparison.png
outputs/confusion_matrix_naive_bayes.png
outputs/confusion_matrix_logistic_regression.png
outputs/confusion_matrix_svm.png
```

Actual metric values depend on your dataset and are calculated at runtime. No experimental results are hard-coded.

## Testing the Application

Try legitimate work messages, promotional messages, urgent prize messages, URLs, numbers, and long email content. Empty input should show a validation warning.

## Possible Future Improvements

- Larger and more diverse dataset
- Hyperparameter tuning
- Cross-validation
- Email subject and sender features
- Character-level TF-IDF
- Explainable AI techniques
- User authentication and prediction history
- Deployment to a cloud platform

## Viva Summary

Spam detection automatically identifies unwanted or suspicious emails. Machine learning learns patterns from previously labelled examples. TF-IDF converts text into numerical features based on word importance. Three algorithms are trained so their actual performance can be compared on the same test data. The model with the highest F1-score is saved and used by Streamlit to classify new emails.
