import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "emails.csv"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
RANDOM_STATE = 42


def clean_text(text):
    if pd.isna(text):
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r"\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^\w\s$€£%!?]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\nCreate data/emails.csv with columns: text, spam"
        )
    try:
        df = pd.read_csv(DATA_PATH)
    except Exception as exc:
        raise ValueError(f"Unable to read dataset: {exc}") from exc

    required = {"text", "spam"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset columns are {list(df.columns)}. Missing required columns: {sorted(missing)}. "
            "Expected columns: text, spam"
        )
    if df.empty:
        raise ValueError("Dataset is empty. Add valid email samples before training.")

    df = df[["text", "spam"]].copy()
    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.len() > 0].copy()
    df = df.drop_duplicates(subset=["text"])

    df["spam"] = pd.to_numeric(df["spam"], errors="coerce")
    invalid = df[~df["spam"].isin([0, 1])]
    if not invalid.empty:
        raise ValueError(
            f"Found {len(invalid)} invalid labels. The spam column must contain only 0 (ham) or 1 (spam)."
        )
    df["spam"] = df["spam"].astype(int)

    if df.empty:
        raise ValueError("No valid rows remain after cleaning the dataset.")
    if df["spam"].nunique() != 2:
        raise ValueError("Dataset must contain both classes: 0 (ham) and 1 (spam).")
    class_counts = df["spam"].value_counts()
    if class_counts.min() < 2 or len(df) < 10:
        raise ValueError(
            "Insufficient samples for a stratified train/test split. Use at least 10 rows and at least 2 samples per class."
        )
    return df


def create_pipeline(classifier):
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            preprocessor=clean_text,
            lowercase=False,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.98,
            sublinear_tf=True
        )),
        ("classifier", classifier)
    ])


def save_confusion_matrix(y_true, y_pred, model_key, model_name):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / f"confusion_matrix_{model_key}.png", dpi=300)
    plt.close()


def save_comparison_chart(results_df):
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    plot_df = results_df.set_index("Model")[metrics]
    ax = plot_df.plot(kind="bar", figsize=(10, 6))
    ax.set_title("Model Performance Comparison")
    ax.set_xlabel("Machine Learning Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    plt.xticks(rotation=0)
    plt.legend(title="Metric")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "model_comparison.png", dpi=300)
    plt.close()


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)

    print("Loading and validating dataset...")
    df = load_dataset()
    print(f"Valid samples: {len(df)}")
    print("Class distribution:")
    print(df["spam"].value_counts().sort_index())

    X = df["text"]
    y = df["spam"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    models = {
        "naive_bayes": ("Multinomial Naive Bayes", MultinomialNB()),
        "logistic_regression": (
            "Logistic Regression",
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, class_weight="balanced")
        ),
        "svm": ("Linear Support Vector Machine", LinearSVC(random_state=RANDOM_STATE, class_weight="balanced"))
    }

    results = []
    trained = {}

    for key, (name, classifier) in models.items():
        print(f"\nTraining {name}...")
        pipeline = create_pipeline(classifier)
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1-Score : {f1:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, predictions, target_names=["Ham", "Spam"], zero_division=0))

        joblib.dump(pipeline, MODELS_DIR / f"{key}.pkl")
        save_confusion_matrix(y_test, predictions, key, name)
        trained[key] = pipeline
        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "ModelKey": key
        })

    results_df = pd.DataFrame(results)
    display_df = results_df[["Model", "Accuracy", "Precision", "Recall", "F1-Score"]]
    print("\nModel Comparison:")
    print(display_df.to_string(index=False))
    display_df.to_csv(OUTPUTS_DIR / "model_results.csv", index=False)
    save_comparison_chart(display_df)

    best_row = results_df.loc[results_df["F1-Score"].idxmax()]
    best_key = best_row["ModelKey"]
    joblib.dump(trained[best_key], MODELS_DIR / "best_model.pkl")
    print(f"\nBest model selected by F1-score: {best_row['Model']}")
    print(f"Best F1-score: {best_row['F1-Score']:.4f}")
    print(f"Saved: {MODELS_DIR / 'best_model.pkl'}")


if __name__ == "__main__":
    main()
