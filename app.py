from pathlib import Path

import joblib
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"

st.set_page_config(page_title="Email Spam Detection", page_icon="📧", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def get_confidence(model, text):
    try:
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([text])[0]
            return float(max(probabilities))
    except Exception:
        return None
    return None


def main():
    st.title("📧 Email Spam Detection")
    st.subheader("Machine Learning Based Email Spam Detection System")
    st.write("Enter or paste an email below and let the trained machine learning model classify it.")

    with st.sidebar:
        st.header("ℹ️ System Information")
        st.write("**Algorithms trained:**")
        st.write("- Multinomial Naive Bayes")
        st.write("- Logistic Regression")
        st.write("- Linear Support Vector Machine")
        st.write("**Feature extraction:** TF-IDF")
        st.write("**Best model selection:** Highest F1-score")
        st.divider()
        st.write("Run `python train_model.py` before using the application.")

    example_spam = "URGENT! You have won $10,000. Click http://example.com now to claim your prize before your account expires!"
    example_ham = "Hi team, the project meeting is scheduled for tomorrow at 10 AM. Please bring your progress update. Thanks."

    col1, col2 = st.columns(2)
    if col1.button("Load Spam Example"):
        st.session_state.email_text = example_spam
    if col2.button("Load Legitimate Example"):
        st.session_state.email_text = example_ham

    if "email_text" not in st.session_state:
        st.session_state.email_text = ""

    email_text = st.text_area(
        "Email Content",
        value=st.session_state.email_text,
        height=250,
        placeholder="Paste the email content here..."
    )

    if st.button("🔍 Detect Spam", type="primary", use_container_width=True):
        if not email_text or not email_text.strip():
            st.warning("Please enter a valid email message before detection.")
            return

        if not MODEL_PATH.exists():
            st.error("Trained model not found. Please run `python train_model.py` first.")
            return

        try:
            model = load_model()
            prediction = int(model.predict([email_text])[0])
            confidence = get_confidence(model, email_text)

            if prediction == 1:
                st.error("🚨 SPAM EMAIL DETECTED")
                st.write("This email was classified as **Spam**.")
            else:
                st.success("✅ LEGITIMATE EMAIL")
                st.write("This email was classified as **Ham / Legitimate**.")

            if confidence is not None:
                st.metric("Prediction Confidence", f"{confidence * 100:.2f}%")
            else:
                st.info("Confidence is not available for the selected model.")
        except Exception:
            st.error("Unable to process the prediction. Please verify that the model was trained correctly.")


if __name__ == "__main__":
    main()
