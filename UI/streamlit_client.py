import streamlit as st
import requests
import os

# Hardcoded paths and target column for automation
default_train_path = "data/mushroom_train.csv"
default_test_path = "data/mushroom_train.csv"
default_target_col = "edible"
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.title("Naive Bayes Classifier - Automated Streamlit Client")

# Button to rerun workflow
def rerun():
    st.session_state['run'] = True

if 'run' not in st.session_state:
    st.session_state['run'] = True

st.button("Reload & Run Automated Workflow", on_click=rerun)

if st.session_state['run']:
    # --- Train Model ---
    st.header("1. Train Model (Automated)")
    with open(default_train_path, "rb") as f:
        files = {"file": (os.path.basename(default_train_path), f, "text/csv")}
        data = {"target_column": default_target_col}
        response = requests.post(f"{API_URL}/train", files=files, data=data)
    if response.ok and response.json().get("status"):
        if response.json().get("cached"):
            st.info("Model is already built for this dataset and target column.")
        else:
            st.success(f"Model trained! Target column: {default_target_col}")
    else:
        st.error(f"Error: {response.json().get('error', response.text)}")

    # --- Test Model Accuracy ---
    st.header("2. Test Model Accuracy (Automated)")
    with open(default_test_path, "rb") as f:
        files = {"file": (os.path.basename(default_test_path), f, "text/csv")}
        data = {"target_column": default_target_col}
        response = requests.post(f"{API_URL}/test", files=files, data=data)
    if response.ok and "accuracy" in response.json():
        accuracy = response.json()["accuracy"]
        st.success(f"Model accuracy: {accuracy:.2%}")
        if "confusion_matrix" in response.json():
            st.write("Confusion Matrix:")
            st.write(response.json()["confusion_matrix"])
    else:
        st.error(f"Error: {response.json().get('error', response.text)}")

    # --- Model Info ---
    st.header("3. Model Info (Automated)")
    info_response = requests.get(f"{API_URL}/info")
    if info_response.ok:
        st.json(info_response.json())
    else:
        st.error("Error fetching model info.")

    # --- Classify a Sample Record (Optional, Automated Example) ---
    st.header("4. Classify Example Record (Automated)")
    # Try to get features from model info
    features = info_response.json().get("Features") if info_response.ok else None
    if features:
        # Example: use the first row from the training data as a sample
        import pandas as pd
        df = pd.read_csv(default_train_path)
        sample = df.iloc[0][[f for f in features]].to_dict()
        st.write("Classifying sample:", sample)
        response = requests.post(f"{API_URL}/predict", json=sample)
        if response.ok and "prediction" in response.json():
            st.success(f"Prediction: {response.json()['prediction']}")
        else:
            st.error(f"Error: {response.json().get('error', response.text)}")
    else:
        st.info("Model features not available.")

    # Reset run state so rerun button works
    st.session_state['run'] = False 