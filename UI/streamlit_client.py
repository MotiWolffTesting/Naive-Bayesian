import streamlit as st
import pandas as pd
import requests
import io

API_URL = "http://127.0.0.1:8000"

st.title("Naive Bayes Classifier - Streamlit Client")

# --- Train Model ---
st.header("1. Train Model")
train_file = st.file_uploader("Upload training CSV", type=["csv"], key="train")
if train_file is not None:
    train_df = pd.read_csv(train_file)
    st.write("Preview of training data:", train_df.head())
    target_col = st.selectbox("Select target column", train_df.columns)
    if st.button("Train Model"):
        train_file.seek(0)
        files = {"file": (train_file.name, train_file, "text/csv")}
        data = {"target_column": target_col}
        with st.spinner("Training model..."):
            response = requests.post(f"{API_URL}/train", files=files, data=data)
        if response.ok and response.json().get("status"):
            st.success(f"Model trained! Target column: {target_col}")
        else:
            st.error(f"Error: {response.json().get('error', response.text)}")

# --- Test Model Accuracy ---
st.header("2. Test Model Accuracy")
test_file = st.file_uploader("Upload test CSV", type=["csv"], key="test")
if test_file is not None:
    test_df = pd.read_csv(test_file)
    st.write("Preview of test data:", test_df.head())
    test_target_col = st.selectbox("Select target column for test", test_df.columns)
    if st.button("Test Accuracy"):
        test_file.seek(0)
        files = {"file": (test_file.name, test_file, "text/csv")}
        data = {"target_column": test_target_col}
        with st.spinner("Testing model accuracy..."):
            response = requests.post(f"{API_URL}/test", files=files, data=data)
        if response.ok and "accuracy" in response.json():
            accuracy = response.json()["accuracy"]
            st.success(f"Model accuracy: {accuracy:.2%}")
        else:
            st.error(f"Error: {response.json().get('error', response.text)}")

# --- Classify Single Record ---
st.header("3. Classify Single Record")
if st.button("Get Model Info", key="get_info1"):
    info_response = requests.get(f"{API_URL}/info")
    if info_response.ok and "Features" in info_response.json():
        features = info_response.json()["Features"]
        st.session_state["features"] = features
        st.success(f"Model features: {features}")
    else:
        st.error("Model is not trained or error fetching info.")

features = st.session_state.get("features", None)
if features:
    st.write("Enter values for the following features:")
    record = {}
    for feature in features:
        record[feature] = st.text_input(f"{feature}", key=f"input_{feature}")
    if st.button("Classify Record"):
        with st.spinner("Classifying..."):
            response = requests.post(f"{API_URL}/predict", json=record)
        if response.ok and "prediction" in response.json():
            st.success(f"Prediction: {response.json()['prediction']}")
        else:
            st.error(f"Error: {response.json().get('error', response.text)}")

# --- Model Info ---
st.header("4. Model Info")
if st.button("Show Model Info", key="get_info2"):
    info_response = requests.get(f"{API_URL}/info")
    if info_response.ok:
        st.json(info_response.json())
    else:
        st.error("Error fetching model info.") 