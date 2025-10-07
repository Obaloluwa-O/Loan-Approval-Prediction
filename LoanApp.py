import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load trained pipeline and feature lists
model = joblib.load("loan_approval_model.pkl")  # fitted pipeline
num_features = joblib.load("num_features.pkl")
cat_features = joblib.load("cat_features.pkl")

# Load original dataset to get unique values for dropdowns
data = pd.read_csv(r"C:\Users\ObaloluwaOluokun\Downloads\loan_approval_dataset.csv")
data.columns = data.columns.str.strip()  # remove spaces
data = data.drop(columns=[col for col in ["loan_status", "loan_id"] if col in data.columns])

st.title("Loan Approval Predictor 💰")
st.write("Enter applicant details below:")

# Prepare layout: left, center, right
col_left, col_center, col_right = st.columns([1, 1, 1])

# Distribute features to left and right columns
left_features = num_features[:len(num_features)//2] + cat_features[:len(cat_features)//2]
right_features = num_features[len(num_features)//2:] + cat_features[len(cat_features)//2:]

input_data = {}

# Left column inputs
with col_left:
    st.header("Inputs")
    for feature in left_features:
        if feature in data.columns:
            if feature in num_features:
                input_data[feature] = st.number_input(feature, value=float(data[feature].median()))
            else:
                unique_vals = data[feature].dropna().unique().tolist()
                input_data[feature] = st.selectbox(feature, unique_vals)
        else:
            input_data[feature] = st.text_input(feature, value="") if feature in cat_features else st.number_input(feature, value=0.0)

# Right column inputs
with col_right:
    st.header("Inputs")
    for feature in right_features:
        if feature in data.columns:
            if feature in num_features:
                input_data[feature] = st.number_input(feature, value=float(data[feature].median()))
            else:
                unique_vals = data[feature].dropna().unique().tolist()
                input_data[feature] = st.selectbox(feature, unique_vals)
        else:
            input_data[feature] = st.text_input(feature, value="") if feature in cat_features else st.number_input(feature, value=0.0)

# Center column: button and prediction result
with col_center:
    st.write("")  # spacing
    st.write("")
    predict_button = st.button("Predict")  # centered
    if predict_button:
        try:
            X_new = pd.DataFrame([input_data])
            pred = model.predict(X_new)[0]
            if pred == 1:
                result = "Approve ✅"
                color = "green"
            else:
                result = "Reject ❌"
                color = "red"
            st.markdown(f"<h2 style='text-align: center; color: {color};'>{result}</h2>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error during prediction: {e}")

# Feature importance
if st.checkbox("Show Feature Importance"):
    try:
        xgb_model = model.named_steps['Classifier']
        cat_names = []
        if 'car' in model.named_steps['preprocessor'].named_transformers_:
            cat_names = model.named_steps['preprocessor'].named_transformers_['car'].get_feature_names_out(cat_features)
        all_features = np.concatenate([num_features, cat_names])
        importance = xgb_model.feature_importances_
        indices = np.argsort(importance)

        plt.figure(figsize=(10, len(all_features)/2))
        plt.barh(all_features[indices], importance[indices], color="teal")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.title("Feature Importance")
        st.pyplot(plt)
    except Exception as e:
        st.error(f"Cannot show feature importance: {e}")
