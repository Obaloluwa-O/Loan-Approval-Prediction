Loan Approval Predictor App

This project is a Loan Approval Predictor, built using Python, Scikit-learn, XGBoost, and Streamlit. The app leverages machine learning to predict whether a loan application is likely to be approved or rejected based on key financial and personal features.

The underlying dataset contains financial records and associated information such as:

CIBIL score

Income

Employment status

Loan term

Loan amount

Asset value

Loan status (Approved/Rejected)

Using this dataset, I trained an XGBoost classifier within a Scikit-learn pipeline that handles preprocessing, feature scaling, and encoding of categorical variables. Feature importance analysis was performed to identify the most influential factors in loan approval decisions.

The Streamlit app provides an interactive interface where users can:

Input numeric and categorical applicant details

Instantly see predictions for loan approval

Visualize feature importance to understand the model's decision-making

This app demonstrates end-to-end machine learning deployment, from dataset preprocessing, model training, and evaluation, to building a user-friendly interactive application.

Technologies used: Python, Pandas, NumPy, Scikit-learn, XGBoost, Streamlit, Matplotlib, Seaborn
