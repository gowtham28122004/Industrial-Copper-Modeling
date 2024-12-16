import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, VotingRegressor, VotingClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

@st.cache_data
def load_data():
    data = pd.read_csv(r'../Copper_Set_processed.csv')
    return data

def save_models(reg_model, clf_model):
    with open("regression_model.pkl", "wb") as f:
        pickle.dump(reg_model, f)
    with open("classification_model.pkl", "wb") as f:
        pickle.dump(clf_model, f)

def train_models():
    data = load_data()
    X = data[['quantity', 'thickness', 'width']]
    y_reg = data['selling_price']
    y_clf = data['status']

    X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X, y_clf, test_size=0.2, random_state=42)

    rf_reg = RandomForestRegressor()
    xgb_reg = XGBRegressor()
    voting_reg = VotingRegressor([('rf', rf_reg), ('xgb', xgb_reg)])
    voting_reg.fit(X_train, y_train_reg)
    reg_predictions = voting_reg.predict(X_test)
    st.write(f"Regression Model R2 Score: {1 - mean_squared_error(y_test_reg, reg_predictions)}")

    rf_clf = RandomForestClassifier()
    xgb_clf = XGBClassifier()
    voting_clf = VotingClassifier([('rf', rf_clf), ('xgb', xgb_clf)], voting='soft')
    voting_clf.fit(X_train_clf, y_train_clf)
    clf_predictions = voting_clf.predict(X_test_clf)
    st.write(f"Classification Model Accuracy: {accuracy_score(y_test_clf, clf_predictions)}")

    save_models(voting_reg, voting_clf)

def load_models():
    with open("regression_model.pkl", "rb") as f:
        reg_model = pickle.load(f)
    with open("classification_model.pkl", "rb") as f:
        clf_model = pickle.load(f)
    return reg_model, clf_model

def main():
    st.sidebar.title("Industrial Copper App")
    page = st.sidebar.radio("Navigation", ["About", "Selling Price Prediction", "Status Prediction"])

    if page == "About":
        st.title("About")
        st.write("""
        This Streamlit app predicts:
        - **Selling Price** of industrial copper products (regression task)
        - **Status** of deals (Won/Lost) (classification task)
        
        The models use ensemble methods like **RandomForest** and **XGBoost** to improve accuracy.
        """)

    elif page == "Selling Price Prediction":
        st.title("Selling Price Prediction")
        reg_model, _ = load_models()

        st.write("Enter input features:")
        quantity = st.number_input("Quantity", min_value=1, value=100)
        thickness = st.number_input("Thickness", min_value=0.1, value=2.0)
        width = st.number_input("Width", min_value=1.0, value=5.0)

        if st.button("Predict Price"):
            inputs = np.array([[quantity, thickness, width]])
            prediction = reg_model.predict(inputs)
            st.write(f"**Predicted Selling Price:** ${prediction[0]:.2f}")

    elif page == "Status Prediction":
        st.title("Status Prediction")
        _, clf_model = load_models()

        st.write("Enter input features:")
        quantity = st.number_input("Quantity", min_value=1, value=100)
        thickness = st.number_input("Thickness", min_value=0.1, value=2.0)
        width = st.number_input("Width", min_value=1.0, value=5.0)

        if st.button("Predict Status"):
            inputs = np.array([[quantity, thickness, width]])
            prediction = clf_model.predict(inputs)
            status = "WON" if prediction[0] == 1 else "LOST"
            st.write(f"**Predicted Status:** {status}")

if __name__ == "__main__":
    train_models()
    main()
