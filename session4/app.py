import streamlit as st
import joblib

# Load the model
model = joblib.load('Linear_Regression_model.pkl')

st.title("Student Marks Prediction")
st.write("Predict marks based on study hours")

study_hours = st.number_input("Enter study hours", min_value=0.0, max_value=24.0, value=1.0, step=0.5)

if st.button("Predict"):
    # Convert the array output directly to a float scalar
    prediction = float(model.predict([[study_hours]]))
    
    # Print the float directly without
    st.success(f"Predicted marks: {prediction:.2f}")