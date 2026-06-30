import streamlit as st
import requests

st.set_page_config(page_title="Cat vs Dog Classifier", page_icon="🐾", layout="centered")

st.title("🐾 Cat vs Dog Image Classifier")
st.write("Upload an image of a cat or a dog, and the model will predict what it is!")

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    
    # Classify button
    if st.button("Classify Image", type="primary"):
        with st.spinner("Analyzing..."):
            try:
                # Prepare the file payload for FastAPI
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                
                # Send POST request to your running FastAPI backend
                response = requests.post("http://127.0.0.1:8000/predict", files=files)
                
                if response.status_code == 200:
                    data = response.json()
                    prediction = data["prediction"].upper()
                    confidence = data["probability"]
                    
                    # Adjust confidence look depending on output structure
                    # If probability is near 1 for dog and 0 for cat:
                    prob_pct = confidence * 100 if prediction == "DOG" else (1 - confidence) * 100
                    
                    st.success(f"### Prediction: **{prediction}**")
                    st.metric(label="Confidence", value=f"{prob_pct:.2f}%")
                else:
                    st.error("Error: Backend server returned an invalid response.")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to FastAPI server. Is it running on port 8000?")