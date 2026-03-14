import streamlit as st
from PIL import Image
import requests

st.title("🌙 Moon Surface Classification")

uploaded_file = st.file_uploader(
    "Upload lunar image",
    type=["jpg","png","jpeg"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    st.image(image,use_container_width=True)

    if st.button("Predict"):

        files = {
            "file":(
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type
            )
        }

        response = requests.post(
            "http://127.0.0.1:8000/predict",
            files=files
        )

        result = response.json()

        st.success(f"Prediction: {result['prediction']}")
        st.write(f"Confidence: {result['confidence']:.2f}")
