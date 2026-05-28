import streamlit as st
import numpy as np
import pandas as pd
from tensorflow import keras
from PIL import Image

# Load model
# model = keras.models.load_model("brain_tumor_classifier.keras")
@st.cache_resource
def load_model():
    return keras.models.load_model("brain_tumor_classifier.keras")

model = load_model()

# Class labels
classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Streamlit UI
st.title("Brain Tumor MRI Classifier")

st.write("Upload an MRI image to predict the tumor type.")

st.warning("For educational purposes only. Not for medical diagnosis.")

uploaded_file = st.file_uploader(
    "Choose an MRI image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file).convert("RGB")

    # Display image
    st.image(image, caption="Uploaded MRI Image", use_container_width=True)

    # Preprocess image
    image = image.resize((224, 224))

    img_array = np.array(image) / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)

    predicted_class = classes[np.argmax(prediction)]

    confidence = np.max(prediction) * 100

    # Output
    st.subheader("Prediction")

    if predicted_class == "notumor":
        st.success("No Tumor Detected")
    else:
        st.error(f"Tumor Type: {predicted_class}")

    # st.write(f"Confidence: {confidence:.2f}%")

    st.markdown("### Class Probabilities")
    probability_df = pd.DataFrame(
        {
            "Class": [label.title() for label in classes],
            # "Probability (%)": [round(float(prob * 100), 2) for prob in prediction[0]],
            "Probability (%)": [f"{prob * 100:.8f}" for prob in prediction[0]],
        }
    ).sort_values("Probability (%)", ascending=False)

    st.dataframe(probability_df, use_container_width=True, hide_index=True)

# streamlit run app.py