import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load model
model = load_model(r'C:\Users\mpawe\Downloads\model_trained_class.hdf5', compile=False)

# Define class labels
food_list = ['chicken_wings','chocolate_cake','donuts','french_fries','french_toast',
             'fried_rice','hamburger','ice_cream','omelette','pancakes','pizza',
             'pork_chop','samosa','spring_rolls','waffles']

# Define healthiness labels
health_labels = {
    'chicken_wings': 'Unhealthy',
    'chocolate_cake': 'Unhealthy',
    'donuts': 'Unhealthy',
    'french_fries': 'Unhealthy',
    'french_toast': 'Unhealthy',
    'fried_rice': 'Unhealthy',
    'hamburger': 'Unhealthy',
    'ice_cream': 'Unhealthy',
    'omelette': 'Healthy',
    'pancakes': 'Unhealthy',
    'pizza': 'Unhealthy',
    'pork_chop': 'Unhealthy',
    'samosa': 'Unhealthy',
    'spring_rolls': 'Unhealthy',
    'waffles': 'Unhealthy'
}

# Title
st.title('🍔🍩 Food Image Classifier')

# Upload image
uploaded_file = st.file_uploader("Choose a food image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    if st.button('Predict'):
        # Preprocess the image
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Prediction
        pred = model.predict(img_array)
        index = np.argmax(pred)
        predicted_label = food_list[index]
        health_label = health_labels[predicted_label]

        # Show results
        st.success(f'Predicted Food: **{predicted_label}**')
        st.info(f'🩺 Healthiness: **{health_label}**')
