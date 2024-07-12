import streamlit as st
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img
from tensorflow.keras.models import model_from_json
import numpy as np
import os

# Title
st.title("Cat 🐱 Or Dog 🐶 Recognizer")

# File Uploader
img_file_buffer = st.file_uploader("Upload an image here 👇🏻")

if img_file_buffer is not None:
    try:
        image = Image.open(img_file_buffer)
        img_array = np.array(image)
        st.image(image, use_column_width=True)
        st.write("Now, click the '👉🏼 Predict' button to see the prediction!")
    except Exception as e:
        st.write(f"### ❗ Error loading image: {e}")

# Predict Button
submit = st.button("👉🏼 Predict")

# Model Prediction Function
def processing(testing_image_path):
    IMG_SIZE = 50
    img = load_img(testing_image_path, target_size=(IMG_SIZE, IMG_SIZE), color_mode="grayscale")
    img_array = img_to_array(img) / 255.0
    img_array = img_array.reshape((1, 50, 50, 1))
    prediction = loaded_model.predict(img_array)
    return prediction

# Generate Result Function
def generate_result(prediction):
    st.write("## 🎯 RESULT")
    if prediction[0] < 0.5:
        st.write("## Model predicts it as an image of a CAT 🐱!!!")
    else:
        st.write("## Model predicts it as an image of a DOG 🐶!!!")

# Predict Button Clicked
if submit:
    if img_file_buffer is not None:
        try:
            # Ensure the temporary directory exists
            temp_dir = "temp_dir"
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            
            save_img(os.path.join(temp_dir, "test_image.png"), img_array)
            image_path = os.path.join(temp_dir, "test_image.png")

            # Load Model
            try:
                with open("model/model.json", 'r') as json_file:
                    loaded_model_json = json_file.read()
                loaded_model = model_from_json(loaded_model_json)
                loaded_model.load_weights("model/model.h5")
                loaded_model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')
            except Exception as e:
                st.write(f"### ❗ Error loading model: {e}")
                raise e

            prediction = processing(image_path)
            generate_result(prediction)
        except Exception as e:
            st.write(f"### ❗ Error during prediction: {e}")
    else:
        st.write("### ❗ No image uploaded yet")

# Footer
st.write("Cooked By Talib Rz")
