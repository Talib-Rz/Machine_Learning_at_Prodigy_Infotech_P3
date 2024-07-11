import streamlit as st
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img
from tensorflow.keras.models import model_from_json
import numpy as np
import random
import os

# Define the list of image paths
img_path_list = ["path/to/your/cat_image.jpg", "path/to/your/dog_image.jpg"]

# Check if image paths exist
for img_path in img_path_list:
    if not os.path.exists(img_path):
        st.error(f"Image path does not exist: {img_path}")

# Title of the app
st.title("Cat or Dog Recognizer")

# Randomly select an image to display
index = random.choice([0, 1])
image_path = img_path_list[index]
image = Image.open(image_path)
st.image(image, use_column_width=True)

# File uploader for user to upload an image
img_file_buffer = st.file_uploader("Please upload an image:")

if img_file_buffer:
    try:
        image = Image.open(img_file_buffer)
        img_array = np.array(image)
        st.image(image, use_column_width=True)
    except Exception as e:
        st.error(f"An error occurred while processing the image: {e}")

# Button for prediction
submit = st.button("Predict")

def processing(testing_image_path):
    IMG_SIZE = 50
    img = load_img(testing_image_path, target_size=(IMG_SIZE, IMG_SIZE), color_mode="grayscale")
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = img_array.reshape((1, 50, 50, 1))
    prediction = loaded_model.predict(img_array)
    return prediction

def generate_result(prediction):
    if prediction[0] < 0.5:
        st.write("Model predicted it as an image of a CAT.")
    else:
        st.write("Model predicted it as an image of a DOG.")

if submit:
    if img_file_buffer:
        try:
            # Ensure the directory exists
            if not os.path.exists("temp_dir"):
                os.makedirs("temp_dir")
            
            # Save uploaded image
            save_img("temp_dir/test_image.png", img_array)

            image_path = "temp_dir/test_image.png"
            model_path_h5 = "model/model.h5"
            model_path_json = "model/model.json"
            
            # Load the model
            with open(model_path_json, 'r') as json_file:
                loaded_model_json = json_file.read()
            
            loaded_model = model_from_json(loaded_model_json)
            loaded_model.load_weights(model_path_h5)
            loaded_model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')

            # Make prediction
            prediction = processing(image_path)

            # Generate result
            generate_result(prediction)

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.write("No image uploaded.")
