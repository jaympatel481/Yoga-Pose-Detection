from img_classification import teachable_machine_classification
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import cv2
st.header('AI Body Tracker')
# st.write("Leaf Disease Detection")
uploaded_file = st.file_uploader("Choose Pose Image")

if uploaded_file is not None:
    # data = np.ndarray(shape=(1,224, 224,3), dtype=np.float32)
    image = Image.open(uploaded_file)
    
    
    #turn the image into a numpy array
    
    # Normalize the image
    
    # Load the image into the array
    # data[0] = normalized_image_array

    st.image(image, caption='Uploaded Photo', use_column_width=True)
    st.write("")
    st.write("Detecting...")
    label = teachable_machine_classification(image, 'keras_model.h5')


    if label == 0:
        text = "<h4 style='color:red'>DownDog</h4>"
        # st.write("Pothole Detected on Road")
        st.markdown(text,unsafe_allow_html=True)
      
    if label == 1:
        text = "<h4 style='color:green'>Goddess</h4>"
        st.markdown(text,unsafe_allow_html=True)

    if label == 2:
        text = "<h4 style='color:red'>Plank</h4>"
        # st.write("Pothole Detected on Road")
        st.markdown(text,unsafe_allow_html=True)
    if label == 3:
        text = "<h4 style='color:green'>Plank</h4>"
        st.markdown(text,unsafe_allow_html=True)

    if label == 4:
        text = "<h4 style='color:red'>Tree</h4>"
        # st.write("Pothole Detected on Road")
        st.markdown(text,unsafe_allow_html=True)

    if label == 5:
        text = "<h4 style='color:green'>Warrior</h4>"
        st.markdown(text,unsafe_allow_html=True)
