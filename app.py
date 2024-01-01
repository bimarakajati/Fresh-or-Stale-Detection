import streamlit as st
import pickle
import numpy as np
import cv2
import pickle

st.set_page_config(
    page_title = "Fresh or Stale Detection",
    page_icon = ":apple:"
)

st.sidebar.title(f"Fresh or Stale Detection")

add_selectbox = st.sidebar.selectbox(
    "Select a fruit or vegetable",
    ("Apple", "Banana", "Bitter Gourd", "Capsicum", "Orange", "Tomato")
)

with st.sidebar:
    add_radio = st.radio(
        "Select a model",
        ("Raw Pixel Model", "Histogram Model")
    )

raw_pixel = ['apple_raw_pixel_model.pkl', 'banana_raw_pixel_model.pkl', 'bitter_gourd_raw_pixel_model.pkl', 'capsicum_raw_pixel_model.pkl', 'orange_raw_pixel_model.pkl', 'tomato_raw_pixel_model.pkl']
histogram = ['apple_histogram_model.pkl', 'banana_histogram_model.pkl', 'bitter_gourd_histogram_model.pkl', 'capsicum_histogram_model.pkl', 'orange_histogram_model.pkl', 'tomato_histogram_model.pkl']
acc_raw_pixel = {'apple': 81.50, 'banana': 91.15, 'bitter_gourd': 80.71, 'capsicum': 80.78, 'orange': 85.45, 'tomato': 73.51}
acc_histogram = {'apple': 78.15, 'banana': 73.44, 'bitter_gourd': 77.66, 'capsicum': 75.79, 'orange': 73.86, 'tomato': 77.76}

if add_radio == "Raw Pixel Model":
    model = pickle.load(open(f'model/{raw_pixel[["Apple", "Banana", "Bitter Gourd", "Capsicum", "Orange", "Tomato"].index(add_selectbox)]}', 'rb'))
    acc = acc_raw_pixel[add_selectbox.lower().replace(' ', '_')]
else:
    model = pickle.load(open(f'model/{histogram[["Apple", "Banana", "Bitter Gourd", "Capsicum", "Orange", "Tomato"].index(add_selectbox)]}', 'rb'))
    acc = acc_histogram[add_selectbox.lower().replace(' ', '_')]

st.title(f"Fresh or Stale Detection for {add_selectbox}")
st.write(f"**_Model's Accuracy_** :  :green[**{acc}**]%")

# Preprocessing
def image_to_feature_vector(image, size=(32, 32)):
    return cv2.resize(image, size).flatten()
def hist_eq(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])
    eq_color = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return  cv2.resize(eq_color, (32,32)).flatten()

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "webp"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1) # 1 means load color image
    if add_radio == "Raw Pixel Model":
        pre = image_to_feature_vector(img)
    else:
        pre = hist_eq(img)
    pre = pre.reshape(1,-1)
    prediction = model.predict(pre)
    probability_prediction = model.predict_proba(pre).max()
    # Show Labels and Probability of Predictions
    label = 'Prediction: '+prediction[0]
    probab = 'Probability: '+str(round(probability_prediction, 3))
    # Determine the font, scale, color, and text position
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1
    color = (0, 0, 0)
    x1, y1 = 10, 30
    x2, y2 = 10, 60
    # Add the label text to the picture
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert BGR to RGB
    cv2.putText(img, label, (x1, y1), font, scale, color, 2, cv2.LINE_AA)
    cv2.putText(img, probab, (x2, y2), font, scale, color, 2, cv2.LINE_AA)
    # Show images with labels
    st.image(img)