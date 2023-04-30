import streamlit as st
from PIL import Image
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import cv2

st.set_page_config(page_title="Vijay's App",page_icon="memo",layout="wide")

img1 = Image.open('head.png')
#img2 = Image.open('logo.png')

#st.image(img2)
#st.write("---")
st.header("Predicted MeanVB of a Tool")

st.sidebar.image(img1)
st.sidebar.header("Predict Tool MeanVB")
img = st.sidebar.file_uploader("Choose Input Image",type=["jpg"])

def r2_score(y_true,y_pred):
    u = sum(square(y_true-y_pred))
    v = sum(square(y_true-mean(y_true)))
    return (1-u/(v+epsilon()))

model = load_model('surface_model_24IMG.h5',custom_objects={"r2_score": r2_score})

if img:
    img = Image.open(img)
    st.image(img,caption="Tool Image")
    img_grey = img.convert("L")
    img_grey = img_grey.resize((64,64))
    imgs = np.array(img_grey)
    data = np.reshape(imgs,(1,64,64,1))
    def preprocess(img):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        return final

    data = np.array([preprocess(img) for img in data])

    cnn_features = model.predict(data)
    cnn_features = cnn_features.reshape(cnn_features.shape[0], -1)
    
    import pickle
    model = pickle.load(open("rf_model.pkl","rb"))
    pred = model.predict(cnn_features)

    if st.sidebar.button("Predict MeanVB "):
        st.subheader("MeanVB : {}".format(pred[0]))

