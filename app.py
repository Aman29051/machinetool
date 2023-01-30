import streamlit as st
from PIL import Image
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Vijay's App",page_icon="memo",layout="wide")

img1 = Image.open('head.png')
#img2 = Image.open('logo.png')

#st.image(img2)
#st.write("---")
st.header("Predicted Roughness of a Tool")


st.sidebar.image(img1)
st.sidebar.header("Predict Tool Roughness")
img = st.sidebar.file_uploader("Choose Input Image",type=["jpg"])

model = load_model('surface_model.h5')

if img:
	img = Image.open(img)
	st.image(img,caption="Tool Image")
	img_grey = img.convert("L")
	img_grey = img_grey.resize((64,64))
	imgs = np.array(img_grey)
	data = np.reshape(imgs,(1,64,64,1))

	roughness = model.predict(data)

if st.sidebar.button("Predict Roughness "):
    st.subheader("Roughness : {}".format(roughness[0][0]))


