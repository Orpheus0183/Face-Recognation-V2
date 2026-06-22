import streamlit as st
import numpy as np
from PIL import Image

from utils import process_pca

st.set_page_config(page_title="Kompresi PCA - Grayscale")

st.title("Kompresi Gambar Menggunakan PCA")
st.header("Foto Grayscale")

uploaded = st.file_uploader(
    "Upload Foto",
    type=["jpg", "jpeg", "png"],
    key="gray"
)

k = st.slider(
    "Nilai K",
    1,
    500,
    50,
    key="slider2"
)

if uploaded:

    image = Image.open(
        uploaded
    ).convert("RGB")

    image_np = np.array(image)

    st.image(
        image_np,
        caption="Gambar Asli"
    )

    if st.button("Kompres Grayscale"):

        hasil, analisis = process_pca(
            image_np,
            k,
            True
        )

        st.image(
            hasil,
            caption="Hasil PCA"
        )

        st.text(analisis)
