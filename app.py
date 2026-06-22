import streamlit as st
import numpy as np
from PIL import Image

from utils import process_pca

st.set_page_config(page_title="Kompresi PCA - Berwarna")

st.title("Kompresi Gambar Menggunakan PCA")
st.header("Foto Berwarna")

uploaded = st.file_uploader(
    "Upload Foto Berwarna",
    type=["jpg", "jpeg", "png"],
    key="warna"
)

k = st.slider(
    "Nilai K",
    1,
    500,
    50
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

    if st.button("Kompres Berwarna"):

        hasil, analisis = process_pca(
            image_np,
            k,
            False
        )

        st.image(
            hasil,
            caption="Hasil PCA"
        )

        st.text(analisis)
