import numpy as np
from sklearn.decomposition import PCA
import math


def calculate_psnr(original, compressed):
    mse = np.mean(
        (original.astype(np.float64) -
         compressed.astype(np.float64)) ** 2
    )

    if mse == 0:
        return float('inf')

    max_pixel = 255.0
    return 20 * math.log10(max_pixel / math.sqrt(mse))


def get_pca_size_kb(h, w, k, n_channels):

    pca_floats = n_channels * (h * k + k * w + w)

    return (pca_floats * 4) / 1024


def get_original_size_kb(h, w, n_channels):

    return (h * w * n_channels) / 1024


def process_pca(input_img, k_value, is_gray=False):

    if input_img is None:
        return None, "Tidak ada gambar."

    if is_gray and len(input_img.shape) == 3:

        input_img = np.dot(
            input_img[..., :3],
            [0.299, 0.587, 0.114]
        ).astype(np.uint8)

    original_shape = input_img.shape

    h = original_shape[0]
    w = original_shape[1]

    n_channels = (
        1 if len(original_shape) == 2 or is_gray
        else 3
    )

    max_k = min(h, w)

    k = min(int(k_value), max_k)

    k_warning = ""

    if k < int(k_value):
        k_warning = (
            f"K dipotong dari {int(k_value)} "
            f"menjadi {k}\n\n"
        )

    if n_channels == 3:

        channels = [
            input_img[:, :, i]
            for i in range(3)
        ]

        compressed_channels = []

        variances = []

        for ch in channels:

            pca = PCA(n_components=k)

            transformed = pca.fit_transform(
                ch.astype(np.float64)
            )

            reconstructed = pca.inverse_transform(
                transformed
            )

            compressed_channels.append(
                reconstructed
            )

            variances.append(
                np.sum(
                    pca.explained_variance_ratio_
                )
            )

        img_reconstructed = np.stack(
            compressed_channels,
            axis=2
        )

        avg_variance = (
            sum(variances) / 3
        ) * 100

    else:

        pca = PCA(n_components=k)

        transformed = pca.fit_transform(
            input_img.astype(np.float64)
        )

        img_reconstructed = pca.inverse_transform(
            transformed
        )

        avg_variance = (
            np.sum(
                pca.explained_variance_ratio_
            ) * 100
        )

    img_final = np.clip(
        img_reconstructed,
        0,
        255
    ).astype(np.uint8)

    psnr_value = calculate_psnr(
        input_img,
        img_final
    )

    original_kb = get_original_size_kb(
        h,
        w,
        n_channels
    )

    compressed_kb = get_pca_size_kb(
        h,
        w,
        k,
        n_channels
    )

    size_ratio = (
        original_kb / compressed_kb
        if compressed_kb > 0
        else 0
    )

    original_params = (
        h * w * n_channels
    )

    pca_params = (
        n_channels * (k * w + h * k + w)
    )

    parameter_ratio = (
        original_params / pca_params
        if pca_params > 0
        else 0
    )

    if psnr_value == float('inf'):
        kategori = "Identik dengan asli"

    elif psnr_value > 40:
        kategori = "Sangat Bagus"

    elif psnr_value > 30:
        kategori = "Bagus"

    elif psnr_value >= 20:
        kategori = "Cukup"

    else:
        kategori = "Jelek"

    analysis_text = (
        f"{k_warning}"
        f"--- KUALITAS ---\n"
        f"Nilai K : {k}/{max_k}\n"
        f"Informasi dipertahankan : {avg_variance:.2f}%\n"
        f"PSNR : {psnr_value:.2f} dB\n"
        f"Kategori : {kategori}\n\n"

        f"--- UKURAN ---\n"
        f"Original : {original_kb:.2f} KB\n"
        f"PCA : {compressed_kb:.2f} KB\n"
        f"Rasio ukuran : {size_ratio:.2f}x\n\n"

        f"--- PARAMETER ---\n"
        f"Parameter asli : {original_params:,}\n"
        f"Parameter PCA : {pca_params:,}\n"
        f"Rasio parameter : {parameter_ratio:.2f}x"
    )

    return img_final, analysis_text
