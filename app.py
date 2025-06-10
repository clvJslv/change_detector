import streamlit as st
import torch

from compare_module import (
    load_image as load_cmp_image,
    get_resnet_embedding, cosine, euclidean, perceptual_score
)
from deepfake_module import (
    load_deepfake_model, detect_deepfake, load_image as load_df_image, SUPPORTED
)

st.set_page_config(page_title="Detector Modular", layout="wide")
mode = st.radio("Modo:", ["Comparar Imagens", "Detectar DeepFake"])

if mode == "Comparar Imagens":
    st.header("Comparação de Imagens")
    imgA = st.file_uploader("Imagem Base", type=["jpg","png"])
    imgB = st.file_uploader("Imagem Suspeita", type=["jpg","png"])
    if imgA and imgB:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        col1, col2 = st.columns(2)
        with col1: st.image(imgA, caption="A", width=200)
        with col2: st.image(imgB, caption="B", width=200)
        sim_rigor = st.slider("Quão rigoroso deve ser para considerar as imagens iguais?", 0.0, 1.0, 0.9)
        sim_diferenca = st.slider("Quão diferentes devem ser para considerar alteradas?", 0.0, 5.0, 1.0)
        use_lpips = st.checkbox("Incluir avaliação perceptual (LPIPS)?", True)
        if st.button("Comparar"):
            pathA = imgA
            pathB = imgB
            embA = get_resnet_embedding(load_cmp_image(pathA, device), device)
            embB = get_resnet_embedding(load_cmp_image(pathB, device), device)
            c = cosine(embA, embB)
            e = euclidean(embA, embB)
            st.write(f"Similaridade (cosine): {c:.3f}")
            st.write(f"Diferença (euclidean): {e:.3f}")
            if c >= sim_rigor and e <= sim_diferenca:
                st.success("Imagens consideradas iguais.")
            else:
                st.error("Imagens consideradas diferentes.")
            if use_lpips:
                p = perceptual_score(pathA, pathB, device)
                st.write(f"LPIPS: {p:.3f}")

else:
    st.header("Detecção de DeepFake Facial")
    img = st.file_uploader("Imagem (rosto)", type=["jpg","png"])
    if img:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        st.image(img, caption="Imagem para análise", width=300)
        model_name = st.selectbox("Escolha modelo:", SUPPORTED)
        if st.button("Detectar DeepFake"):
            fe, m = load_deepfake_model(model_name, device)
            p_real, p_fake = detect_deepfake(fe, m, load_df_image(img), device)
            st.write(f"Prob real: {p_real:.3f} — Prob fake: {p_fake:.3f}")
            if p_fake>0.5: st.error("Provavelmente um DeepFake")
            else:        st.success("Imagem aparentemente real")