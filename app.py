#app.py
import streamlit as st
import torch
from PIL import Image

from compare_module import (
    load_image as load_cmp_image,
    get_resnet_embedding, cosine, euclidean, perceptual_score
)
from deepfake_module import (
    load_local_model, predict_local, load_image as load_df_image
)

st.set_page_config(page_title="Detector de Imagem IA/Real", layout="wide")
mode = st.radio("Modo de Operação:", ["Comparar Imagens", "Detectar IA-Gerada"])

if mode == "Comparar Imagens":
    st.header("🔍 Comparação de Duas Imagens")
    imgA = st.file_uploader("Imagem A (base):", type=["jpg","png","jpeg"], key="cmpA")
    imgB = st.file_uploader("Imagem B (suspeita):", type=["jpg","png","jpeg"], key="cmpB")
    if imgA and imgB:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        c1, c2 = st.columns(2)
        with c1:
            st.image(imgA, caption="Imagem A", width=250)
        with c2:
            st.image(imgB, caption="Imagem B", width=250)

        rigor = st.slider(
            "Nível de rigor para considerar IGUAIS:",
            min_value=0.0, max_value=1.0, value=0.9, step=0.01
        )
        diff_thresh = st.slider(
            "Quão diferentes para considerar ALTERADAS:",
            min_value=0.0, max_value=5.0, value=1.0, step=0.1
        )
        lpips_opt = st.checkbox("Incluir LPIPS (avaliação perceptual)?", value=True)

        if st.button("Executar Comparação"):
            embA = get_resnet_embedding(load_cmp_image(imgA, device), device)
            embB = get_resnet_embedding(load_cmp_image(imgB, device), device)
            cos_sim = cosine(embA, embB)
            euc_dist = euclidean(embA, embB)
            st.write(f"• Similaridade (Cosine): {cos_sim:.3f}")
            st.write(f"• Distância (Euclidiana): {euc_dist:.3f}")
            if cos_sim >= rigor and euc_dist <= diff_thresh:
                st.success("✅ Imagens consideradas IGUAIS.")
            else:
                st.error("❌ Imagens consideradas DIFERENTES.")
            if lpips_opt:
                lp = perceptual_score(imgA, imgB, device)
                st.write(f"• LPIPS (perceptual): {lp:.3f}")

else:
    st.header("🤖 Detecção de Imagem Gerada por IA")
    img = st.file_uploader("Selecione uma imagem (jpg/png):", type=["jpg","png","jpeg"], key="ia_img")
    if img:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        st.image(img, caption="Imagem para análise", width=300)

        # carrega modelo local
        if "local_model" not in st.session_state:
            st.session_state["local_model"] = load_local_model(
                "models/best_deepfake_model.pth", device
            )
        model = st.session_state["local_model"]

        if st.button("Analisar Imagem"):
            p_real, p_fake = predict_local(load_df_image(img), model, device)
            st.write(f"• Prob. REAL: {p_real:.3f}")
            st.write(f"• Prob. IA-GERADA: {p_fake:.3f}")
            if p_fake > 0.5:
                st.error("⚠️ Imagem considerada GERADA por IA.")
            else:
                st.success("✔️ Imagem considerada REAL (não IA-gerada).")