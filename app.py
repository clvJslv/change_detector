import streamlit as st
import torch

from compare_module import (
    load_image as load_cmp_image,
    get_resnet_embedding, cosine, euclidean, perceptual_score
)
from deepfake_module import (
    load_local_model, predict_local, load_image
)

st.set_page_config(page_title="Detector Modular", layout="wide")
mode = st.radio("Modo:", ["Detectar Geração por IA"])

if mode == "Detectar Geração por IA":
    st.header("Verificação de Imagem Gerada por IA")
    img = st.file_uploader("Selecione uma imagem para análise", type=["jpg","png","jpeg"], key="ia_img")
    if img:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        st.image(img, caption="Imagem para verificação", use_column_width=True)

        # Carrega modelo finetunado localmente
        if "local_model" not in st.session_state:
            st.session_state["local_model"] = load_local_model(
                "models/best_deepfake_model.pth", device
            )
        model = st.session_state["local_model"]

        if st.button("Analisar Imagem"):
            p_real, p_fake = predict_local(load_image(img), model, device)
            st.write(f"Probabilidade de imagem REAL: {p_real:.3f}")
            st.write(f"Probabilidade de imagem GERADA por IA: {p_fake:.3f}")

            if p_fake > 0.5:
                st.error("⚠️ Imagem considerada GERADA por IA")
            else:
                st.success("✔️ Imagem considerada REAL / não gerada por IA")