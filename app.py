# app.py

import streamlit as st
import torch
from PIL import Image
import matplotlib.pyplot as plt

from utils import (
    load_image_resnet, get_resnet50_encoder, extract_resnet_embedding,
    get_lpips_model, calculate_lpips_score,
    cosine_similarity, euclidean_distance
)

st.set_page_config(
    page_title="🔍 Change Detector",
    layout="centered"
)

# ========================================================
# 1) Cabeçalho e descrição
# ========================================================
st.title("🔍 Detector de Mudanças entre Duas Imagens")
st.markdown(
    """
    Este app utiliza **ResNet50 embeddings** e a métrica **LPIPS (Learned Perceptual Image Patch Similarity)**
    para verificar se duas imagens são essencialmente idênticas ou apresentam alguma alteração perceptual
    (filtro de cor, rotação, brilho, etc.). Ajuste os limiares conforme sua necessidade.
    """
)

st.markdown("---")

# ========================================================
# 2) Painel de Upload de Imagens
# ========================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("Imagem A (Base)")
    uploaded_A = st.file_uploader(
        label="Faça upload de Imagem A (original)", type=["png", "jpg", "jpeg"], key="imgA"
    )
with col2:
    st.subheader("Imagem B (Suspeita de Alteração)")
    uploaded_B = st.file_uploader(
        label="Faça upload de Imagem B (a verificar)", type=["png", "jpg", "jpeg"], key="imgB"
    )

if not uploaded_A or not uploaded_B:
    st.info("👉 Faça upload de ambas as imagens para prosseguir.")
    st.stop()

# Exiba as duas imagens lado a lado
imgA_pil = Image.open(uploaded_A).convert("RGB")
imgB_pil = Image.open(uploaded_B).convert("RGB")

st.markdown("**Pré-visualização das Imagens:**")
col1, col2 = st.columns(2)
with col1:
    st.image(imgA_pil, caption="Imagem A (Original)", use_column_width=True)
with col2:
    st.image(imgB_pil, caption="Imagem B (Suspeita)", use_column_width=True)

st.markdown("---")

# ========================================================
# 3) Seção de Parâmetros / Limiar
# ========================================================
st.subheader("⚙️ Ajuste de Parâmetros")

with st.expander("Parâmetros ResNet50", expanded=True):
    resnet_size = st.slider(
        label="Tamanho de entrada (ResNet50)", min_value=128, max_value=512, value=224, step=32
    )
    cosine_threshold = st.slider(
        label="Limiar Cosine Similarity (ResNet) 0.0..1.0",
        min_value=0.0, max_value=1.0, value=0.90, step=0.01
    )
    euclid_threshold = st.slider(
        label="Limiar Euclidean Distance (ResNet)",
        min_value=0.0, max_value=5.0, value=1.0, step=0.1
    )

with st.expander("Parâmetros LPIPS (Opcional)", expanded=False):
    use_lpips = st.checkbox("Usar LPIPS (Perceptual Similarity)", value=True)
    lpips_net = st.selectbox(
        label="Backbone LPIPS",
        options=["vgg", "alex", "squeeze"],
        index=0,
        help="Arquitetura de base para LPIPS"
    )
    lpips_size = st.slider(
        label="Tamanho de entrada LPIPS", min_value=128, max_value=512, value=256, step=32
    )
    lpips_threshold = st.slider(
        label="Limiar LPIPS (0.0..1.0)",
        min_value=0.00, max_value=1.00, value=0.10, step=0.01
    )

st.markdown("---")

# ========================================================
# 4) Botão de “Detectar”
# ========================================================
if st.button("🚀 Detectar Mudança"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.write(f"**Executando no device:** `{device}`")

    # -------------------------------
    # 4.1) Preparar imagens para ResNet50
    # -------------------------------
    imgA_resnet = load_image_resnet(uploaded_A, device, img_size=resnet_size)
    imgB_resnet = load_image_resnet(uploaded_B, device, img_size=resnet_size)

    # -------------------------------
    # 4.2) Extrair embeddings ResNet50
    # -------------------------------
    encoder = get_resnet50_encoder(device)
    embA = extract_resnet_embedding(encoder, imgA_resnet)  # (1,2048)
    embB = extract_resnet_embedding(encoder, imgB_resnet)  # (1,2048)

    # -------------------------------
    # 4.3) Calcular métricas ResNet
    # -------------------------------
    cos_sim = cosine_similarity(embA, embB)
    euc_dist = euclidean_distance(embA, embB)

    st.subheader("📊 Métricas ResNet50")
    st.write(f"- **Cosine Similarity**: `{cos_sim:.4f}` (1.0 = idêntico, 0 = ortogonal)")
    st.write(f"- **Euclidean Distance**: `{euc_dist:.4f}` (0.0 = idêntico)")

    # Decisão ResNet
    resnet_altered = False
    if (cos_sim < cosine_threshold) or (euc_dist > euclid_threshold):
        st.error(
            f"**[ResNet50]** Cosine ({cos_sim:.4f}) < {cosine_threshold} "
            f"ou Euclid ({euc_dist:.4f}) > {euclid_threshold} → IMAGEM ALTERADA"
        )
        resnet_altered = True
    else:
        st.success(
            f"**[ResNet50]** Cosine ({cos_sim:.4f}) ≥ {cosine_threshold} "
            f"e Euclid ({euc_dist:.4f}) ≤ {euclid_threshold} → NÃO ALTERADA"
        )

    # -------------------------------
    # 4.4) Calcular LPIPS (se habilitado)
    # -------------------------------
    if use_lpips:
        lpips_model = get_lpips_model(device, net_type=lpips_net)
        lpips_score = calculate_lpips_score(
            lpips_model,
            uploaded_A,
            uploaded_B,
            device,
            img_size=lpips_size
        )
        st.subheader("📊 Métrica LPIPS")
        st.write(f"- **LPIPS Score** (net={lpips_net}): `{lpips_score:.4f}` (0 = idêntico)")

        if lpips_score > lpips_threshold:
            st.error(f"**[LPIPS]** Score ({lpips_score:.4f}) > {lpips_threshold} → IMAGEM ALTERADA")
            lpips_altered = True
        else:
            st.success(f"**[LPIPS]** Score ({lpips_score:.4f}) ≤ {lpips_threshold} → NÃO ALTERADA")
            lpips_altered = False
    else:
        lpips_score = None
        lpips_altered = False

    # -------------------------------
    # 4.5) Resultado Final Consolidado
    # -------------------------------
    st.markdown("---")
    st.subheader("✅ Resultado Final")
    if resnet_altered or (use_lpips and lpips_altered):
        st.error("❌ As imagens foram consideradas **DIFERENTES** (ALTERADAS).")
    else:
        st.success("✔️ As imagens foram consideradas **IGUAIS** (NENHUMA ALTERAÇÃO SIGNIFICATIVA).")

    st.caption("A detecção baseia-se nas métricas ResNet50 e LPIPS conforme limiares definidos.")


# ========================================================
# 5) Footer / Créditos
# ========================================================
st.markdown("---")
st.markdown(
    """
    **Como Funciona Internamente**  
    1. **ResNet50 Embeddings:** extraímos um vetor de 2048-dim de cada imagem (pós-avgpool).  
       - Cosine Similarity alto (próximo a 1.0) indica imagens semelhantes.  
       - Euclidean Distance baixa (próximo a 0) indica imagens quase idênticas.  

    2. **LPIPS (Perceptual Similarity):** utiliza rede tipo VGG/AlexNet para medir diferenças perceptuais  
       (cores, texturas, filtros). Score 0.0 = idêntico. Quanto maior, mais diferente.  

    Ajuste os **limiares** acima para calibrar sensibilidade (rotação leve, leve filtro de cor, etc.).
    """
)

