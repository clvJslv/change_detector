import os
import streamlit as st
import torch
from PIL import Image

from utils import (
    load_image_resnet, get_resnet50_encoder, extract_resnet_embedding,
    get_lpips_model, calculate_lpips_score,
    cosine_similarity, euclidean_distance
)
from utils_forensics import (
    load_forensics_model, predict_tampering,
    load_faceforgery_model, predict_deepfake,
    load_image
)

# (Caso deseje usar o token também via variável, mas não necessário)
# os.environ["HUGGINGFACE_TOKEN"] = "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"

st.set_page_config(
    page_title="🔍 Change & Tampering Detector",
    layout="wide",
)

# --------------------------------------------------
# 1) Cabeçalho e Descrição Geral
# --------------------------------------------------
st.title("🔍 Detector de Mudança e Manipulação de Imagens")
st.markdown(
    """
    **Funcionalidades**:
    1. **Comparar Duas Imagens** (A × B) usando ResNet50 & LPIPS.
    2. **Detectar Manipulação em Uma Única Imagem** com modelos forenses ou de DeepFake.
    """
)

# --------------------------------------------------
# 2) Seletor de Modo
# --------------------------------------------------
mode = st.radio(
    "Escolha o modo de operação:",
    options=["Comparar Duas Imagens", "Detectar Manipulação de Uma Imagem"]
)

st.markdown("---")

# --------------------------------------------------
# 3) Modo 1: Comparar Duas Imagens
# --------------------------------------------------
if mode == "Comparar Duas Imagens":
    st.header("1️⃣ Comparar Duas Imagens (A × B)")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Imagem A (Base)")
        uploaded_A = st.file_uploader(
            label="Faça upload de Imagem A (original)",
            type=["png", "jpg", "jpeg"],
            key="imgA_cmp"
        )
    with col2:
        st.subheader("Imagem B (Suspeita de Alteração)")
        uploaded_B = st.file_uploader(
            label="Faça upload de Imagem B (a verificar)",
            type=["png", "jpg", "jpeg"],
            key="imgB_cmp"
        )

    if not uploaded_A or not uploaded_B:
        st.info("▶️ Carregue ambas as imagens para prosseguir.")
        st.stop()

    # Pré-visualização
    imgA_pil = Image.open(uploaded_A).convert("RGB")
    imgB_pil = Image.open(uploaded_B).convert("RGB")
    st.markdown("**Pré-visualização das Imagens:**")
    c1, c2 = st.columns(2)
    with c1:
        st.image(imgA_pil, caption="Imagem A (Original)", use_column_width=True)
    with c2:
        st.image(imgB_pil, caption="Imagem B (Suspeita)", use_column_width=True)

    st.markdown("---")
    st.subheader("⚙️ Ajuste de Parâmetros para Comparação")

    with st.expander("Parâmetros ResNet50", expanded=True):
        resnet_size = st.slider(
            label="Tamanho de entrada (ResNet50)",
            min_value=128, max_value=512, value=224, step=32
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
            label="Tamanho de entrada LPIPS",
            min_value=128, max_value=512, value=256, step=32
        )
        lpips_threshold = st.slider(
            label="Limiar LPIPS (0.0..1.0)",
            min_value=0.00, max_value=1.00, value=0.10, step=0.01
        )

    st.markdown("---")
    if st.button("🚀 Detectar Mudança nas Imagens"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        st.write(f"**Executando no device:** `{device}`")

        # 1) Preparar imagens para ResNet50
        imgA_resnet = load_image_resnet(uploaded_A, device, img_size=resnet_size)
        imgB_resnet = load_image_resnet(uploaded_B, device, img_size=resnet_size)

        # 2) Extrair embeddings ResNet50
        encoder = get_resnet50_encoder(device)
        embA = extract_resnet_embedding(encoder, imgA_resnet)  # (1,2048)
        embB = extract_resnet_embedding(encoder, imgB_resnet)  # (1,2048)

        # 3) Calcular métricas ResNet50
        cos_sim = cosine_similarity(embA, embB)
        euc_dist = euclidean_distance(embA, embB)

        st.subheader("📊 Métricas ResNet50")
        st.write(f"- **Cosine Similarity**: `{cos_sim:.4f}` (1.0 = idêntico, 0 = ortogonal)")
        st.write(f"- **Euclidean Distance**: `{euc_dist:.4f}` (0.0 = idêntico)")

        # Decisão ResNet50
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

        # 4) Calcular LPIPS (se habilitado)
        if use_lpips:
            lpips_model = get_lpips_model(device, net_type=lpips_net)
            lpips_score = calculate_lpips_score(
                lpips_model, uploaded_A, uploaded_B, device, img_size=lpips_size
            )
            st.subheader("📊 Métrica LPIPS")
            st.write(f"- **LPIPS Score** (net={lpips_net}): `{lpips_score:.4f}` (0 = idêntico)")

            lpips_altered = False
            if lpips_score > lpips_threshold:
                st.error(f"**[LPIPS]** Score ({lpips_score:.4f}) > {lpips_threshold} → IMAGEM ALTERADA")
                lpips_altered = True
            else:
                st.success(f"**[LPIPS]** Score ({lpips_score:.4f}) ≤ {lpips_threshold} → NÃO ALTERADA")
                lpips_altered = False
        else:
            lpips_score = None
            lpips_altered = False

        st.markdown("---")
        st.subheader("✅ Resultado Final (Comparação A×B)")
        if resnet_altered or (use_lpips and lpips_altered):
            st.error("❌ As imagens foram consideradas **DIFERENTES** (ALTERADAS).")
        else:
            st.success("✔️ As imagens foram consideradas **IGUAIS** (NENHUMA ALTERAÇÃO SIGNIFICATIVA).")

        st.caption("Dados baseados em métricas ResNet50 e LPIPS conforme limiares definidos.")

# --------------------------------------------------
# 4) Modo 2: Detectar Manipulação em Uma Única Imagem
# --------------------------------------------------
else:
    st.header("2️⃣ Detectar Manipulação em Uma Única Imagem (No-Reference)")

    uploaded_file = st.file_uploader(
        "Faça upload de uma imagem (jpg/png) para verificar",
        type=["png", "jpg", "jpeg"],
        key="img_forensic"
    )
    if not uploaded_file:
        st.info("▶️ Carregue uma imagem para prosseguir.")
        st.stop()

    # Pré-visualização
    image = load_image(uploaded_file)
    st.image(image, caption="Imagem a ser verificada", use_column_width=True)

    st.markdown("---")
    st.subheader("⚙️ Escolha do Modelo e Configurações")

    col1, col2 = st.columns(2)
    with col1:
        model_category = st.selectbox(
            "Categoria de Modelo Forense:",
            options=["Tampering Geral", "DeepFake Facial"]
        )
    with col2:
        device_option = st.selectbox(
            "Device para Inferência:",
            options=["cuda" if torch.cuda.is_available() else "cpu", "cpu"]
        )
    device = torch.device(device_option)

    # -------------------------------
    # 4.1) Definir lista de modelos conforme categoria
    # -------------------------------
    if model_category == "Tampering Geral":
        model_options = ["coastalfrost/clarifai-image-tampering-detection"]
    else:  # DeepFake Facial
        model_options = [
            "dima806/deepfake_vs_real_image_detection",
            "prithivMLmods/deepfake-detector-model-v1",
            "prithivMLmods/Deep-Fake-Detector-v2-Model",
            "prithivMLmods/Deepfake-Detection-Exp-02-22",
            "Wvolf/ViT_Deepfake_Detection"
        ]

    model_name = st.selectbox("Selecione o modelo:", options=model_options, index=0)

    # ---------------------------------------------------
    # 4.2) Carregar o modelo correspondente uma vez em session_state
    # ---------------------------------------------------
    if model_category == "Tampering Geral":
        key_extractor = "forensic_extractor"
        key_model = "forensic_model"
        key_loaded_name = "forensic_model_name"

        if (key_loaded_name not in st.session_state) or (st.session_state.get(key_loaded_name) != model_name):
            with st.spinner("Carregando modelo de Tampering Geral..."):
                feature_extractor, forensics_model = load_forensics_model(model_name, device)
                st.session_state[key_extractor] = feature_extractor
                st.session_state[key_model] = forensics_model
                st.session_state[key_loaded_name] = model_name
            st.success("✔️ Modelo de Tampering Geral carregado com sucesso!")
        else:
            feature_extractor = st.session_state[key_extractor]
            forensics_model = st.session_state[key_model]

    else:  # DeepFake Facial
        key_extractor = "face_extractor"
        key_model = "face_model"
        key_loaded_name = "face_model_name"

        if (key_loaded_name not in st.session_state) or (st.session_state.get(key_loaded_name) != model_name):
            with st.spinner("Carregando modelo DeepFake Facial..."):
                face_extractor, face_model = load_faceforgery_model(model_name, device)
                st.session_state[key_extractor] = face_extractor
                st.session_state[key_model] = face_model
                st.session_state[key_loaded_name] = model_name
            st.success("✔️ Modelo DeepFake Facial carregado com sucesso!")
        else:
            face_extractor = st.session_state[key_extractor]
            face_model = st.session_state[key_model]

    # ---------------------------------------------------
    # 4.3) Rodar inferência ao clicar no botão
    # ---------------------------------------------------
    if st.button("🚀 Detectar Manipulação/DeepFake"):
        with st.spinner("Analisando a imagem..."):
            if model_category == "Tampering Geral":
                p_auth, p_tamp = predict_tampering(feature_extractor, forensics_model, image, device)
                st.subheader("📈 Probabilidades de Tampering Geral")
                st.write(f"- **Probabilidade de SER autêntica:**  `{p_auth:.4f}`")
                st.write(f"- **Probabilidade de SER manipulada:** `{p_tamp:.4f}`")
                threshold = 0.5
                if p_tamp > threshold:
                    st.error(f"❌ A imagem é **provavelmente manipulada** (p_manipulação > {threshold}).")
                else:
                    st.success(f"✔️ A imagem é **aparentemente autêntica** (p_manipulação ≤ {threshold}).")

            else:  # DeepFake Facial
                p_real, p_fake = predict_deepfake(face_extractor, face_model, image, device)
                st.subheader("📈 Probabilidades DeepFake Facial")
                st.write(f"- **Probabilidade de SER real:**  `{p_real:.4f}`")
                st.write(f"- **Probabilidade de SER deepfake:** `{p_fake:.4f}`")
                threshold = 0.5
                if p_fake > threshold:
                    st.error(f"❌ A imagem é **provavelmente um DeepFake** (p_fake > {threshold}).")
                else:
                    st.success(f"✔️ A imagem é **aparentemente real** (p_fake ≤ {threshold}).")

        st.markdown("---")
        if model_category == "Tampering Geral":
            st.markdown(
                """
                **Como Funciona (Tampering Geral)**  
                - Modelo treinado em splicing, colagens e retoques de cena.  
                - Saída: [p_autêntica, p_manipulada].  
                """
            )
        else:
            st.markdown(
                """
                **Como Funciona (DeepFake Facial)**  
                - Modelos treinados para detectar deepfakes em rostos:
                  - dima806/deepfake_vs_real_image_detection  
                  - prithivMLmods/deepfake-detector-model-v1  
                  - prithivMLmods/Deep-Fake-Detector-v2-Model  
                  - prithivMLmods/Deepfake-Detection-Exp-02-22  
                  - Wvolf/ViT_Deepfake_Detection  
                - Saída: [p_real, p_deepfake].  
                """
            )
