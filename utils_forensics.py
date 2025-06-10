import torch
from PIL import Image
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

# Token embutido:
HF_TOKEN = "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"

# -----------------------------
# Funções para tampering geral
# -----------------------------
def load_forensics_model(model_name: str, device: torch.device):
    """
    Carrega o modelo de detecção de manipulação genérica e seu feature extractor.
    Usa HF_TOKEN embutido para autenticação.
    """
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_name, use_auth_token=HF_TOKEN
    )
    model = AutoModelForImageClassification.from_pretrained(
        model_name, use_auth_token=HF_TOKEN
    ).to(device)
    model.eval()
    return feature_extractor, model

@torch.no_grad()
def predict_tampering(feature_extractor, model, image: Image.Image, device: torch.device):
    """
    Dada uma PIL Image, retorna (p_autenticidade, p_manipulação) do modelo genérico.
    Classe 0 = autêntica; classe 1 = manipulada.
    """
    inputs = feature_extractor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)
    logits = outputs.logits            # shape (1,2)
    probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()  # [p0, p1]
    return float(probs[0]), float(probs[1])


# -----------------------------
# Funções para deepfake facial
# -----------------------------
def load_faceforgery_model(model_name: str, device: torch.device):
    """
    Carrega o modelo de DeepFake (face forgery) e seu feature extractor.
    Modelos possíveis:
      - "dima806/deepfake_vs_real_image_detection"
      - "prithivMLmods/deepfake-detector-model-v1"
      - "prithivMLmods/Deep-Fake-Detector-v2-Model"
      - "prithivMLmods/Deepfake-Detection-Exp-02-22"
      - "Wvolf/ViT_Deepfake_Detection"
    Usa HF_TOKEN embutido para autenticação.
    """
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_name, use_auth_token=HF_TOKEN
    )
    model = AutoModelForImageClassification.from_pretrained(
        model_name, use_auth_token=HF_TOKEN
    ).to(device)
    model.eval()
    return feature_extractor, model

@torch.no_grad()
def predict_deepfake(feature_extractor, model, image: Image.Image, device: torch.device):
    """
    Dada uma PIL Image, retorna (p_real, p_fake) do modelo de deepfake.
    Classe 0 = real; classe 1 = deepfake/manipulada.
    """
    inputs = feature_extractor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)
    logits = outputs.logits           # shape (1,2)
    probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    return float(probs[0]), float(probs[1])


# ------------------------------------------------
# Carregar imagem via PIL (aceita caminho ou buffer)
# ------------------------------------------------
def load_image(path_or_buffer):
    """
    Lê a imagem e converte para RGB.
    """
    return Image.open(path_or_buffer).convert("RGB")
