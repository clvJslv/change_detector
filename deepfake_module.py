import torch
from PIL import Image
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

HF_TOKEN = "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"

SUPPORTED = [
    "prithivMLmods/Deep-Fake-Detector-v2-Model",
    "prithivMLmods/Deepfake-Detection-Exp-02-22",
    "aznasut/ai_vs_fake_image"
]

def load_deepfake_model(name, device):
    if name not in SUPPORTED:
        raise ValueError(f"Modelo {name} não suportado")
    fe = AutoFeatureExtractor.from_pretrained(name, use_auth_token=HF_TOKEN)
    model = AutoModelForImageClassification.from_pretrained(name, use_auth_token=HF_TOKEN).to(device)
    model.eval()
    return fe, model

@torch.no_grad()
def detect_deepfake(feature_extractor, model, img, device):
    inputs = feature_extractor(images=img, return_tensors="pt").to(device)
    out = model(**inputs).logits
    probs = torch.softmax(out, dim=-1)[0].cpu().numpy()
    return float(probs[0]), float(probs[1])  # [p_real, p_fake]


def load_image(path_or_buffer):
    return Image.open(path_or_buffer).convert("RGB")