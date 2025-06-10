# deepfake_module.py

import torch
from PIL import Image
from torchvision import transforms
import timm

def load_local_model(checkpoint_path: str, device: torch.device):
    """
    Carrega o modelo EfficientNet-B0 treinado localmente.
    """
    # Mesma arquitetura usada no treino
    model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=2)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model

def predict_local(image: Image.Image, model, device: torch.device):
    """
    Dada uma PIL Image, retorna (p_real, p_fake) usando o modelo local.
    """
    tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])
    x = tf(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    return float(probs[0]), float(probs[1])  # [p_real, p_fake]

def load_image(path_or_buffer):
    return Image.open(path_or_buffer).convert("RGB")
