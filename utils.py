# utils.py

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import lpips

# -------------------------------------------------------------------
# 1) Carrega e pré-processa uma imagem para ResNet50 (224×224, ImageNet)
# -------------------------------------------------------------------
def load_image_resnet(path_or_buffer, device, img_size=224):
    """
    Lê uma imagem (RGB) de path_or_buffer (pode ser arquivo ou buffer),
    redimensiona para (img_size, img_size), normaliza segundo ImageNet
    e retorna tensor 1×3×HxW no device.
    """
    preprocess = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # padrões ImageNet
            std=[0.229, 0.224, 0.225]
        )
    ])
    image = Image.open(path_or_buffer).convert("RGB")
    img_tensor = preprocess(image).unsqueeze(0).to(device)  # shape (1,3,H,W)
    return img_tensor


# --------------------------------------------------------
# 2) Carrega encoder ResNet50 (pré-treinado; sem a última FC)
# --------------------------------------------------------
@torch.no_grad()
def get_resnet50_encoder(device):
    """
    Carrega ResNet50 (pretrained on ImageNet), remove a última camada FC
    e retorna apenas o encoder (até avgpool). Modo eval.
    """
    resnet = models.resnet50(pretrained=True).to(device)
    for param in resnet.parameters():
        param.requires_grad = False
    modules = list(resnet.children())[:-1]  # remove o último bloco (fc)
    encoder = nn.Sequential(*modules).to(device)
    encoder.eval()
    return encoder


# ------------------------------------------------------------
# 3) Extrai embedding (1×2048) de um tensor pré-processado (1×3×HxW)
# ------------------------------------------------------------
@torch.no_grad()
def extract_resnet_embedding(encoder, img_tensor):
    """
    Recebe encoder (ResNet50 sem última FC) e um tensor de imagem 1×3×HxW,
    retorna vetor 1×2048.
    """
    feats = encoder(img_tensor)            # (1, 2048, 1, 1)
    feats = feats.view(feats.size(0), -1)  # (1, 2048)
    return feats  # tensor (1,2048)


# --------------------------------------------------------
# 4) Carrega modelo LPIPS (VGG/Alex/Squeeze) e define função
# --------------------------------------------------------
@torch.no_grad()
def get_lpips_model(device, net_type='vgg'):
    """
    Carrega o modelo LPIPS (perceptual similarity) com backbone net_type,
    envia para device e retorna o objeto LPIPS em modo eval.
    net_type pode ser: 'vgg', 'alex' ou 'squeeze'.
    """
    loss_fn = lpips.LPIPS(net=net_type).to(device)
    loss_fn.eval()
    return loss_fn


def calculate_lpips_score(lpips_model, imgA_buffer, imgB_buffer, device, img_size=256):
    """
    Dadas duas imagens (RGB), redimensiona ambas para (img_size, img_size),
    normaliza para [-1,1] (requisito do LPIPS) e retorna o score perceptual.
    """
    preprocess = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        # LPIPS requer entrada no intervalo [-1,1]
    ])
    imgA = Image.open(imgA_buffer).convert("RGB")
    imgB = Image.open(imgB_buffer).convert("RGB")
    imgA_t = preprocess(imgA).unsqueeze(0).to(device)  # (1,3,H,W)
    imgB_t = preprocess(imgB).unsqueeze(0).to(device)
    with torch.no_grad():
        score = lpips_model(imgA_t, imgB_t)  # retorna tensor (1,1,1,1)
    return score.item()  # float


# ------------------------------------------------
# 5) Métricas de Similaridade/Distância (Embedding)
# ------------------------------------------------
def cosine_similarity(vec1, vec2):
    """
    Recebe dois tensores (1,C) e retorna a similaridade coseno (escala 0..1).
    """
    vec1_norm = vec1 / (vec1.norm(dim=1, keepdim=True) + 1e-8)
    vec2_norm = vec2 / (vec2.norm(dim=1, keepdim=True) + 1e-8)
    sim = (vec1_norm * vec2_norm).sum(dim=1)
    return sim.item()


def euclidean_distance(vec1, vec2):
    """
    Recebe dois tensores (1,C) e retorna a distância euclidiana.
    """
    dist = torch.norm(vec1 - vec2, p=2, dim=1)
    return dist.item()
