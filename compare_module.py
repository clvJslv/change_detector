#compare_module.py
import torch
from PIL import Image
from torchvision import models, transforms
import lpips

def load_image(path, device, size=224):
    image = Image.open(path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    return preprocess(image).unsqueeze(0).to(device)

@torch.no_grad()
def get_resnet_embedding(img_tensor, device):
    resnet = models.resnet50(pretrained=True).to(device)
    for p in resnet.parameters(): p.requires_grad = False
    encoder = torch.nn.Sequential(*list(resnet.children())[:-1]).to(device)
    feats = encoder(img_tensor)
    return feats.view(feats.size(0), -1)

@torch.no_grad()
def cosine(u, v):
    u_norm = u / (u.norm(dim=1, keepdim=True) + 1e-8)
    v_norm = v / (v.norm(dim=1, keepdim=True) + 1e-8)
    return (u_norm * v_norm).sum(dim=1).item()

def euclidean(u, v):
    return torch.norm(u - v, p=2, dim=1).item()

@torch.no_grad()
def perceptual_score(pathA, pathB, device, net='vgg', size=256):
    loss_fn = lpips.LPIPS(net=net).to(device)
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    a = transform(Image.open(pathA).convert("RGB")).unsqueeze(0).to(device)
    b = transform(Image.open(pathB).convert("RGB")).unsqueeze(0).to(device)
    return loss_fn(a, b).item()