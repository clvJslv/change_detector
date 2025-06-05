python - <<EOF
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
# Tenta baixar o feature extractor e o modelo:
feat = AutoFeatureExtractor.from_pretrained("prithivMLmods/open-deepfake-detection")
model = AutoModelForImageClassification.from_pretrained("prithivMLmods/open-deepfake-detection")
print("✅ Modelo prithivMLmods/open-deepfake-detection carregado com sucesso!")
EOF