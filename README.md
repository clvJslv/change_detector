# Chage Detector  & Detect IA-Generated Images

Ferramenta em **PyTorch** + **Streamlit** para:
1. **Comparar** duas imagens (A × B) e decidir se são iguais ou diferentes.
2. **Detectar** se uma única imagem foi **gerada por IA** ou é **real**, usando modelo finetunado.


Descrição do Projeto
Este projeto tem como objetivo fornecer duas funcionalidades principais para análise de imagens:

Comparação de Duas Imagens
Permite ao usuário carregar duas imagens (A e B) e verificar, de forma intuitiva, se elas são essencialmente iguais (mesmo conteúdo, pequenas rotações ou filtros) ou diferentes. Para isso, combinamos:

ResNet50 Pré-treinada (PyTorch)
Extraímos embeddings de 2 048 dimensões do penúltimo bloco da ResNet50 para cada imagem e medimos:

Similaridade Cosine (1.0 = idêntico; 0 = totalmente diferente)

Distância Euclidiana (0 = idêntico; quanto maior, mais diferente)

LPIPS (Learned Perceptual Image Patch Similarity)
Métrica perceptual baseada em VGG/AlexNet para captar diferenças de textura, cor e “aparência” que o simples cosine não vê.
O usuário pode optar por ativar ou desativar essa etapa.

Sliders amigáveis permitem ao leigo ajustar “quão rigoroso” o comparador deve ser para considerar imagens iguais, e “quão diferentes” para marcar alterações.

Detecção de Imagem Gerada por IA
Com foco em rostos ou qualquer imagem, fornecemos um detector que diz se a imagem foi gerada por inteligência artificial (ou deepfake) ou é real. Para isso:

Fine-tuning de EfficientNet-B0
Utilizamos o backbone EfficientNet-B0 (via timm), substituímos a última camada por uma saída binária (“real” vs. “IA-gerada”) e treinamos em um dataset próprio de imagens reais e geradas por IA (cerca de 3 000 amostras).

Pipeline de Treino

Data Augmentation: cortes aleatórios, flips, ajustes de cor, normalização (Albumentations).

Divisão Automática: 80 % treino / 20 % validação (“train_test_split”).

Batch size 16, 10 épocas, AdamW (lr=1e-4).

Salvamos o melhor checkpoint em models/best_deepfake_model.pth (baseado em F1-score de validação).

Inferência Local
No app, ao carregar uma imagem, usamos o checkpoint salvo para inferir (softmax) e exibir as probabilidades de “real” vs. “IA-gerada”, classificando com threshold 0.5.

Ferramentas e Bibliotecas Utilizadas
PyTorch + torchvision
Framework principal de deep learning, ResNet50, DataLoader e treinamentos.

timm
Biblioteca para modelos pré-implementados (EfficientNet-B0).

Albumentations
Para augmentations avançados (RandomResizedCrop, ColorJitter, flips).

LPIPS
Métrica de similaridade perceptual.

Transformers (Hugging Face)
(Em versões anteriores) para carregar modelos de forense genérico; atualmente removido do fluxo principal.

Streamlit
Interface web simples, interativa, sem necessidade de front-end customizado.

scikit-learn
Métricas de acurácia, F1-score e divisão de dados.

Resultados Esperados
Comparação de Imagens:

Usuário vê, de forma visual e num slider, se duas fotos são “iguais” ou “diferentes”, com métricas claras sem jargões técnicos.

Detecção IA-Gerada:

Para cada imagem única, recebe probabilidades e um veredito simples (“⚠️ IA-Gerada” ou “✔️ Real”), usando um modelo treinado especificamente no seu próprio dataset.

Com isso, você tem uma ferramenta completa, modularizada, que une visão computacional clássica (ResNet) e deep learning customizado (EfficientNet-B0) para tarefas práticas de comparação e detecção de geração artificial.
---

## 📁 Estrutura
change_detector/
├── compare_module.py # Funções de comparação A×B (ResNet50 + LPIPS)
├── deepfake_module.py # Carregamento / inferência do modelo finetunado local
├── app.py # Interface Streamlit
├── train_deepfake.py # Script de fine-tuning (EfficientNet-B0)
├── models/ # model.pth final (ignorado pelo Git)
│ └── best_deepfake_model.pth
├── requirements.txt
├── .gitignore
└── README.md

## 🔧 Instalação

1. **Clone** este repositório:
   git clone <seu-repo-url>
   cd change_detector

2. Crie e ative um ambiente virtual:
   python -m venv venv
   source venv/bin/activate    # Linux/macOS
   venv\Scripts\activate       # Windows

3. Instale as dependências:
   pip install -r requirements.txt

4. Inicie a interface
   streamlit run app.py 

5. Usando a Aplicação
Abra o navegador em http://localhost:8501 e escolha:

Comparar Imagens

Faça upload das duas imagens.

Ajuste os sliders de rigor/sensibilidade.

Clique em “Executar Comparação” e veja Cosine, Euclid e (opcional) LPIPS.

Detectar IA-Gerada

Faça upload de uma única imagem de rosto/objeto.

Clique em “Analisar Imagem” e veja a probabilidade de ser real vs. IA-gerada.



📖 Licença & Créditos
Frameworks: PyTorch, Albumentations, timm, Streamlit

Modelos: EfficientNet-B0 (via timm)

Métrica perceptual: LPIPS

