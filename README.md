# Change Detector & IA-Generated Image Detector

Ferramenta em **PyTorch** + **Streamlit** para duas tarefas:

1. **Comparar duas imagens** (A × B) e decidir se são iguais ou diferentes.  
2. **Detectar** se uma única imagem foi **gerada por IA** ou é **real**, usando um modelo finetunado.

---

## 📖 Descrição do Projeto

### 1. Comparação de Duas Imagens

Permite ao usuário carregar duas imagens e, de forma intuitiva, verificar:

- **Similaridade Cosine** (1.0 = idêntico; 0 = totalmente diferente),  
- **Distância Euclidiana** (0 = idêntico; quanto maior, mais diferente),  
- **LPIPS** (avaliação perceptual opcional para capturar diferenças de textura e cor).

O usuário inclui sliders amigáveis para ajustar:

- *Rigor* mínimo para considerar “iguais”,  
- *Nível de diferença* para considerar “alteradas”.

---

### 2. Detecção de Imagem Gerada por IA

Analisa **uma única imagem** e informa a probabilidade de ser:

- **Real** (fotografia sem geração artificial),  
- **IA-Gerada** (deepfake ou sintetizada por modelo de geração).

Baseado em um **EfficientNet-B0** finetunado no script de treino:

- **Data Augmentation** (Albumentations)  
- **Divisão Automática**: 80 % treino / 20 % validação  
- **Batch size** 16, **10 épocas**, **AdamW (lr=1e-4)**  
- **Checkpoint** salvo em `models/best_deepfake_model.pth`
- o modelo foi  treinado atraves do dataset encontrado no kaggle: https://www.kaggle.com/datasets/saurabhbagchi/deepfake-image-detection?select=Sample_fake_images

Na interface, basta fazer upload da imagem para ver “Prob. REAL” vs. “Prob. IA-Gerada” e um veredito simples.

---

## 🔧 Estrutura do Repositório

change_detector/
├── compare_module.py # Compara A×B (ResNet50 + LPIPS)
├── deepfake_module.py # Inferência local do modelo finetunado
├── app.py # Interface Streamlit
├── train_deepfake.py # Script de fine-tuning (EfficientNet-B0)
├── models/ # Checkpoint final (ignored by Git)
│ └── best_deepfake_model.pth
├── requirements.txt
├── .gitignore
└── README.md

## 🚀 Instalação e Uso

1. **Clone** o repositório e entre na pasta:
   ```bash
   git clone <seu-repo-url>
   cd change_detector
   
2. **Crie e ative um ambiente virtual:**
   python -m venv venv
   # Linux/macOS
   source venv/bin/activate
   # Windows
   venv\Scripts\activate
   
4. **Instrtale as Dependencias:
   pip install -r requirements.txt
   
6. Execute a interface:
   streamlit run app.py
   
▶️ Como Usar
**Comparar Imagens**

1. Selecione “Comparar Imagens”.
2. Faça upload das imagens A e B.
3. Ajuste os sliders:
  -Nível de rigor (quanto menor, menos rigor).
  -Limiar de diferença (quanto maior, mais tolerante).

4. Clique em Executar Comparação.
5. Veja os valores de Cosine, Euclid e, se selecionado, LPIPS, e o resultado (IGUAIS / DIFERENTES).

**Detectar IA-Gerada**

1. Selecione “Detectar IA-Gerada”.
2. Faça upload de uma imagem (rosto, cenário, objeto).
3. Clique em Analisar Imagem.
4. Veja as probabilidades de REAL vs. IA-Gerada e o veredito.

📚 Ferramentas e Bibliotecas
-PyTorch & torchvision
-tmm (EfficientNet-B0)
-Albumentations (Data Augmentation)
-LPIPS (Perceptual Similarity)
-Streamlit (UI)
-scikit-learn (métricas)
-NumPy, Pillow
   
