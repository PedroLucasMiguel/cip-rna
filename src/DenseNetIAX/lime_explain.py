
import torch
import torch.nn as nn
import numpy as np
import cv2
import warnings
import torch.nn.functional as F
import os
from torchvision import transforms
from lime import lime_image
from PIL import Image
from skimage.segmentation import mark_boundaries
from matplotlib import pyplot as plt
warnings.filterwarnings("ignore", category=UserWarning) 

# Função responsável por obter a explicação LIME de uma determinada imagem
def get_lime(img:str):
    # Criando a instância do Lime
    explainer = lime_image.LimeImageExplainer()

    IMG_NAME = img

    # Definindo processos de pré-processamento
    def get_pil_transform(): 
        transf = transforms.Compose([
            transforms.Resize((224, 224)),
        ])    

        return transf

    def get_preprocess_transform():
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])     
        transf = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ]) 
        return transf

    pill_transf = get_pil_transform()
    preprocess_transform = get_preprocess_transform()

    device = "cuda"
    print(f"Usando {device}")

    # Função responsável por ler a imagem em um formato que o LIME consegue trabalhar
    def get_image(path):
        with open(os.path.abspath(path), 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB') 
            
    img = get_image(IMG_NAME)

    # Construindo e carregando o treinamento do modelo
    model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet201', pretrained=True)
    model.classifier = nn.Linear(in_features=1920, out_features=2, bias=True)
    model.load_state_dict(torch.load("../DenseNet-Checkpoints/densenet_cat_dog.pth"))
    model = nn.DataParallel(model)
    model = model.to(device)

    # Função de "classificação"
    # É necessária para o LIME
    def classify_func(img):
        model.eval()
        batch = torch.stack(tuple(preprocess_transform(i) for i in img), dim=0)
        batch = batch.to(device)
        outputs = model(batch)

        return F.softmax(outputs, dim=1).detach().cpu().numpy()

    # Criando a imagem com a explicação e apresentando ela em um plot
    explanation = explainer.explain_instance(np.array(pill_transf(img)), classify_func, top_labels=1, hide_color=0, num_samples=1000)
    print("\nClassificação do modelo: {}\n".format("Gato" if explanation.top_labels[0] == 0 else "Cachorro"))
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
    img_boundry1 = mark_boundaries(temp/255.0, mask)
    o_img = np.array(img)
    img_boundry1 = cv2.resize(img_boundry1, dsize=(o_img.shape[1], o_img.shape[0]), interpolation=cv2.INTER_CUBIC)
    plt.imshow(img_boundry1)
    plt.show()