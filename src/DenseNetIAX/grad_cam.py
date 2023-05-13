import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import warnings
import cv2
from PIL import Image
from torchvision import transforms
warnings.filterwarnings("ignore", category=UserWarning) 

# Modelo responsável por viabilizar a obtenção dos gradientes
# para construir o Grad-CAM
class DenseNetGradCam(nn.Module):
    def __init__(self, model, *args, **kwargs,) -> None:
        super().__init__(*args, **kwargs)
        self.model = model

        self.gradients = None

    def activations_hook(self, grad):
        self.gradients = grad

    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, x):
        return self.model.features(x)
    
    def forward(self, x):
        # 1° Passa a entrada pelas camadas convolucionais
        out = self.model.features(x)
        # 2° Passa a saída das camadas por uma camda de Relu e pooling
        out = F.relu(out, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        if out.requires_grad:
            # 3° Hook que salva o gradiente dessa saída quando backwards() é chamado
            h = out.register_hook(self.activations_hook)
        # 4° Termina de passar a entrada para o resto da rede
        out = torch.flatten(out, 1)
        out = self.model.classifier(out)
        return out

# Função responsável por obter o Grad-CAM de uma imagem
def get_grad_cam(img:str):
    
    IMG_NAME = img

    device = "cpu"
    print(f"Using {device}")

    img = Image.open(IMG_NAME).convert("RGB")

    # Transformações da imagem de entrada
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Pré-processando a imagem e transformando ela em um "mini-batch"
    img = preprocess(img)
    input_batch = img.unsqueeze(0)
    input_batch = input_batch.to(device)

    # Construindo e carregando o treinamento do modelo
    model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet201', pretrained=True)
    model.classifier = nn.Linear(in_features=1920, out_features=2, bias=True)
    model.load_state_dict(torch.load("../DenseNet-Checkpoints/densenet_cat_dog.pth"))
    model = DenseNetGradCam(model)
    model = model.to(device)

    model.eval()

    # Obtendo a classificação do modelo e calculando o gradiente da maior classe
    outputs = model(input_batch)
    class_to_backprop = F.softmax(outputs).detach().cpu().numpy()[0].argmax()
    print("\nClassificação do modelo: {}".format("Gato" if class_to_backprop == 0 else "Cachorro"))
    outputs[:, class_to_backprop].backward()

    # Obtendo informações dos gradienttes e construindo o "heatmap"
    gradients = model.get_activations_gradient()
    gradients = torch.mean(gradients, dim=[0, 2, 3])
    layer_output = model.get_activations(input_batch)

    for i in range(len(gradients)):
        layer_output[:, i, :, :] *= gradients[i]

    layer_output = layer_output[0, : , : , :]

    # Salvando imagens
    img = cv2.imread(IMG_NAME)

    heatmap = torch.mean(layer_output, dim=0).detach().numpy()
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0])) 
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img
    cv2.imwrite("../output/gradient.jpg", heatmap)
    final_img = np.concatenate((img, superimposed_img), axis=1)
    cv2.imwrite("../output/map.jpg", final_img)

    print("Imagens salvas em output/gradient.jpg e output/map.jpg\n")
