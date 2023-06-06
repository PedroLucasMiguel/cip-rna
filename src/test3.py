import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import warnings
import cv2
from PIL import Image
from torchvision import transforms
warnings.filterwarnings("ignore", category=UserWarning) 

class DenseNetCam(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.__base_model = model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet201', pretrained=True)
        self.features = self.__base_model.features

    def forward(self, x):
        # Alimentando a rede com a imagem até a última camada convolucional
        out = self.features(x)
        # Camada de GAP
        out = torch.mean(x, dim=(2, 3))
        out = nn.Softmax(out)
        return out
    
if __name__ == "__main__":
    device = "cpu"
    print(f"Using {device}")

    IMG_NAME = "../images/dog1.jpg"

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
    model = DenseNetCam()
    model = model.to(device)

    model.eval()
    model(input_batch)

    conv_forward_output = []

    def hook(model, input, output):
        conv_forward_output.append(output)

    model.features.register_forward_hook(hook)

    # Armazenando os pesos da última camada convolucional
    params = list(model.parameters())
    weight_softmax = np.squeeze(params[0].data.numpy())

    # Obtendo os pesos da última camada convolucional e realizando um "reshape"
    conv_out = conv_forward_output[0].detach().numpy()
    bz, nc, h, w = conv_out.shape
    conv_out = conv_out.reshape((nc, h*w))
    # Estou acessando a posição "1" do vetor "weight_softmax" pois é referente a classe
    # de cachorro
    cam = weight_softmax[1].dot(conv_out)
