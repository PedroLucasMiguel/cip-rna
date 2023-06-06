import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import warnings
import cv2
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=UserWarning) 

conv_forward_output = []

def hook(model, input, output):
    conv_forward_output.append(output)

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

# Construindo e carregando o treinamento do modelo
model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet201', pretrained=True)
model.classifier = nn.Linear(in_features=1920, out_features=2, bias=True)
model.load_state_dict(torch.load("../DenseNet-Checkpoints/densenet_cat_dog.pth"))
print(model)
model = model.to(device)

model.features.register_forward_hook(hook)

# Armazenando os pesos da última camada convolucional
params = list(model.parameters())
weight_softmax = np.squeeze(params[-2].data.numpy())

model.eval()

# Realizando o processo de "forward"
model(input_batch)

# Obtendo os pesos da última camada convolucional e realizando um "reshape" para facilitar os cálculos
conv_out = conv_forward_output[0].detach().numpy()
bz, nc, h, w = conv_out.shape
conv_out = conv_out.reshape((nc, h*w))

# Calculando o produto escalar para cada classe
cam1 = weight_softmax[0].dot(conv_out)
cam2 = weight_softmax[1].dot(conv_out)
# Calculando a média desse produto escalar para cada classe
cam1 = np.mean(cam1)
cam2 = np.mean(cam2)
# Calculando o softmax entre os dois valores
output = torch.tensor(np.array([cam1, cam2]))
print(F.softmax(output).cpu().numpy().argmax())



cam = weight_softmax[1].dot(conv_out)

# Criando o heatmap
img = cv2.imread(IMG_NAME)
cam = cam.reshape(h, w)
cam = cam - np.min(cam)
cam_img = cam / np.max(cam)
cam_img = np.uint8(255 * cam_img)
cam_img = cv2.resize(cam_img, (img.shape[1],img.shape[0]))
cam_img = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)
superimposed_img = cam_img * 0.4 + img
cv2.imwrite("../output/gradient.jpg", cam_img)
cv2.imwrite("../output/map.jpg", superimposed_img)
plt.imshow(cam_img)
plt.show()