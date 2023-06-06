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

img = Image.open("../images/dog1.jpg").convert("RGB")

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
model = model.to(device)

params = list(model.parameters())
##weight_softmax = np.squeeze(params[-2].data.numpy())
print(model)
model.features.denseblock4.denselayer32.conv2.register_forward_hook(hook)
model.eval()
model(input_batch)
conv_out = conv_forward_output[0].detach().numpy()
print(conv_out.shape)

mco = np.max(conv_out[0, 0, : , :])

img_plot = np.concatenate((conv_out[0, 0, : , :], conv_out[0, 1, : , :], conv_out[0, 2, : , :]), axis=1)
img_plot2 = np.concatenate((conv_out[0, 3, : , :], conv_out[0, 4, : , :], conv_out[0, 5, : , :]), axis=1)
img_plot3 = np.concatenate((conv_out[0, 6, : , :], conv_out[0, 7, : , :], conv_out[0, 8, : , :]), axis=1)
img_plot = np.concatenate((img_plot, img_plot2, img_plot3), axis=0)

plt.imshow(img_plot)
plt.jet()
plt.show()

#model.features.register_forward_hook(hook)

# Armazenando os pesos da última camada convolucional
#params = list(model.parameters())
#weight_softmax = np.squeeze(params[-2].data.numpy())

#model.eval()

# Realizando o processo de "forward"
#model(input_batch)
# Obtendo os pesos da última camada convolucional e realizando um "reshape" para facilitar os cálculos
#conv_out = conv_forward_output[0].detach().numpy()
#bz, nc, h, w = conv_out.shape
#conv_out = conv_out.reshape((nc, h*w))

# Calculando o produto escalar para cada classe
#cam1 = weight_softmax[0].dot(conv_out)
#cam2 = weight_softmax[1].dot(conv_out)
# Calculando a média desse produto escalar para cada classe
#cam1 = np.mean(cam1)
#cam2 = np.mean(cam2)
# Calculando o softmax entre os dois valores
#output = torch.tensor(np.array([cam1, cam2]))
#print(F.softmax(output).cpu().numpy().argmax())

# Criando o heatmap
#cam = weight_softmax[0].dot(conv_out)
#cam = cam.reshape(h, w)
#cam = cam - np.min(cam)
#cam_img = cam / np.max(cam)
#cam_img = np.uint8(255 * cam_img)
#cam_img = cv2.resize(cam_img, (256,256))
#plt.imshow(cam_img)
#plt.show()