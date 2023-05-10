import os
import json
import torch
import warnings
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from torchvision import transforms
from torchvision import datasets
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
warnings.filterwarnings("ignore", category=UserWarning) 

EPOCHS = 10
BATCH_SIZE = 256
LEARNING_RATE = 0.001

# Função responsável por treinar o modelo Densenet-201
def train():
    # Definindo as transformações que precisam ser feitas no conjunto de imagens
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Carregando o dataset a partir da pasta
    dataset = datasets.ImageFolder("../Datasets/Dog-cat/train", preprocess)

    # Criando o dataset com split 80/20
    dataset_train, dataset_validation = random_split(dataset, [0.8, 0.2])

    # Criando os "loaders" para o nosso conjunto de treino e validação
    trainloader = torch.utils.data.DataLoader(
                    dataset_train, 
                    batch_size=BATCH_SIZE)
    validationloader = torch.utils.data.DataLoader(
                    dataset_validation,
                    batch_size=BATCH_SIZE)

    # Utiliza GPU caso possível
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} for training")

    # Cria o modelo, reseta os pesos e define o dispositivo de execução
    model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet201', pretrained=True)

    # Congelando o treinamento das demais camadas que não sejam a última FC
    for params in model.parameters():
        params.requires_grad = False
    model.classifier = nn.Linear(in_features=1920, out_features=2, bias=True)

    model.to(device)

    # Definindo nossa função para o calculo de loss e o otimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adamax(model.parameters(), lr=LEARNING_RATE)

    # Melhor resultado da métrica F1 (utilizado no processo de checkpoint)
    best_f1 = 0
    best_f1_file_name = ""

    metrics_json = {}

    # Iniciando o processo de treinamento
    for epoch in range(0, EPOCHS):
        model.train()
        print(f"Executando época {epoch+1}")
            
        print("Treinando...")
        for i, data in enumerate(trainloader, 0):
            img, label = data
            img = img.to(device)
            label = label.to(device)
            model.train()
            pred = model(img)
            loss = criterion(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        accuracy = []
        precision = []
        recall = []
        f1 = []
        model.eval()

        print("Validando...")
        # Iniciando o processo de validação
        with torch.no_grad():
            for i, data in enumerate(validationloader, 0):
                img, label = data
                img = img.to(device)
                label = label.to(device)
                outputs = model(img)
                _, predicted = torch.max(outputs.data, 1)
                # Convertendo os arrays das labels e das previsões para uso em CPU
                label_cpu = label.cpu()
                predicted_cpu = predicted.cpu()
                # Calculando as métricas
                precision.append(precision_score(label_cpu, predicted_cpu))
                recall.append(recall_score(label_cpu, predicted_cpu))
                f1.append(f1_score(label_cpu, predicted_cpu))
                accuracy.append(accuracy_score(label_cpu, predicted_cpu))

        # Apresentando as métricas
        accuracy = np.mean(accuracy)
        precision = np.mean(precision)
        recall = np.mean(recall)
        f1 = np.mean(f1)
        print("\nResultados:")
        print(f"Acurácia da época {epoch+1} (%): {100.0 * accuracy}%")
        print(f"Precisão: {(precision)}")
        print(f"Recall: {recall}")
        print(f"Medida-f: {f1}")
            
        # Se o resultado possuir a melhor medida F de todo o processo, salve o treinamento
        if f1 > best_f1:
            if best_f1_file_name != "":
                os.remove(f"../DenseNet-Checkpoints/{best_f1_file_name}")

            print(f"\nSalvando o treinamento referente a época {epoch+1}")
            best_f1 = f1
            torch.save(model.state_dict(), f"../DenseNet-Checkpoints/e_{epoch+1}_savestate.pth")
            best_f1_file_name = f"e_{epoch+1}_savestate.pth"

        metrics_json[epoch+1] = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
        }
        print('--------------------------------')

    # Exporta as métricas em um arquivo .json
    with open("../output/metrics_densenet.json", "w") as json_file:
        json.dump(metrics_json, json_file)

    print("Métricas para cada época salvas em: /output/metrics_densenet.json")