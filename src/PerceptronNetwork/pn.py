import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Classe responsável pela criação da rede Perceptron
class PerceptronNetwork:
    def __init__(self, 
                 n_input_features:int = 4,
                 random_weight_initialization:bool = False,
                 random_bias_initialization:bool = False,
                 n_classes:int = 1,
                 identifier:str = "") -> None:
        
        self.__identifier = identifier
        self.__weights = np.zeros((n_classes, n_input_features), dtype=np.float64)
        self.__bias = np.zeros((n_classes), dtype=np.float64) if n_classes > 1 else 0.0
        self.__n_classes = n_classes
        
        # Inicialização aleatória de pesos
        if random_weight_initialization:
            for i in range(self.__weights.shape[0]):
                for j in range(self.__weights.shape[1]):
                    self.__weights[i][j] = np.random.normal(scale=0.01)

        # Inicialização aleatória de Bias
        if random_bias_initialization:
            if n_classes == 1:
                self.__bias = np.random.normal(scale=0.01)
            else:
                for i in range(len(self.__bias)):
                        self.__bias[i] = np.random.normal(scale=0.01)

        pass
    
    # Função responsável pelo "Forward pass" da rede perceptron
    def forward(self, data_point):
        
        outputs = []
        outputs_exp = []

        for i in range(self.__weights.shape[0]):
            # Calculando o resultado da classificação
            neuron_output = self.__weights[i][:]*data_point
            neuron_output += self.__bias[i]
            neuron_output = np.sum(neuron_output)

            # Usando ReLU
            data_output = np.max([0, neuron_output])
            outputs.append(data_output)
            outputs_exp.append(np.exp(data_output))
        
        # Calculando Softmax
        softmax = []
        outputs_exp = np.sum(outputs_exp)
        for out in outputs:
            softmax.append(np.exp(out)/outputs_exp)

        return np.array(softmax)

    # Função responsável por realizar o treino da rede perceptron
    def train(self, dataset, epochs:int = 30, learning_rate:float = 0.01):
        epoch = 1

        # Necessário para realizar o processo de "checkpoint"
        best_f1 = None
        best_epoch = 0
        weights_checkpoint = None
        bias_checkpoint = None

        # Datasets
        training_data = dataset[0]
        validation_data = dataset[1]
        test_data = dataset[2]

        # Dados para os gráficos finais
        accuracy_data = []
        precision_data = []
        recall_data = []
        f1_data = []
        error_data = []

        # Dicionário para armazenar os resultados de cada epoch
        json_output = {}

        # Aqui a condição de parada é apenas o número de epochs
        # Isso é perfeitamente aceitável dado o uso dos "checkpoints"
        while epoch <= epochs:
            
            error = 0
            network_outputs = []
            expected_values = []

            # Etapa de treinamento
            for data in training_data:
                # Construindo o vetor de resultados esperados.
                # Formato: [(1,0), (1,0), ...]
                expected = np.zeros(self.__n_classes, dtype=np.float64)
                expected[data[1]] = 1

                # Obtendo saída da rede e calculando erro
                output = self.forward(data[0])
                data_error = expected - output

                for i in range(self.__weights.shape[0]):
                    for j in range(self.__weights.shape[1]):
                        # Atualizando os valores dos pesos baseado no erro
                        self.__weights[i][j] = self.__weights[i][j] + learning_rate*data_error[i]*data[0][j]

                        # Atualizando os valores de bias
                        if self.__n_classes > 1:
                            self.__bias[i] = self.__bias[i] + learning_rate*data_error[i]
                        else:
                            self.__bias = self.__bias[i] + learning_rate*data_error[i]
                
                # Calculando o erro médio quadrático
                error += np.mean(data_error**2)
            
            # Realizando teste de validação
            for data in validation_data:
                expected_values.append(data[1])
                network_outputs.append(np.argmax(self.forward(data[0])))
            
            # Calculando as métricas de: Precisão, recall, medida-f e acurácia pós validação
            precision = precision_score(expected_values, network_outputs, average="macro", zero_division=1)
            recall = recall_score(expected_values, network_outputs, average="macro", zero_division=1)
            f1 = f1_score(expected_values, network_outputs, average="macro", zero_division=1)
            accuracy = accuracy_score(expected_values, network_outputs)

            # Salvando dados para criar os gráficos e o arquivo .json de resultados
            precision_data.append(precision)
            recall_data.append(recall)
            f1_data.append(f1)
            accuracy_data.append(accuracy)
            error_data.append(error)
            json_output[epoch] = {
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "Error": error
            }

            # Processo de "checkpoint"
            if best_f1 is None:
                best_epoch = epoch
                best_f1 = f1
                weights_checkpoint = self.__weights.copy()
                bias_checkpoint = self.__bias.copy()

            # Salvamos a epoch que possuiu a melhor medida-f de todas as epochs no momento
            # Isso ajuda a evitar problemas como overfitting
            elif f1 > best_f1:
                best_epoch = epoch
                best_f1 = f1
                weights_checkpoint = self.__weights.copy()
                bias_checkpoint = self.__bias.copy()

            print("{}° época finalizada".format(epoch))

            epoch += 1
        
        print("\nTreinamento Finalizado!\nExecutando os testes...\n")
        time.sleep(1)

        # Definindo os pesos/bias como os melhores obtidos durante a fase de treinamento
        self.__weights = weights_checkpoint.copy()
        self.__bias = bias_checkpoint.copy()

        # Executando teste
        test_expected = []
        text_output = []
        for data in test_data:
            expected = data[1]
            output = self.forward(data[0])
            test_expected.append(expected)
            text_output.append(np.argmax(output))

            print("Esperado: {} | Saída da rede: {}".format(expected, np.argmax(output)))

        # Resultados finais
        print("\nAcurácia dos testes: {}".format(accuracy_score(test_expected, text_output)))
        print("Melhor época: {}".format(best_epoch))
        print("Métricas da melhor época:")
        print(json_output[best_epoch])
        print("Pesos: {}".format(self.__weights))
        print("Bias: {}".format(self.__bias))

        folder_name = datetime.now()
        folder_name = folder_name.strftime("%d-%m-%Y_%H-%M-%S")
        folder_name = folder_name + "_{}".format(self.__identifier)
        
        try:
            os.mkdir("../output/{}".format(folder_name))
        except FileExistsError:
            pass

        # Gerando gráficos
        x_axis = list(range(epochs))
        plt.figure(figsize=(12.8, 4.8))
        plt.plot(x_axis, precision_data, label="Precisão")
        plt.plot(x_axis, recall_data, label="Recall")
        plt.plot(x_axis, f1_data, label="Medida-F")
        plt.xlabel("Época")
        plt.title("Precisão, Recall e Medida F para {} épocas".format(epochs))
        plt.legend()
        plt.savefig("../output/{}/metrics.png".format(folder_name))
        plt.cla()

        x_axis = list(range(epochs))
        plt.figure(figsize=(12.8, 4.8))
        plt.plot(x_axis, accuracy_data, label="Acurácia")
        plt.xlabel("Época")
        plt.title("Acurácia para {} épocas".format(epochs))
        plt.legend()
        plt.savefig("../output/{}/accuracy.png".format(folder_name))
        plt.cla()

        x_axis = list(range(epochs))
        plt.figure(figsize=(12.8, 4.8))
        plt.plot(x_axis, error_data, label="Erro médio quadrático")
        plt.xlabel("Época")
        plt.title("Erro médio quadrático para {} épocas".format(epochs))
        plt.legend()
        plt.savefig("../output/{}/mean_square_error.png".format(folder_name))
        plt.cla()

        # Salvando os resultados de cada epoch em um arquivo .json
        with open("../output/{}/epochs.json".format(folder_name), "w") as json_file:
            json_string = json.dumps(json_output)
            json_file.write(json_string)

        print("\nGráficos e métricas de treino salvos em output/{}".format(folder_name))

    def get_weights(self):
        return self.__weights
    
    def get_bias(self):
        return self.__bias