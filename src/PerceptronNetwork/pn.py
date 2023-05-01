import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class PerceptronNetwork:
    def __init__(self, 
                 n_input_features:int = 4,
                 random_weight_initialization:bool = False,
                 random_bias_initialization:bool = False,
                 n_classes:int = 1) -> None:
        
        self.__weights = np.zeros((n_classes, n_input_features), dtype=np.float64)
        self.__bias = np.zeros((n_classes), dtype=np.float64) if n_classes > 1 else 0.0
        self.__n_classes = n_classes

        # TODO - Precisa revisar
        if random_weight_initialization:
            for i in range(self.__weights.shape[0]):
                for j in range(self.__weights.shape[1]):
                    self.__weights[i][j] = np.random.normal(scale=0.01)

        # TODO - Precisa revisar
        if random_bias_initialization:
            if n_classes == 1:
                self.__bias = np.random.normal(scale=0.01)
            else:
                for i in range(len(self.__bias)):
                        self.__bias[i] = np.random.normal(scale=0.01)

        pass

    def forward(self, data_point):
        
        outputs = []
        outputs_exp = []

        for i in range(self.__weights.shape[0]):
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

    def train(self, dataset, epochs:int = 30, learning_rate:float = 0.01):
        epoch = 1
        error = 1

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
            
            # Realizando validação
            for data in validation_data:
                expected_values.append(data[1])
                network_outputs.append(np.argmax(self.forward(data[0])))
            
            # Calculando as métricas de: Precisão, recall, medida-f e acurácia
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
            # Salvamos a epoch que possuiu a melhor medida-f de todas as epochhs no momento
            # Isso ajuda a evitar problemas como overfitting
            elif f1 > best_f1:
                best_epoch = epoch
                best_f1 = f1
                weights_checkpoint = self.__weights.copy()
                bias_checkpoint = self.__bias.copy()

            print("Finished: {}° epoch".format(epoch))

            epoch += 1
        
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

            print("Expected: {} | Output: {}".format(expected, np.argmax(output)))

        # Resultados finais
        print("Test accuracy: {}".format(accuracy_score(test_expected, text_output)))
        print("Best epoch: {}".format(best_epoch))
        print(json_output[best_epoch])

        # Gerando gráficos
        x_axis = list(range(epochs))
        plt.figure(figsize=(12.8, 4.8))
        plt.plot(x_axis, precision_data, label="Precisão")
        plt.plot(x_axis, recall_data, label="Recall")
        plt.plot(x_axis, f1_data, label="Medida F")
        plt.xlabel("Epoch")
        plt.title("Precisão, Recall e Medida F para {} epochs".format(epochs))
        plt.legend()
        plt.savefig("../output/metrics.png")
        plt.cla()

        x_axis = list(range(epochs))
        plt.figure(figsize=(12.8, 4.8))
        plt.plot(x_axis, accuracy_data, label="Acurácia")
        plt.xlabel("Epoch")
        plt.title("Acurácia para {} epochs".format(epochs))
        plt.legend()
        plt.savefig("../output/accuracy.png")
        plt.cla()

        x_axis = list(range(epochs))
        plt.figure(figsize=(12.8, 4.8))
        plt.plot(x_axis, error_data, label="Erro médio quadrático")
        plt.xlabel("Epoch")
        plt.title("Erro médio quadrático para {} epochs".format(epochs))
        plt.legend()
        plt.savefig("../output/meansquareerror.png")
        plt.cla()

        # Salvando os resultados de cada epoch em um arquivo .json
        with open("../output/epochs.json", "w") as json_file:
            json_string = json.dumps(json_output)
            json_file.write(json_string)

    def get_weights(self):
        return self.__weights
    
    def get_bias(self):
        return self.__bias

    def __str__(self) -> str:
        return "Hello World!"