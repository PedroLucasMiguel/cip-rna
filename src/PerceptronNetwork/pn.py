import time
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class Activations:
    def ReLU(x):
        pass

    def sigmoid(x):
        pass

    def Tanh(x):
        pass

class PerceptronNetwork:
    def __init__(self, 
                 n_input_features:int = 4,
                 random_weight_initialization:bool = False,
                 random_bias_initialization:bool = False,
                 n_classes:int = 1) -> None:
        
        self.__weights = np.zeros((n_classes, n_input_features), dtype=np.float64)
        self.__bias = np.zeros((n_classes), dtype=np.float64) if n_classes > 1 else 0.0
        self.__n_classes = n_classes

        if random_weight_initialization:
            for i in range(self.__weights.shape[0]):
                for j in range(self.__weights.shape[1]):
                    self.__weights[i][j] = np.random.normal(scale=0.01)

        if random_bias_initialization:
            if n_classes == 1:
                self.__bias = np.random.normal(scale=0.01)
            else:
                for i in range(len(self.__bias)):
                        self.__bias[i] = np.random.normal(scale=0.01)

        pass

    def forward(self, data_point) -> int:
        
        outputs = []
        outputs_exp_sum = []
        for i in range(self.__weights.shape[0]):
            neuron_output = self.__weights[i]*data_point
            neuron_output += self.__bias[i]
            neuron_output = np.sum(neuron_output)

            # ReLU
            outputs.append(np.max([0, neuron_output]))

            outputs_exp_sum.append(np.exp(np.max([0, neuron_output])))

        # Softmax
        outputs_exp_sum = np.sum(outputs_exp_sum)

        softmax = []
        for out in outputs:
            softmax.append(np.exp(out)/outputs_exp_sum)

        return softmax

    def train(self, dataset, epochs:int = 30, learning_rate:float = 0.01):
        epoch = 1
        error = 1
        training_data = dataset[0]
        validation_data = dataset[1]
        test_data = dataset[2]

        while epoch <= epochs:
            error = 0
            network_outputs = []
            expected_values = []

            # Training pass
            for data in training_data:
                expected = np.zeros(self.__n_classes, dtype=np.float64)
                expected[data[1]] = 1

                output = self.forward(data[0])

                data_error = expected - output

                for i in range(self.__weights.shape[0]):
                    for j in range(self.__weights.shape[1]):
                        self.__weights[i][j] = self.__weights[i][j] + learning_rate*data_error[i]*data[0][j]
                        if self.__n_classes > 1:
                            self.__bias[i] = self.__bias[i] + learning_rate*data_error[i]*data[0][j]
                
                error += data_error
            
            # Validation
            for data in validation_data:
                expected_values.append(data[1])
                network_outputs.append(np.argmax(self.forward(data[0])))

            precision = precision_score(expected_values, network_outputs, average="macro")
            recall = recall_score(expected_values, network_outputs, average="macro")
            f1 = f1_score(expected_values, network_outputs, average="macro")
            accuracy = accuracy_score(expected_values, network_outputs)

            print("\nEpoch {} Results:".format(epoch))
            print("Accuracy: {}".format(accuracy))
            print("Precision: {}".format(precision))
            print("Recall: {}".format(recall))
            print("F1: {}\n".format(f1))

            epoch += 1

        test_expected = []
        text_output = []
        for data in test_data:
            expected = data[1]
            output = self.forward(data[0])
            test_expected.append(expected)
            text_output.append(np.argmax(output))

            print("Expected: {} | Output: {}".format(expected, np.argmax(output)))

        print("Test accuracy: {}".format(accuracy_score(test_expected, text_output)))

    def get_weights(self):
        return self.__weights
    
    def get_bias(self):
        return self.__bias

    def __str__(self) -> str:
        return "Hello World!"