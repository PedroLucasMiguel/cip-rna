import os
from utils.datasets import *
from PerceptronNetwork.pn import PerceptronNetwork

if __name__ == "__main__":

    try:
        os.mkdir("../output/")
    except FileExistsError:
        pass
    
    #data = parse_iris_data()
    #a = PerceptronNetwork(random_bias_initialization=False, random_weight_initialization=False, n_input_features=4, n_classes=3)
    #a.train(data, 600, 0.001)

    data = parse_wine_data()
    a = PerceptronNetwork(random_bias_initialization=False, random_weight_initialization=False, n_input_features=13, n_classes=3)
    a.train(data, 600, 0.0001)

    