import os
import numpy as np
from sklearn.model_selection import train_test_split
from PerceptronNetwork.pn import PerceptronNetwork

def parse_iris_data():
    data = []

    with open("../Datasets/iris.data", "r") as dataset:
        lines = dataset.readlines()
        last_class_name = ""
        class_index = 0
        for line in lines:
            if line != "\n":
                splited_line = line.split(',')
                class_name = splited_line[4]

                if last_class_name == "":
                    last_class_name = class_name
                elif last_class_name != class_name:
                    last_class_name = class_name
                    class_index+=1

                data.append([np.array([float(splited_line[0]), 
                                        float(splited_line[1]),
                                        float(splited_line[2]),
                                        float(splited_line[3])
                                        ], dtype=np.float16), class_index])


    train, validation = train_test_split(data, test_size=0.15)
    train, test = train_test_split(train, test_size=0.15)

    return (train, validation, test)


if __name__ == "__main__":

    try:
        os.mkdir("../output/")
    except FileExistsError:
        pass
    
    data = parse_iris_data()

    a = PerceptronNetwork(random_bias_initialization=True, random_weight_initialization=True, n_classes=3)
    a.train(data, 100, 0.01)