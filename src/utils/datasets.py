import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split

# Função responsável por ler o dataset de iris e converter para o formato que o algoritmo consegue
# compreender
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

# Função responsável por ler o dataset de vinho e converter para o formato que o algoritmo consegue
# compreender
def parse_wine_data():
    data = []
    class_array = []
    data_to_zscore = []

    with open("../Datasets/wine.data", "r") as dataset:
        lines = dataset.readlines()
        last_class_name = ""
        class_index = 0

        for line in lines:
            if line != "\n":
                line.replace("\n", "")
                splited_line = line.split(',')
                class_name = splited_line[0]

                if last_class_name == "":
                    last_class_name = class_name
                elif last_class_name != class_name:
                    last_class_name = class_name
                    class_index+=1

                data_line = []
                for i in range(1, len(splited_line)):
                    data_line.append(float(splited_line[i]))

                data_to_zscore.append(data_line.copy())
                class_array.append(class_index)

    # Calculando o zscore dos dados
    # Fazer esse processo melhora a performance da rede consideravelmente
    zscored_data = stats.zscore(data_to_zscore)
    
    for i in range(len(zscored_data)):
        data.append([zscored_data[i], class_array[i]])

    train, validation = train_test_split(data, test_size=0.15)
    train, test = train_test_split(train, test_size=0.15)

    return (train, validation, test)