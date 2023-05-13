import os
from utils.datasets import *
from PerceptronNetwork.pn import PerceptronNetwork
from DenseNetIAX.lime_explain import get_lime
from DenseNetIAX.train_network import train
from DenseNetIAX.grad_cam import get_grad_cam

"""
Neste arquivo é tratado apenas a lógica dos menus de navegação da aplicação
"""

def call_algorithms(algorithm:int) -> None:

    if algorithm == 1 or algorithm == 2:
        while True:
            os.system('cls' if os.name == 'nt' else 'clear')
            print("Exercício {}".format(algorithm))
            epochs = int(input("N° épocas (>=1): "))

            if epochs >= 1:
                learning_rate = float(input("\nTaxa de aprendizado (<=0.9): "))

                if learning_rate <= 0.9:
                    
                    print("\nUsar pesos aleatórios?")
                    print("[0] - Não")
                    print("[1] - Sim")
                    random_weight = bool(input("Resposta: "))

                    print("\nUsar valores de bias aleatórios?")
                    print("[0] - Não")
                    print("[1] - Sim")
                    random_bias = bool(input("Resposta: "))
                    
                    os.system('cls' if os.name == 'nt' else 'clear')

                    if algorithm == 1:
                        data = parse_iris_data()
                        pn = PerceptronNetwork(
                            random_bias_initialization=random_bias, 
                            random_weight_initialization=random_weight, 
                            n_input_features=4, 
                            n_classes=3, 
                            identifier="exercicio1"
                            )
                        pn.train(data, epochs, learning_rate)

                    if algorithm == 2:
                        data = parse_wine_data()
                        pn = PerceptronNetwork(
                            random_bias_initialization=random_bias, 
                            random_weight_initialization=random_weight, 
                            n_input_features=13, 
                            n_classes=3, 
                            identifier="exercicio2"
                        )
                        pn.train(data, epochs, learning_rate)

                    input("Pressione ENTER para continuar...")
                    break;
                
                else:
                    print("\nTaxa de aprendizado inválida!")
                    input("Pressione ENTER para tentar novamente.")
            
            else:
                print("\nNúmero de épocas inválido!")
                input("Pressione ENTER para tentar novamente.")
    else:
        
        while True:
            os.system('cls' if os.name == 'nt' else 'clear')
            print("Exercício 3 - DenseNet-201 IAX\nEscolha o tipo de operação:")
            print("[1] - Treinar o modelo")
            print("[2] - Obter classificação e gerar explicações")
            print("[3] - Voltar")
            answ = int(input("Resposta: "))

            os.system('cls' if os.name == 'nt' else 'clear')

            if answ == 1:
                train()

            elif answ == 2:
                images = os.listdir("../images")
                print("Selecione a imagem:\n(Você pode adicionar novas imagens a lista inserindo novas imagens na pasta \"images/\")")

                for i in range(len(images)):
                    print("[{}] - {}".format(i, images[i]))

                img = int(input("Resposta: "))

                while True:
                    os.system('cls' if os.name == 'nt' else 'clear')
                    print("Escolha o modelo de explicação:")
                    print("[1] - Grad Cam")
                    print("[2] - LIME")
                    print("[3] - Voltar")
                    answ = int(input("Resposta: "))

                    os.system('cls' if os.name == 'nt' else 'clear')

                    if answ == 1:
                        get_grad_cam("../images/{}".format(images[img]))
                        input("Pressione ENTER para continuar...")

                    elif answ == 2:
                        get_lime("../images/{}".format(images[img]))
                        input("Pressione ENTER para continuar...")

                    elif answ == 3:
                        break;
                    
                    else:
                        print("Opção inválida!")
                        input("Pressine ENTER para tentar novamente.")
            
            elif answ == 3:
                break;

            else:
                print("Opção inválida!")
                input("Pressine ENTER para tentar novamente.")

    pass


if __name__ == "__main__":

    try:
        os.mkdir("../output/")
    except FileExistsError:
        pass
    
    stop = False

    while not stop:
        os.system('cls' if os.name == 'nt' else 'clear')
        print("Escolha o exercício:")
        print("[1] - Exercício 1")
        print("[2] - Exercício 2")
        print("[3] - Exercício 3")
        print("[4] - Sair")
        answ = int(input("Resposta: "))

        if answ >= 1 and answ <= 3:
            call_algorithms(answ)
        
        elif answ == 4:
            print("\nFinalizando...")
            stop = True

        else:
            print("\nOpção inválida!")
            input("Pressione ENTER para tentar novamente.")

    