import pandas as pd, csv, numpy as np, tensorflow as tf, random
from tensorflow                      import keras
from tensorflow.keras                import layers
from sklearn.preprocessing           import RobustScaler
from sklearn.model_selection         import train_test_split
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.svm import SVC
tf.random.set_seed(1)

class AlgoritmosPreditivos:
    @staticmethod
    def pre_proc(dataset, x_columns, y_columns, hotEncoding=True):
        df_input = pd.DataFrame(dataset)

        scalers_x = RobustScaler()
        df_input_x = scalers_x.fit_transform(df_input[x_columns])
        if(hotEncoding):
            df_input_y = to_categorical(df_input[y_columns])
        else:
            df_input_y = np.array(df_input[y_columns], dtype=int)

        X_train, X_test, Y_train, Y_test = train_test_split(df_input_x, df_input_y, test_size=0.2, random_state=1)

        return [X_train, Y_train], [X_test, Y_test]


    @staticmethod
    def treina_e_avalia_MLP(dataset):     

        XY_train, XY_test = AlgoritmosPreditivos.pre_proc(dataset,[0, 1, 2, 3], [4])
        HIDDEN_LAYERS = 50

        model = keras.Sequential() 
        model.add(layers.Dense(HIDDEN_LAYERS, input_shape=(4,), activation="relu"))
        model.add(layers.Dropout(0.75))
        model.add(layers.Dense(3, activation="softmax"))

        model.compile(loss='categorical_crossentropy',
                    optimizer='sgd',
                    metrics=['accuracy'])
        
        history = model.fit(
            XY_train[0],    # X
            XY_train[1],    # Y
            validation_split = 0.2,
            batch_size = 64,
            epochs = 200,
            verbose=2)
        

        # plot training history
        plt.plot(history.history['accuracy'],      label='train (Ac)')
        plt.plot(history.history['val_accuracy'],  label='test  (Ac)')
        model.summary()
        test_loss, test_acc = model.evaluate(XY_test[0], XY_test[1])
        # print('Test Accuracy: ', test_acc, '\nTest Loss: ', test_loss)

        plt.legend()
        plt.show()
        return test_acc

    @staticmethod
    def treina_e_avalia_DT(dataset):     

        XY_train, XY_test = AlgoritmosPreditivos.pre_proc(dataset,[0, 1, 2, 3], [4]) 
        modelo = tree.DecisionTreeClassifier()
        modelo = modelo.fit(XY_train[0], XY_train[1])
        y_true, y_prev = [], []
        for i, x  in enumerate(XY_test[0]):
            y_true.append(XY_test[1][i].tolist())
            y_prev.append(modelo.predict([x]).tolist()[0])
        ac_m = tf.keras.metrics.Accuracy()
        ac_m.update_state(y_true, y_prev)
        test_acc = ac_m.result().numpy()
        tree.plot_tree(modelo)
        plt.show()
        return test_acc

    @staticmethod
    def treina_e_avalia_SVM(dataset, kernel):     
        XY_train, XY_test = AlgoritmosPreditivos.pre_proc(dataset,[0, 1, 2, 3, 4, 5, 6, 7], [8], hotEncoding=False) 
        modelo = SVC(kernel=kernel)
        modelo = modelo.fit(XY_train[0], XY_train[1].ravel())
        test_acc = modelo.score(XY_test[0], XY_test[1].ravel())
        return test_acc

    @staticmethod
    def carrega_arquivo(nome):
        dataset = []
        with open(nome) as _csv:
            _r = csv.reader(_csv, delimiter=',')
            linha= 0
            for row in _r:
                if linha > 0:
                    dataset.append(row)
                linha += 1
        random.shuffle(dataset)
        return dataset

def main():
    dataset_iris = AlgoritmosPreditivos.carrega_arquivo('Data_Iris.csv')
    # Base de diabetes: https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv
    # Indexado em https://machinelearningmastery.com/standard-machine-learning-datasets/
    dataset_diabetes =AlgoritmosPreditivos.carrega_arquivo('Diabetes.csv')
    while True:
        selection = int(input("\n Gostaria de prever::\n [1] Iris com MLP\n [2] Iris com Árvores de Decisão\n [3] Diabetes com SVM\n>> "))
        if(selection==1):
            ac = AlgoritmosPreditivos.treina_e_avalia_MLP(dataset_iris)
            print(f"Acurácia do Multi Layer Perceptron (MLP) p/ o dataset Iris: {np.average(ac)}")
        elif(selection==2):
            np.random.seed(1) 
            ac = AlgoritmosPreditivos.treina_e_avalia_DT(dataset_iris)
            print(f"Acurácia da Árvore de Decisão p/ o dataset Iris: {np.average(ac)}")
            np.random.seed(None) 
        else:
            lin, sig, poly, rbf = [], [], [], []
            for _ in range(10):
                lin.append(AlgoritmosPreditivos.treina_e_avalia_SVM(dataset_diabetes, kernel="linear"))
                sig.append(AlgoritmosPreditivos.treina_e_avalia_SVM(dataset_diabetes, kernel="sigmoid"))
                poly.append(AlgoritmosPreditivos.treina_e_avalia_SVM(dataset_diabetes, kernel="poly"))
                rbf.append(AlgoritmosPreditivos.treina_e_avalia_SVM(dataset_diabetes, kernel="rbf"))
            print(f"Acurácia da SVM c/ kernel Sigmoide: {np.average(sig)}")
            print(f"Acurácia da SVM c/ kernel Polinomial: {np.average(poly)}")
            print(f"Acurácia da SVM c/ kernel Linear: {np.average(lin)}")
            print(f"Acurácia da SVM c/ kernel de Base Radial Gaussiana: {np.average(rbf)}")
        

    
main()



