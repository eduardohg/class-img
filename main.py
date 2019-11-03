from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn.metrics import classification_report
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

def retornaImagensTrain():
	vetor = []
	for file in glob.glob("./mnist/train/*.jpeg"):
		vetor.append(file)
	return vetor

def retornaImagensValid():
    vetor = []
    for file in glob.glob("./mnist/val/*.jpeg"):
        vetor.append(file)
    return vetor

def gen_computed_dataset_train(dataset):
    csv = open("train.csv", "w+")
    winSize = (28,28)
    blockSize = (4,4)
    blockStride = (4,4)
    cellSize = (4,4)
    nbins = 9
    classes = nbins * winSize[0]/blockSize[0] * winSize[1]/blockSize[1]
    dims = 10
    line = ""
    # line = "classes:" + str(int(classes)) + "\n" + "dims:" + str(dims) + "\n"
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins)
    for file in dataset:
        img = cv2.imread(file)
        h = hog.compute(img)
        line += str(h[0][0])
        for i, dot in enumerate(h):
            if i != 0:
                line += ", " + str(dot[0])
        name_file = ''
        for val in file.split("."):
            if len(val) > len(name_file):
                name_file = val       
        for i in range(dims):
            if str(i) == name_file[-1]:
                line += ", 1"
            else:
                line += ", 0"
        line += "\n"
    csv.write(line)
    csv.close()
    return classes

def gen_computed_dataset_valid(dataset):
    csv = open("valid.csv", "w+")
    winSize = (28,28)
    blockSize = (4,4)
    blockStride = (4,4)
    cellSize = (4,4)
    nbins = 9
    classes = nbins * winSize[0]/blockSize[0] * winSize[1]/blockSize[1]
    dims = 10
    line = ""
    # line = "classes:" + str(int(classes)) + "\n" + "dims:" + str(dims) + "\n"
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins)
    for file in dataset:
        img = cv2.imread(file)
        h = hog.compute(img)
        line += str(h[0][0])
        for i, dot in enumerate(h):
            if i != 0:
                line += ", " + str(dot[0])
        name_file = ''
        for val in file.split("."):
            if len(val) > len(name_file):
                name_file = val       
        for i in range(dims):
            if str(i) == name_file[-1]:
                line += ", 1"
            else:
                line += ", 0"
        line += "\n"
    csv.write(line)
    csv.close()
    return classes

def main():
    # load the dataset

    file = retornaImagensTrain()
    dim = gen_computed_dataset_train(file)
    D = np.genfromtxt('train.csv',delimiter=',')
    
    file = retornaImagensValid()
    gen_computed_dataset_valid(file)
    E = np.genfromtxt('valid.csv',delimiter=',')

    # dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')
    # split into input (X) and output (y) variables
    X = D[:,0:441]
    y = D[:,441:]
    testX = E[:,0:441]
    testY = E[:,441:]
    # define the keras model
    model = Sequential()
    model.add(Dense(128, input_dim=441, activation='sigmoid'))
    model.add(Dense(64, activation='sigmoid'))
    model.add(Dense(10, activation='softmax'))
    # compile the keras model
    model.compile(loss='categorical_crossentropy', optimizer=SGD(0.01), metrics=['accuracy'])
    # fit the keras model on the dataset
    H = model.fit(X, y, epochs=200, batch_size=10, verbose=2, validation_data=(testX, testY))
    # evaluate the keras model
    print("[INFO] avaliando a rede neural...")
    predictions = model.predict(testX, batch_size=128)
    print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1)))
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0,200), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0,200), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0,200), H.history["accuracy"], label="train_accuracy")
    plt.plot(np.arange(0,200), H.history["val_accuracy"], label="val_accuracy")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()