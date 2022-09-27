import os
import cv2
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Deep Neural Network 2 Layer
class NeuralNetwork():
    def __init__(self, input_size, hidden_size, output_size):
        self.in_size = input_size
        self.hidden_size = hidden_size
        self.out_size = output_size
        self.parameters = self.initialize()
        self.cache = {}

    def initialize(self):
        input_layer = self.in_size
        hidden_layer = self.hidden_size
        output_layer = self.out_size

        param = {
            "W1": np.random.randn(hidden_layer, input_layer) * np.sqrt(1./input_layer),
            "b1": np.zeros((hidden_layer, 1)) * np.sqrt(1./input_layer),
            "W2": np.random.randn(output_layer, hidden_layer) * np.sqrt(1./hidden_layer),
            "b2": np.zeros((output_layer, 1)) * np.sqrt(1./hidden_layer)
        }
        return param

    def sigmoid(self, x, derivative = False):
        if derivative:
            return (np.exp(-x)) / ((np.exp(-x)+1) ** 2)
        return 1 / (1 + np.exp(-x))

    def feed_forward(self, x):
        self.cache["X"] = x
        self.cache["Z1"] = np.matmul(self.parameters["W1"], self.cache["X"].T) + self.parameters["b1"]
        self.cache["A1"] = self.sigmoid(self.cache["Z1"])
        self.cache["Z2"] = np.matmul(self.parameters["W2"], self.cache["A1"]) + self.parameters["b2"]
        self.cache["A2"] = self.sigmoid(self.cache["Z2"])
        return self.cache["A2"]

    def back_propagate(self, y, output):
        batch_size = y.shape[0]
        dZ2 = output - y.T
        dW2 = (1./batch_size) * np.matmul(dZ2, self.cache["A1"].T)
        db2 = (1./batch_size) * np.sum(dZ2, axis=1, keepdims=True)

        dA1 = np.matmul(self.parameters["W2"].T, dZ2)
        dZ1 = dA1 * self.sigmoid(self.cache["Z1"], derivative=True)
        dW1 = (1./batch_size) * np.matmul(dZ1, self.cache["X"])
        db1 = (1./batch_size) * np.sum(dZ1, axis=1, keepdims=True)

        self.grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
        return self.grads

    def SGD(self, lr = 0.1, beta = 0.9):
        for key in self.parameters:
            self.parameters[key] = self.parameters[key] - lr * self.grads[key]

    def accuracy(self, y, output):
        return np.mean(np.argmax(y, axis=-1) == np.argmax(output.T, axis=-1))

    def cross_entropy_loss(self, y, output):
        l_sum = np.sum(np.multiply(y.T, np.log(output)))
        m = y.shape[0]
        l = -(1./m) * l_sum
        return l

    def train(self, x_train, y_train, x_test, y_test, epochs=100, batch_size=64, lr=0.1, beta=0.9):
        self.epochs = epochs
        self.batch_size = batch_size
        num_batches = -(-x_train.shape[0] // self.batch_size)     # upper bound of data size

        train_loss_curve = []
        test_loss_curve = []
        
        for i in range(self.epochs):
            shuffled = np.random.permutation(x_train.shape[0])
            x_train_shuffled = x_train[shuffled]
            y_train_shuffled = y_train[shuffled]

            for j in range(num_batches):
                # Setting Batch
                begin = j * self.batch_size
                end = min(begin + self.batch_size, x_train.shape[0]-1)
                x = x_train_shuffled[begin:end]
                y = y_train_shuffled[begin:end]
                
                output = self.feed_forward(x)             # Forward 
                grad = self.back_propagate(y, output)     # Backprop
                self.SGD(lr=lr, beta=beta)                # Optimize

            # Training Evaluation
            output = self.feed_forward(x_train)
            train_acc = self.accuracy(y_train, output)
            train_loss = self.cross_entropy_loss(y_train, output)
            train_loss_curve.append(train_loss)
            # Test data Evaluation
            output = self.feed_forward(x_test)
            test_acc = self.accuracy(y_test, output)
            test_loss = self.cross_entropy_loss(y_test, output)
            test_loss_curve.append(test_loss)
            print(f"Epoch {i + 1}: train acc={train_acc:.2f}, train loss={train_loss:.2f}, test acc={test_acc:.2f}, test loss={test_loss:.2f}")
        return test_acc, test_loss, train_loss_curve, test_loss_curve

# Deep Neural Network 3 Layer
class NeuralNetwork_3layer():
    def __init__(self, input_size, hidden_size, output_size):
        self.in_size = input_size
        self.hidden_size = hidden_size
        self.out_size = output_size
        self.parameters = self.initialize()
        self.cache = {}

    def initialize(self):
        input_layer = self.in_size
        hidden_layer = self.hidden_size
        output_layer = self.out_size

        param = {
            "W0": np.random.randn(hidden_layer, input_layer) * np.sqrt(1./input_layer),
            "b0": np.zeros((hidden_layer, 1)) * np.sqrt(1./input_layer),
            "W1": np.random.randn(hidden_layer*2, hidden_layer) * np.sqrt(1./hidden_layer),
            "b1": np.zeros((hidden_layer*2, 1)) * np.sqrt(1./hidden_layer),
            "W2": np.random.randn(output_layer, (hidden_layer*2)) * np.sqrt(1./(hidden_layer*2)),
            "b2": np.zeros((output_layer, 1)) * np.sqrt(1./(hidden_layer*2))
        }
        return param

    def sigmoid(self, x, derivative = False):
        if derivative:
            return (np.exp(-x)) / ((np.exp(-x)+1) ** 2)
        return 1 / (1 + np.exp(-x))

    def feed_forward(self, x):
        self.cache["X"] = x
        self.cache["Z0"] = np.matmul(self.parameters["W0"], self.cache["X"].T) + self.parameters["b0"]
        self.cache["A0"] = self.sigmoid(self.cache["Z0"])
        self.cache["Z1"] = np.matmul(self.parameters["W1"], self.cache["A0"]) + self.parameters["b1"]
        self.cache["A1"] = self.sigmoid(self.cache["Z1"])
        self.cache["Z2"] = np.matmul(self.parameters["W2"], self.cache["A1"]) + self.parameters["b2"]
        self.cache["A2"] = self.sigmoid(self.cache["Z2"])
        return self.cache["A2"]

    def back_propagate(self, y, output):
        batch_size = y.shape[0]
        dZ2 = output - y.T
        dW2 = (1./batch_size) * np.matmul(dZ2, self.cache["A1"].T)
        db2 = (1./batch_size) * np.sum(dZ2, axis=1, keepdims=True)

        dA1 = np.matmul(self.parameters["W2"].T, dZ2)
        dZ1 = dA1 * self.sigmoid(self.cache["Z1"], derivative=True)
        dW1 = (1./batch_size) * np.matmul(dZ1, self.cache["A0"].T)
        db1 = (1./batch_size) * np.sum(dZ1, axis=1, keepdims=True)

        dA0 = np.matmul(self.parameters["W1"].T, dZ1)
        dZ0 = dA0 * self.sigmoid(self.cache["Z0"], derivative=True)
        dW0 = (1./batch_size) * np.matmul(dZ0, self.cache["X"])
        db0 = (1./batch_size) * np.sum(dZ0, axis=1, keepdims=True)

        self.grads = {"W0": dW0, "b0": db0, "W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
        return self.grads

    def SGD(self, lr = 0.1, beta = 0.9):
        for key in self.parameters:
            self.parameters[key] = self.parameters[key] - lr * self.grads[key]

    def accuracy(self, y, output):
        return np.mean(np.argmax(y, axis=-1) == np.argmax(output.T, axis=-1))

    def cross_entropy_loss(self, y, output):
        l_sum = np.sum(np.multiply(y.T, np.log(output)))
        m = y.shape[0]
        l = -(1./m) * l_sum
        return l

    def train(self, x_train, y_train, x_test, y_test, epochs=100, batch_size=64, lr=0.1, beta=0.9):
        print("========================================================================")
        self.epochs = epochs
        self.batch_size = batch_size
        num_batches = -(-x_train.shape[0] // self.batch_size)     # upper bound of data size

        train_loss_curve = []
        test_loss_curve = []
        
        for i in range(self.epochs):
            shuffled = np.random.permutation(x_train.shape[0])
            x_train_shuffled = x_train[shuffled]
            y_train_shuffled = y_train[shuffled]

            for j in range(num_batches):
                # Setting Batch
                begin = j * self.batch_size
                end = min(begin + self.batch_size, x_train.shape[0]-1)
                x = x_train_shuffled[begin:end]
                y = y_train_shuffled[begin:end]
                
                output = self.feed_forward(x)             # Forward 
                grad = self.back_propagate(y, output)     # Backprop
                self.SGD(lr=lr, beta=beta)                # Optimize

            # Training Evaluation
            output = self.feed_forward(x_train)
            train_acc = self.accuracy(y_train, output)
            train_loss = self.cross_entropy_loss(y_train, output)
            train_loss_curve.append(train_loss)
            # Test data Evaluation
            output = self.feed_forward(x_test)
            test_acc = self.accuracy(y_test, output)
            test_loss = self.cross_entropy_loss(y_test, output)
            test_loss_curve.append(test_loss)
            print(f"Epoch {i + 1}: train acc={train_acc:.2f}, train loss={train_loss:.2f}, test acc={test_acc:.2f}, test loss={test_loss:.2f}")
        return test_acc, test_loss, train_loss_curve, test_loss_curve


if __name__ == '__main__':
    ##### Preprocessing
    classes = ['Carambula', 'Lychee', 'Pear']
    label = 0
    train_data, train_label, test_data, test_label = [], [], [], []
    for class_name in classes:
        for filename in os.listdir('./Data_train/' + class_name):
            image = (np.array(cv2.imread('Data_train/' + class_name + '/' + filename))[:,:,0] / 255.0)
            train_data.append(image.reshape(image.shape[0] * image.shape[1]))
            train_label.append(label)
        for filename in os.listdir('./Data_test/' + class_name):
            image = (np.array(cv2.imread('Data_test/' + class_name + '/' + filename))[:,:,0] / 255.0)
            test_data.append(image.reshape(image.shape[0] * image.shape[1]))
            test_label.append(label)
        label += 1

    # PCA
    pca = PCA(2)                                # 2 principal components as required
    train_X = pca.fit_transform(train_data)
    test_X = pca.transform(test_data)
    train_label = np.array(train_label)
    test_label = np.array(test_label)
    print(train_label.shape)

    # One Hot label
    test_y = np.array(test_label.astype('int32')[:, None] == np.arange(3), dtype=np.float32)
    train_y = np.array(train_label.astype('int32')[:, None] == np.arange(3), dtype=np.float32)

    print(train_X.shape)
    print(train_y.shape)

    ##### Train
    DNN = NeuralNetwork(input_size = 2, hidden_size = 16, output_size = 3)
    layer2_test_acc, layer2_test_loss, layer2_train_loss_curve, layer2_test_loss_curve = DNN.train(train_X, train_y, test_X, test_y, batch_size=32, lr=0.1, beta=0.9)

    DNN3 = NeuralNetwork_3layer(input_size = 2, hidden_size = 16, output_size = 3)
    layer3_test_acc, layer3_test_loss, layer3_train_loss_curve, layer3_test_loss_curve = DNN3.train(train_X, train_y, test_X, test_y, batch_size=32, lr=0.1, beta=0.9)

    ##### Loss Curve
    print(f"2 Layer: test acc={layer2_test_acc}, test loss={layer2_test_loss}")
    print(f"3 Layer: test acc={layer3_test_acc}, test loss={layer3_test_loss}")
    fig = plt.figure()
    ax0 = fig.add_subplot(121, title="2 layers")
    ax1 = fig.add_subplot(122, title="3 layers")
    ax0.plot(layer2_train_loss_curve, 'bo-', label='train')
    ax0.plot(layer2_test_loss_curve, 'ro-', label='test')
    ax1.plot(layer3_train_loss_curve, 'bo-', label='train')
    ax1.plot(layer3_test_loss_curve, 'ro-', label='test')
    ax0.legend()
    ax1.legend()
    fig.savefig('./loss.jpg')

    ##### Decision Region
    test0 = test_X[:,0]
    test1 = test_X[:,1]
    padding = 1.0
    test0_min, test0_max = np.round(test0.min()) - padding, np.round(test0.max() + padding)
    test1_min, test1_max = np.round(test1.min()) - padding, np.round(test1.max() + padding)

    step = 0.01 # mesh stepsize
    test0_axis_range = np.arange(test0_min, test0_max, step)
    test1_axis_range = np.arange(test1_min, test1_max, step)

    test_x0, test_x1 = np.meshgrid(test0_axis_range, test1_axis_range)
    
    fig, ax = plt.subplots()
    output = DNN.feed_forward(np.c_[test_x0.ravel(), test_x1.ravel()])
    Z = np.argmax(output.T, axis=-1)

    # Put the result into a color plot
    Z = Z.reshape(test_x0.shape)
    ax.contourf(test_x0, test_x1, Z, cmap=plt.cm.Paired)

    color_dic = {0 : "r", 1 : "c", 2 : "b"}
    Y = np.argmax(test_y, axis = -1)
    for i in range(3):
        idx = np.where(Y == i)
        for j in idx:
            ax.scatter(test_X[j, 0], test_X[j, 1], c=color_dic[i], label=classes[i]) 
    ax.legend()
    ax.set_title('DNN_2')
    plt.savefig("decision_region_2layer.png")

    fig, ax = plt.subplots()
    output = DNN3.feed_forward(np.c_[test_x0.ravel(), test_x1.ravel()])
    Z = np.argmax(output.T, axis=-1)

    # Put the result into a color plot
    Z = Z.reshape(test_x0.shape)
    ax.contourf(test_x0, test_x1, Z, cmap=plt.cm.Paired)

    color_dic = {0 : "r", 1 : "c", 2 : "b"}
    Y = np.argmax(test_y, axis = -1)
    for i in range(3):
        idx = np.where(Y == i)
        for j in idx:
            ax.scatter(test_X[j, 0], test_X[j, 1], c=color_dic[i], label=classes[i]) 
    ax.legend()
    ax.set_title('DNN_3')
    plt.savefig("decision_region_3layer.png")