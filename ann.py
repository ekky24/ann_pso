# -*- coding: utf-8 -*-

import numpy as np
import dataset
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

class MultiLayerPerceptron:
    def __init__(self, shape, weights=None):
        self.shape = shape
        self.num_layers = len(shape)
        if weights is None:
            self.weights = []
            for i in range(self.num_layers-1):
                W = np.random.uniform(size=(self.shape[i+1], self.shape[i] + 1))
                self.weights.append(W)
        else:
            self.weights = weights


    def run(self, data):
        X = data
        X = np.c_[ X, np.ones(len(data)) ]   
        z1 = X.dot(self.weights[0].T)
        a1 = np.divide(1, np.add(1, np.exp(np.negative(z1))))
        a1 = np.c_[ a1, np.ones(len(a1)) ]
        z2 = a1.dot(self.weights[1].T)
        exp_scores = np.exp(z2) 
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probs


if __name__ == '__main__':
    import pso
    import functools
    import sklearn.metrics
    import sklearn.datasets
    import pandas as pd

    def dim_weights(shape):
        dim = 0
        for i in range(len(shape)-1):
            dim = dim + (shape[i] + 1) * shape[i+1]
        print(dim)
        return dim

    def weights_to_vector(weights):
        w = np.asarray([])
        for i in range(len(weights)):
            v = weights[i].flatten()
            w = np.append(w, v)
        return w

    def vector_to_weights(vector, shape):
        weights = []
        idx = 0
        for i in range(len(shape)-1):
            r = shape[i+1]
            c = shape[i] + 1
            idx_min = idx
            idx_max = idx + r*c
            W = vector[idx_min:idx_max].reshape(r,c)
            weights.append(W)
        return weights

    def eval_neural_network(weights, shape, X, y):
        loss = np.asarray([])
        for w in weights:
            weights = vector_to_weights(w, shape)
            nn = MultiLayerPerceptron(shape, weights=weights)
            y_pred = nn.run(X)
            #loss = np.append(loss, (np.square(y - y_pred)).mean(axis=None))
            loss = np.append(loss, -np.sum(y*np.log(y_pred))/len(y))
        return loss

    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--type", required=True, help="tipe NN")
    args = vars(ap.parse_args())
    np.seterr(all='ignore', divide='ignore')

    if(args["type"] == '1'):
        num_classes = 26
        (X, y) = dataset.character()
        hidden_layer = 20
        data = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,1,1,0,0,0,1,1,1,1,1,0,0,1,1,0,0,1,1,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0]).reshape(1,64)
    elif(args["type"] == '2'):
        num_classes = 6
        (X, y) = dataset.lima_bit()
        hidden_layer = 12
        data = np.array([0,0,0,1,0]).reshape(1,5)
    elif(args["type"] == '3'):
        num_classes = 2
        (X, y) = dataset.xor()
        hidden_layer = 2
        data = np.array([0,0]).reshape(1,2)
    elif(args["type"] == '4'):
        num_classes = 2
        (X, y) = dataset.tiga_bit()
        hidden_layer = 3
        data = np.array([0,1,0]).reshape(1,3)
    
    num_inputs = X.shape[1]
    shape = (num_inputs, hidden_layer, num_classes)

    # Set up
    cost_func = functools.partial(eval_neural_network, shape=shape, X=X, y=y)
    swarm = pso.ParticleSwarm(cost_func, dim=dim_weights(shape), size=50)

    # Train...
    i = 0
    best_scores = [(i, swarm.best_score)]
    print(best_scores[-1])
    loss = []
    while swarm.best_score > 1e-6 and i<5000:
    #while i < 1:
        loss.append(swarm.best_score)
        swarm.update()
        i = i+1
        if swarm.best_score < best_scores[-1][1]:
            best_scores.append((i, swarm.best_score))
            print(best_scores[-1])

    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0,i), loss, label="loss")
    plt.title("Loss on training")
    plt.xlabel("Epoch #")
    plt.ylabel("Cross Entrophy")
    plt.legend(['train loss'], loc='upper right')
    plt.savefig('plot.png')

    # Test...
    best_weights = vector_to_weights(swarm.g, shape)
    best_nn = MultiLayerPerceptron(shape, weights=best_weights)
    test = best_nn.run(data)
    ohe = (test == test.max(axis=1)[:,None]).astype(int)
    print(ohe)