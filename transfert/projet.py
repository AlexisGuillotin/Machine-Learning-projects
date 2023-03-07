from unittest import result
import numpy as np
import matplotlib.pyplot as plt

class perceptron:

    def perceptron_simple(x, w, active):
        '''
        This program must evaluate the output of a simple perceptron (1 neuron) for an element input of R2.
        :param w: The variable w contains the synaptic weights of the neuron. It is a 3 row vector. The first line corresponds to the threshold.
        :param x: The variable x contains the input of the neural network. It is a 2-line vector.
        :param active: The active variable indicates the activation function used. If active==0 =sign(x), if active==1 = tanh(x)
        '''
        y = x[0]*w[1]+x[1]*w[2] + w[0]
        if active == 0:
            y = np.sign(y)
        elif active == 1:
            y = np.tanh(y)
        return y


    def apprentissage_widrow(x,yd,epoch,batch_size):
        '''
        This program returns the weight vector wobtained by learning according to the learning rule using gradient descent.
        :param x: The variable x contains the training set. It is a matrix with 2 rows and n columns.
        :param yd: The variable yd[i] indicates the desired answer for each element x[:,i].yd is a vector of 1 row and n columns of values ​​+1 or -1 (classification with 2 classes).
        :param epoch: The variable epoch indicates the number of epochs of the training.
        :param batch_size: The variable batch_size indicates the size of the batches used for the training.
        :return w: The variable w contains the synaptic weights of the neuron after training. It is a 3 row vector. The first line corresponds to the threshold.
        :return erreur: The variable error contains the cumulative error calculated for the complete passage of the training set. erreur=N−1∑i=0(yd(i)−y(i))2
        '''
        erreur = 0
        w = np.random.rand(3)
        for i in range(epoch):
            for j in range(batch_size):
                y = perceptron.perceptron_simple(x[:,j], w, 0)
                error = pow((yd[j]-y),2)
                erreur += pow((yd[j]-y),2)
                w[0] = w[0] + error*(yd[j]-y)
                w[1] = w[1] + error*(yd[j]-y)*x[0,j]
                w[2] = w[2] + error*(yd[j]-y)*x[1,j]
        return w, erreur
        
    
    def multicouche(x, w1, w2):
        ''''
        :param x: The variable x contains the input of the neural network. It is a 2-line vector.
        :param w1: The variable w1 contains the synaptic weights of the 2 neurons of the hidden layer. It is a matrix with 3 rows and 2 columns. The first column w1(:,1) corresponds to the 1st neuron of the hidden layer and w1(:,2) corresponds to the 2nd neuron of the hidden layer
        :param w2: The variable w2 contains the synaptic weights of the output neuron. It is a 3 row vector.
        :return y: The variable y contains the output of the neural network. It is a scalar.
        '''
        y = perceptron.perceptron_simple(x, w1[:,0], 0)
        y = perceptron.perceptron_simple(x, w1[:,1], 0)
        y = perceptron.perceptron_simple(x, w2, 0)
        return y

    def multiperceptron_widrow(x,yd,Epoch,Batch_size):
        '''
        This program returns the weight vectors w1obtained and w2 by learning according to the learning rule using gradient descent.
        :param x: The variable x contains the training set. It is a matrix with 2 rows and n columns.
        :param yd: The variable yd[i] indicates the desired answer for each element x[:,i].yd is a vector of 1 row and n columns of values ​​+1 or -1 (classification with 2 classes).
        :param Epoch: The variable Epoch indicates the number of epochs of the training.
        :param Batch_size: The variable Batch_size indicates the size of the batches used for the training.
        :return w1: The variable w1 contains the synaptic weights of the 2 neurons of the hidden layer after training. It is a matrix with 3 rows and 2 columns. The first column w1(:,1) corresponds to the 1st neuron of the hidden layer and w1(:,2) corresponds to the 2nd neuron of the hidden layer
        :return w2: The variable w2 contains the synaptic weights of the output neuron after training. It is a 3 row vector.
        '''
        w1 = np.random.rand(3,2)
        w2 = np.random.rand(3)
        for i in range(Epoch):
            for j in range(Batch_size):
                y = perceptron.multicouche(x[:,j], w1, w2)
                w2[0] = w2[0] + 0.1*(yd[j]-y)
                w2[1] = w2[1] + 0.1*(yd[j]-y)*x[0,j]
                w2[2] = w2[2] + 0.1*(yd[j]-y)*x[1,j]
                w1[0,0] = w1[0,0] + 0.1*(yd[j]-y)
                w1[1,0] = w1[1,0] + 0.1*(yd[j]-y)*x[0,j]
                w1[2,0] = w1[2,0] + 0.1*(yd[j]-y)*x[1,j]
                w1[0,1] = w1[0,1] + 0.1*(yd[j]-y)
                w1[1,1] = w1[1,1] + 0.1*(yd[j]-y)*x[0,j]
                w1[2,1] = w1[2,1] + 0.1*(yd[j]-y)*x[1,j]
        return w1, w2
        

# 1.1 Mise en place d'un perceptron simple
w = [0, 0, 1]
x = [0, 1]
active = 0

print("Sortie du neurone : "+str(perceptron.perceptron_simple(x,w,active)))

a=-1
b=0.5
t=[-0.5,1.5] 
z=[a*t[0]+b,a*t[1]+b]
plt.plot(t,z) 
plt.scatter(0,0,c='red')
plt.scatter(1,0,c='red')
plt.scatter(0,1,c='red')
plt.scatter(1,1,c='red')
plt.grid()

# 1.2.2 Apprentissage d'un perceptron simple
Data = np.loadtxt('transfert/p2_d1.txt')

# configuration variables
a = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]

#This file includes a variable that contains the learning set consisting of 2 classes of 25 individuals each in dimension 2. A "teacher" told us that the first 25 individuals come from class 1 and the last 25 from class 2.—Apply your learning algorithm. Display the evolution of the error. Check that the boundary is correct
w, erreur = perceptron.apprentissage_widrow(Data,a,10,25)
print("Poids après apprentissage : "+str(w))
print("erreur cumulée : "+str(erreur))
# 1.2.3 Programmation apprentissage Widrow-hoff



# iterations
iterations = 12

# dataset
Data = np.loadtxt('transfert/p2_d2.txt')
#print(Data[0])
print("----------")
#print(Data[1])

w, erreur = perceptron.apprentissage_widrow(Data, a, iterations, 32)
print("Widrow-Hoff weights : "+str(w))
print("Widrow-Hoff error : "+str(erreur))

# 1.3.1 Apprentissage d'un perceptron multicouche
# This program calculates the output of a multilayer perceptron with 1 neuron on the output layer and two neurons on the hidden layer for an input element of R2. The activation function is '(x) = 1/(1+e-x) .
def multiperceptron(x,w1,w2):
    '''
    :param w1: The variable w1 contains the synaptic weights of the 2 neurons of the hidden layer. It's a 3 matrix rows and 2 columns. The first column w1(:,1) corresponds to the 1st neuron of the hidden layer and w1(:,2) corresponds to the 2nd neuron of the hidden layer.
    :param w2: The variable w2 contains the synaptic weights of the output layer neuron. It is a 3 line vector.
    :param x: The variable x contains the input of the neural network. It is a 2 line vector.
    :return y: The variable y is a scalar corresponding to the output of the neuron.
    '''
    y = perceptron.perceptron_simple(x, w1[:,0], 0)
    y = perceptron.perceptron_simple(x, w1[:,1], 0)
    y = perceptron.perceptron_simple(x, w2, 0)
    return y


#Test your multilayer perceptron with the example below for an input x = [1 1]:
w1 = np.array([[0.5, 0.5], [1, 1], [1, 1]])
w2 = np.array([0.5, 1, 1])
x = np.array([1, 1])
print("Sortie du neurone : "+str(multiperceptron(x,w1,w2)))

# 1.3.2 Programme apprentissage multicouche

def multiperceptron_widrow(x,yd,Epoch,Batch_size):

    '''
    This program returns the weights w1 and w2 obtained by learning according to the learning rule using the
    descent of the gradient. We study the case of a multilayer perceptron with 1 neuron on the output layer and
    two neurons on the hidden layer for an element input of R2. The activation function is '(x) = 1
    1+e .
    :param x: The variable x contains the training set. It is a matrix with 2 rows and n columns.
    :param yd:The variable yd(i) indicates the desired answer for each element x(:,i). yd is a 1 row vector and n columns of values ​​+1 or 0 (classi cation has 2 classes).
    :param Epoch: the number of iterations on the training set.
    :param Batch_size: the number of individuals in the training set processed before updating the weights.

    :return w1: The variable w1 contains the synaptic weights of the 2 neurons of the hidden layer. It's a 3 matrix rows and 2 columns. The first column w1(:,1) corresponds to the 1st neuron of the hidden layer and w1(:,2) corresponds to the 2nd neuron.
    :return w2: The variable w2 contains the synaptic weights of the output layer neuron. It is a 3 line vector.
    :return error: The error variable contains the cumulative error of the calculation for the complete run of the training set
    '''

    # use apprenissage_widrow function
    w1 = np.array([[0.5, 0.5], [1, 1], [1, 1]])
    w2 = np.array([0.5, 1, 1])
    for i in range(Epoch):
        for j in range(Batch_size):
            y = multiperceptron(x[:,j], w1, w2)
            w2[0] = w2[0] + 0.1*(yd[j]-y)
            w2[1] = w2[1] + 0.1*(yd[j]-y)*x[0,j]
            w2[2] = w2[2] + 0.1*(yd[j]-y)*x[1,j]
            w1[0,0] = w1[0,0] + 0.1*(yd[j]-y)
            w1[1,0] = w1[1,0] + 0.1*(yd[j]-y)*x[0,j]
            w1[2,0] = w1[2,0] + 0.1*(yd[j]-y)*x[1,j]
            w1[0,1] = w1[0,1] + 0.1*(yd[j]-y)
            w1[1,1] = w1[1,1] + 0.1*(yd[j]-y)*x[0,j]
            w1[2,1] = w1[2,1] + 0.1*(yd[j]-y)*x[1,j]
    return w1, w2



# You are going to test your learning algorithm for the XOR case. The training set is given below:
# x(1) 0 1 0 1
#x(2) 0 0 1 1
#ydesire 0 1 1 0
x = np.array([[0, 1, 0, 1], [0, 0, 1, 1]])
yd = np.array([0, 1, 1, 0])
w1, w2 = multiperceptron_widrow(x,yd,1000,4)
print("w1 : "+str(w1))
print("w2 : "+str(w2))


# Deep et Full-connected : discrimination d’une image

# 2.1    Approche basée Descripteurs (basée modèle)

# .1.1    Calcul des descripteurs

