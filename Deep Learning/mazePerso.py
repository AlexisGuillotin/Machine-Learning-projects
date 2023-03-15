from __future__ import print_function
import numpy as np
import os, sys, time, datetime, json, random
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD , Adam, RMSprop
import matplotlib.pyplot as plt
from keras.layers import PReLU
from keras.models import Sequential
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# This program must evaluate the return of the environment according to an action of the agent.

def create_space(start, end, dragons):
    '''
    Create a function which makes it possible to simulate the board by positioning in particular the different elements (start, end, dragons).
    '''
    space = np.zeros((4,4))
    space[start] = 1
    space[end] = 2
    for dragon in dragons:
        space[dragon] = 3
    return space

def application_action(action, position, space):
    '''
    Create a function that simulates the interaction between the agent and its environment.
    :param action: The variable action contains the action performed (0,3).
    :param position: The variable position contains the position of the agent.
    :param space: The space variable contains the organization of the board.

    :return position: variable position is the new position.
    :return Reward: Reward is the reward obtained (a real).
    :return end: end variable is a boolean indicating if the game is over.
    '''
    x, y = position
    reward = 0
    end = False
    case = 0
    if action == 0:
        if y+1 <=3:
            case = space[x][y+1]
            position = (x,y+1)
        else:
            reward = -1
            return position, reward, end

    if action == 1:
        if y-1 >=0:
            case = space[x][y-1]
            position = (x,y-1)
        else:
            reward = -1
            return position, reward, end

    if action == 2:
        if x-1 >=0:
            case = space[x-1][y]
            position = (x-1,y)
        else:
            reward = -1
            return position, reward, end

    if action == 3:
        if x+1 <=3:
            case = space[x+1][y]
            position = (x+1,y)
        else:
            reward = -1
            return position, reward, end

    if case == 1:
        reward = 0
    if case == 2:
        reward = 1
        end = True
    if case == 3:
        reward = -1
        end = True
    return position, reward, end

# You are going to develop the procedure allowing an agent to create its policy by the algorithm of Q-learning. For this, it is necessary to develop :
def choose_action(state,epsilon,mat_q):
    '''
    Function which applies the epsilon-greedy strategy to choose the action to perform according to the state of the agent (or its position).
    :param state: The variable state contains the position of the agent.
    :param epsilon: The variable epsilon contains the probability of exploration.
    :param mat_q: The variable mat_q contains the matrix of Q-values.

    :return action: The action to be performed by the agent.
    '''
    if np.random.rand() < epsilon:
        action = np.random.randint(0,4)
    else:
        x, y = state
        action = max(mat_q[0][x+y],mat_q[1][x+y],mat_q[2][x+y],mat_q[3][x+y])
    return action

def onestep(mat_q,state,epsilon):
    '''
    The function receives the current table storing the cumulative rewards for a chosen action for a
    particular state. And so for the current state contained in the variable state, it updates the box associated with
    the action chosen by strategy epsilon-greedy with parameter epsilon. It then returns the updated table, so
    than the new state of the agent.
    :param mat_q: The variable mat_q contains the matrix of Q-values.
    :param state: The variable state contains the position of the agent.
    :param epsilon: The variable epsilon contains the probability of exploration.

    :return mat_q: The variable mat_q contains the matrix of Q-values.
    :return state: The variable state contains the position of the agent.
    '''
    action = choose_action(state,epsilon,mat_q)
    state = action
    return mat_q, state
'''
Nous souhaitons, dans le même contexte, mettre en place une stratégie Deep Q-Learning. Vous pouvez réutiliserles fonctions d’interaction avec l’environnement.
Pour simplifier, nous proposons dans un premier temps de tester l’algorithme à travers une structure simple :2 couches denses ayant 16 entrées (nombre de cases) et 4 sorties (4 actions).Il vous faut alors :
remplir  la  fonction action = choose_action2(state,epsilon,modele)qui  va  choisir  dans  certains  cas l’action à appliquer avec textttSortieQ = model.predict(np.array([vecetat])),
la procédure permettant de mettre à jour les poids du réseau grâce à la différentiation automatique texttttf.GradientTape().
Tester votre algorithme avec des récompensesR=−20,−1,100.
Jouer une partie avec la politique optimale lié à la table et commenter le parcours.  
Faites  ́evoluer votre algorithme en introduisant un second réseau.
'''
def choose_action2(state, epsilon, modele):
    '''
    Function which applies the epsilon-greedy strategy to choose the action to perform according to the state of the agent (or its position).
    :param state: The variable state contains the position of the agent.
    :param epsilon: The variable epsilon contains the probability of exploration.
    :param modele: The variable modele contains the deep neural network.

    :return action: The action to be performed by the agent.
    '''
    if np.random.rand() < epsilon:
        action = np.random.randint(0,4)
    else:
        q_values = modele.predict(np.array([state]))
        action = np.argmax(q_values)
    return action

def Q_learning():
    '''
    Test your algorithm with rewards R = −1, 0, 1. Study the table Q. Play a game with the
    optimal policy related to the table and comment on the path
    '''
    space = create_space((0,0), (3,3), [(0,1),(2,1),(1,3)])
    position = (0,0)
    mat_q = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
    epsilon = 0.5
    
    # we do 100 games
    for i in range(100):
        # one game
        for i in range(100) or end == True:
            action = choose_action(position,epsilon,mat_q)
            position, reward, end = application_action(action,position,space)
            if reward == 1:
                x, y = position
                mat_q[action][x+y] += 1
            if reward == -1:
                x, y = position
                mat_q[action][x+y] -= 1
    print(mat_q)


def DeepQ_learning():
    space = create_space((0,0), (3,3), [(0,1),(2,1),(1,3)])
    position = (0,0)
    mat_q = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
    n_epochs = 1000

    vec_etat= np.zeros(16)
    vec_etat[int(N*position[0] + position[1])] = 1

    model = keras.Sequential([
        keras.layers.Dense(4,  activation='sigmoid',input_shape= [2]),
        keras.layers.Dense(4, activation='sigmoid'),
        keras.layers.Dense(1, activation='sigmoid'),
    ])
    optimizer= keras.optimizers.Nadam(learning_rate=0.01)
    loss_fn= keras.losses.mean_squared_error

    X_train, X_test, Y_train, Y_test= train_test_split(mat_q,space, test_size=0.2, random_state= 1,stratify = space)

    list_idx= np.arange(0,len(X_train))
    for epoch in range(n_epochs):
        som_loss= 0
        random.shuffle(list_idx)
        for idx in list_idx:
            x = np.array([X_train[idx,0:2]])
            y = Y_train[idx]

    with tf.GradientTape() as tape:
        y_pred= model(x)
        loss= loss_fn(y,y_pred)
    gradients = tape.gradient(loss,model.trainable_variables)
    optimizer.apply_gradients(zip(gradients,model.trainable_variables))
    som_loss+=loss


def main():
    # Part 2 : Q-learning
    Q_learning()

    # Part 3 : DeepQ-learning
    DeepQ_learning()


if __name__ == "__main__":
    main()