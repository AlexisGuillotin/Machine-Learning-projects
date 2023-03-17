from __future__ import print_function
import numpy as np

# This program must evaluate the return of the environment according to an action of the agent.

def create_space(start, end, dragons):
    '''
    Create a function which makes it possible to simulate the board by positioning in particular the different elements (start, end, dragons).
    :param start: start element.
    :param end: end element.
    :param dragons: dragons on some cases, if the agent reach one of those cases, he loose and restart.
    :return space: matrice representing the board game.
    '''
    space = np.zeros((4,4))
    space[start] = 2
    space[end] = 3
    for dragon in dragons:
        space[(4//dragon, dragon%4)] = 1
    return space

def application_action(action, state, space):
    '''
    Function to apply the action and return the new position on the maze.
    :param action: The action to be performed by the agent.
    :param current_pos: The current position of the agent.
    :param space: The matrix representing the maze.
    :return new_pos: The new position of the agent after performing the action.
    :return end: A flag indicating whether the episode has ended.
    '''
    reward = 0
    end = False
    if action == 0:
        if state <15:
            state += 1
        else:
            reward = -1
            return state, reward, end

    if action == 1:
        if state >=1:
            state -= 1
        else:
            reward = -1
            return state, reward, end

    if action == 2:
        if state >3:
            state -=4
        else:
            reward = -1
            return state, reward, end

    if action == 3:
        if state < 12:
            state +=4
        else:
            reward = -1
            return state, reward, end

    if state == 9 or state == 11:
        reward == -1
        end == True
    elif state == 15:
        reward = 1
        end = True
    else:
        reward = -0.2
    return state, reward, end

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
        action = np.argmax(mat_q[state])
    return action

def onestep(mat_q, state, reward, action):
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
    mat_q[state, action] = mat_q[state, action] + 0.81*(reward + 0.96*np.argmax(mat_q[state])-mat_q[state][action])
    return mat_q, state


'''
Part 2 : Develop a Q-learning algorithm
'''
def Q_learning():
    # Set up the environment
    start = (0, 0)
    end = (3, 3)
    dragons = [4, 7, 11]
    space = create_space(start, end, dragons)
    win_count = 0

    # Set hyperparameters
    epsilon = 0.1
    episodes = 5000
    max_steps = 100

    # Initialize the Q-matrix with zeros
    mat_q = np.zeros((16, 4))

    # Run Q-learning algorithm
    for episode in range(episodes):
        state = 0
        for step in range(max_steps):
            epsilon = episodes/(episodes+step)
            action = choose_action(state, epsilon, mat_q)
            state, reward, end = application_action(action, state, space)
            mat_q, state = onestep(mat_q, state, reward, action)
            if reward == 1:
                win_count +=reward
            if end:
                break

    # Print the learned Q-matrix
    print("Learned Q-matrix:\n", mat_q)

    # Test the learned policy
    state = 0
    end = False
    path = []
    path.append(state)
    while not end:
        action = np.argmax(mat_q[state])
        state, reward, end = application_action(action, state, space)
        path.append(state)
        if len(path)>100:
            break
    print(path)

    return 0


'''
Main function
'''
def main():
    Q_learning()


if __name__ == "__main__":
    main()