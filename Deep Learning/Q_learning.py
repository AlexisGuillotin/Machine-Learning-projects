from __future__ import print_function
import numpy as np
import numpy as np
import random
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


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

def application_action(action, state, dragons):
    '''
    Function to apply the action and return the new position on the maze.
    :param action: The action to be performed by the agent.
    :param current_pos: The current position of the agent.
    :param dragons: all dragons.
    :return new_pos: The new position of the agent after performing the action.
    :return end: A flag indicating whether the episode has ended.
    '''
    reward = 0
    end = False
    if action == 0:
        if state <15:
            state += 1
        else:
            reward = -0.1
            return state, reward, end

    if action == 1:
        if state >=1:
            state -= 1
        else:
            reward = -0.1
            return state, reward, end

    if action == 2:
        if state >3:
            state -=4
        else:
            reward = -0.1
            return state, reward, end

    if action == 3:
        if state < 12:
            state +=4
        else:
            reward = -0.1
            return state, reward, end

    if state in dragons:
        reward == -1
        end == True
    elif state == 15:
        reward = 10
        end = True
    else:
        reward = -0.1
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
    # Use the 
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
    for episode in range(1, episodes):
        state = 0
        for step in range(max_steps):
            epsilon = episode/(episode+step)
            action = choose_action(state, epsilon, mat_q)
            next_state, reward, end = application_action(action, state, dragons)
            mat_q, state = onestep(mat_q, state, reward, action)
            state = next_state
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
        mat_q[state][action] = -100
        state, reward, end = application_action(action, state, space)
        path.append(state)

        if len(path)>100:
            break
    print(path)

    return 0

class DeepQNetwork:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(tf.keras.layers.Dense(32, activation='relu'))
        model.add(tf.keras.layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action2(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action


def application_action2(state, action):
    maze = np.array(state).reshape((4, 4))  # reshape flattened state back into maze form
    x, y = np.argwhere(maze == 1)[0]  # find current position of agent in maze
    if action == 0:  # move up
        y = max(0, y - 1)
    elif action == 1:  # move down
        y = min(3, y + 1)
    elif action == 2:  # move left
        x = max(0, x - 1)
    elif action == 3:  # move right
        x = min(3, x + 1)
    if maze[y][x] == 1:  # agent hits a wall
        reward = -10
        done = True
    elif maze[y][x] == 2:  # agent reaches the goal
        reward = 10
        done = True
    else:  # agent moves to an empty space
        reward = 0
        done = False
        maze[y][x] = 3  # mark the new position of agent in the maze
    next_state = maze.flatten().reshape(1, 16)  # flatten maze to a state vector
    return next_state, reward, done


def DeepQ_learning():
    EPISODES = 1000
    BATCH_SIZE = 32

    # initialize the maze
    maze = [[0, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 0, 0, 0],
            [1, 1, 0, 2]]

    state_size = 16  # flatten maze size
    action_size = 4  # up, down, left, right
    agent = DeepQNetwork(state_size, action_size)

    for e in range(EPISODES):
        state = np.array(maze).flatten().reshape(1, state_size)
        for time in range(100):
            action = agent.choose_action2(state)
            next_state, reward, done = application_action2(state, action)  # perform action in emulator and observe reward and next state
            next_state = np.array(next_state).flatten().reshape(1, state_size)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                break


'''
Main function
'''
def main():

    #Part 2
    Q_learning()

    # Part 3
    DeepQ_learning()


if __name__ == "__main__":
    main()