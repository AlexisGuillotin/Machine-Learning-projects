import numpy as np
import random
import tensorflow as tf

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

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

def perform_action(state, action):
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


EPISODES = 1000
BATCH_SIZE = 32

# initialize the maze
maze = [[0, 0, 0, 1],
        [0, 1, 0, 1],
        [0, 0, 0, 0],
        [1, 1, 0, 0]]

state_size = 16  # flatten maze size
action_size = 4  # up, down, left, right
agent = DeepQNetwork(state_size, action_size)

for e in range(EPISODES):
    state = np.array(maze).flatten().reshape(1, state_size)
    for time in range(100):
        action = agent.act(state)
        next_state, reward, done = perform_action(state, action)  # perform action in emulator and observe reward and next state
        next_state = np.array(next_state).flatten().reshape(1, state_size)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            break
    if len(agent.memory) > BATCH_SIZE:
        agent.replay(BATCH_SIZE)
    if e % 10 == 0:
        agent.save("maze-dqn.h5")
