import numpy as np

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



def main():
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

if __name__ == "__main__":
    main()