import random
import gym
import numpy as np

env = gym.make("MountainCar-v0")
alpha = 0.1 # learning rate
gamma = 0.9 # discount factor
epsilon = 1.0 # used for Epsilon greedy strategy
epsilon_decay = 0.998 # Helps to go from a exploring state to a exploiting state


DISCRETE_OS_SIZE = [20] * len(env.observation_space.high) # OS => observation space
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE


q_table = np.random.uniform(low=-2, high=0.0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))


done = False

def get_action(state):
    """ For selecting the action the car should take, we use Epsilon-Greedy
    algorithm. It first explores the environment by taking random actions, but
    as the game progresses it will change its behavior from exploring to exploiting
    the 'knowledge' it gained from taking the random steps."""
    if random.random() < epsilon:
        # Select a random action
        return random.choice(range(env.action_space.n))
    else:
        # Get the best action from the Q-table
        q = get_max_q(state)
        # Converting the state to a discrete value for the position [0] and the velocity [2]
        q_discrete = (state - env.observation_space.low) // discrete_os_win_size
        # Convert the content from floats to ints
        q_discrete = [int(i) for i in q_discrete]
        return np.where(q_table[q_discrete[0]][q_discrete[1]] == q)[0][0]

def set_q_value(q, state, action):
    """ Set the new Q-value for a certain state and action in the Q-table """
    q_discrete = (state - env.observation_space.low) // discrete_os_win_size
    q_discrete = [int(i) for i in q_discrete]
    q_table[q_discrete[0]][q_discrete[1]][action] = q

def get_q_value(state, action):
    return get_q_values(state)[action]

def get_q_values(state):
    q_discrete = (state - env.observation_space.low) // discrete_os_win_size
    q_discrete = [int(i) for i in q_discrete]
    return q_table[q_discrete[0]][q_discrete[1]]

def get_max_q(state):
    """ Get from a row of Q-values the maximum value """
    return max(get_q_values(state))

def update_q_table(new_state, state, action, reward):
    # new Q = Q + alpha * (Reward + gamma * maxQ in new state - Q)
    new_Q = get_q_value(state, action) + alpha * (reward + gamma * get_max_q(new_state) - get_q_value(state, action))
    set_q_value(new_Q, state, action)


state = env.reset()

for episode in range(10000):
    state = env.reset()

    while not done:
        action = get_action(state)
        new_state, reward, done, _ = env.step(action)
        # For some reason the reward given by env.step never changes, even if it
        # reaches the goal of the game of getting to the top (which is at 0.5).
        # For this reason we set the reward manually when the cart reaches the top
        if new_state[0] > 0.5:
            reward = 0.0
            done = True

        update_q_table(new_state, state, action, reward)
        state = new_state
        if episode >= 10000-10:
            env.render()

    epsilon *= epsilon_decay
    done = False

env.close()
