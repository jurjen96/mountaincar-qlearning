import random
import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('MountainCar-v0')
alpha = 0.1 # learning rate
gamma = 0.998 # discount factor
epsilon = 1.0 # used for Epsilon greedy strategy
epsilon_decay = 0.998 # Helps to go from a exploring state to a exploiting state
max_episode = 50000

SHOW_EVERY = 500
DISCRETE_OS_SIZE = [50] * len(env.observation_space.high) # OS => observation space
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE


q_table = np.random.uniform(low=-2., high=0.0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

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


aggr_episode_steps = {'episode': [], 'avg': [], 'min': [], 'max': []}
episode_rewards = []


for episode in range(max_episode):
    state = env.reset()
    step = 0
    episode_reward = 0
    while not done:
        action = get_action(state)
        new_state, reward, done, _ = env.step(action)
        # For some reason, the reward given by env.step never changes, even if it
        # reaches the goal of the game of getting to the top (which is at 0.5).
        # For this reason we set the reward manually when the cart reaches the top
        if new_state[0] > 0.5:
            reward = 0.0
            done = True

        update_q_table(new_state, state, action, reward)
        state = new_state
        step += 1
        episode_reward += reward
        if not episode % SHOW_EVERY:
            env.render()

    epsilon *= epsilon_decay
    done = False

    episode_rewards.append(episode_reward)
    if not episode % SHOW_EVERY:
        aggr_episode_steps['episode'].append(episode)
        aggr_episode_steps['avg'].append(sum(episode_rewards[-SHOW_EVERY:])/len(episode_rewards[-SHOW_EVERY:]))
        aggr_episode_steps['min'].append(min(episode_rewards[-SHOW_EVERY:]))
        aggr_episode_steps['max'].append(max(episode_rewards[-SHOW_EVERY:]))


env.close()

plt.plot(aggr_episode_steps['episode'], aggr_episode_steps['avg'], label='avg')
plt.plot(aggr_episode_steps['episode'], aggr_episode_steps['min'], label='min')
plt.plot(aggr_episode_steps['episode'], aggr_episode_steps['max'], label='max')
plt.legend(loc=4) # Right bottom corner
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Best performing mountain car')

plt.show()
