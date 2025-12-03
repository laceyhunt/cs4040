# import gym
import gymnasium as gym
import numpy as np
import random

# Initialize taxi-v3
environment_name = "Taxi-v3"
env = gym.make(environment_name)

# Hyperparameters
alpha = 0.9 # learning rate
gamma = 0.95 # Discount factor
epsilon = 1.0 # Lots of randomness at first
epsilon_decay = 0.9995 # slow down random exploration rate as we go
epsilon_min = 0.01 # Keeps us from being in a place where we never explore random
num_episodes = 10000
max_steps = 100 # Max steps per episode, terminates if you are stuck too long

# Initialize Q Table
action_size = env.action_space.n
state_size = env.observation_space.n 
q_table = np.zeros((state_size, action_size))
# print(q_table)

def choose_action(state):
   if random.uniform(0,1) < epsilon:
      # Explore: Choose a random action
      return env.action_space.sample()
   else:
      # Exploit: choose the best known action
      return np.argmax(q_table[state, :])
   
def update_q_table(state, action, reward, next_state):
   best_next_action = np.argmax(q_table[next_state, :])
   # Gamma - discount factor
   td_target = reward + gamma*q_table[next_state, best_next_action]
   td_error = td_target - q_table[state, action]
   # Alpha = learning rate
   q_table[state, action] += alpha*td_error
   
   # TD Target: expected reward of current action plus discounted future rewards
   # TD Error: difference between TD target and current Q value

# Train
for episode in range(num_episodes):
   # state = env.reset()
   state, info = env.reset()
   state = int(state)
   done = False
   # print(type(done))
   for step in range(max_steps):
      action = choose_action(state)
      # print(type(action))
      # next_state, reward, done, info = env.step(action)
      next_state, reward, terminated, truncated, info = env.step(action)
      done = terminated or truncated
      update_q_table(state, action, reward, next_state)
      state = int(next_state)
      if done:
         break
      epsilon = max(epsilon_min, epsilon*epsilon_decay)
      
print(q_table)

# Test
env = gym.make(environment_name, render_mode = 'human')
for episode in range(5):
   # state = env.reset()
   state, info = env.reset()
   state = int(state)
   done = False
   total_rewards = 0
   for step in range(max_steps):
      env.render()
      action = np.argmax(q_table[state, :])
      # next_state, reward, done, info = env.step(action)
      next_state, reward, terminated, truncated, info = env.step(action)
      done = terminated or truncated
      total_rewards+=reward
      state = int(next_state)
      
      if done:
         print(f"Episode {episode+1}, Total Reward {total_rewards}")
         break

env.close()