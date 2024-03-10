import numpy as np
import gym
import random
import matplotlib.pyplot as plt

class QLAgent:

    def __init__(self,env) -> None:
        self.env = env
        self.q_table = np.zeros([env.observation_space.n, env.action_space.n])

    def train(self, neps=10000, alpha = 0.1, gamma = 0.6, epsilon = 0.1):
        rewards = []
        for i in range(1, neps):
            state,info = self.env.reset()
            
            epochs, penalties, reward, = 0, 0, 0
            done = False
            sum_rewards = 0
            while not done:
                if random.uniform(0, 1) < epsilon:
                    action = self.env.action_space.sample() # Explore action space
                else:
                    action = np.argmax(self.q_table[state]) # Exploit learned values

                next_state, reward, done, truncated, info = self.env.step(action) 
                
                old_value = self.q_table[state, action]
                next_max = np.max(self.q_table[next_state])
                
                new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
                self.q_table[state, action] = new_value

                state = next_state
                # epochs += 1
                # print(f"epochs : {epochs} | reward {reward}")
                sum_rewards += reward
            rewards.append(sum_rewards)
            if i % 100 == 0:
                print(f"Episode: {i}, reward = {sum_rewards}")

        plt.plot(rewards)
        plt.xlabel("step")
        plt.ylabel("summed reward")
        plt.show()
        print("Training finished.\n")
        
    
    def play(self):
        state,info = self.env.reset()
        done = False
        steps_till_done = 0
        while not done:
            action = np.argmax(self.q_table[state])
            state, reward, done, truncated, info = self.env.step(action) 
            steps_till_done += 1
        print(f"play after train - steps until done: {steps_till_done}")
