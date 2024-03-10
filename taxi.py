import numpy as np
import gym
from ql import QLAgent

env = gym.make("Taxi-v3",render_mode="human").env

ql = QLAgent(env)

ql.train()

ql.play()



