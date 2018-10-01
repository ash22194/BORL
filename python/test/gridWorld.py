import env
from value.QLearning import QLearningDiscrete
from value.GPTD import GPTD
import numpy as np
import gym
import matplotlib.pyplot as plt

def dict_to_array(data,n,m):
    d_ = np.zeros((n,m))
    for d in range(n*m):
        d_[int((d-d%m)/m),d%m] = data[d]
    return d_
        
def main():
    envName = 'FrozenLakeNotSlippery-v0'
    env = gym.make(envName)
    gamma = 0.95
    alpha = 0.01
    epsilon = 0.1
    num_episodes = 150
    max_length_episode = 300
    test_every = 10
    agent = QLearningDiscrete(env, gamma, epsilon, alpha)
    agent.train(num_episodes, max_length_episode, test_every)
    policy = agent.get_policy(range(env.observation_space.n))
    V = agent.get_value_function(range(env.observation_space.n))
    p = dict_to_array(policy,4,4)
    v = dict_to_array(V,4,4)
    
    plt.imshow(p)
    plt.colorbar()
    plt.title('Policy')
    plt.savefig(envName+'_policy')
    plt.close()
    plt.imshow(v)
    plt.colorbar()
    plt.title('Value Function')
    plt.savefig(envName+'_valueFunction')
    plt.close()

if __name__=='__main__':
    main()