import env
import numpy as np
import gym
import matplotlib.pyplot as plt
from value.QLearning import QLearningDiscrete
from value.GPTD import GPTD
from utils.kernels.GaussianKernel import GaussianKernel
from ipdb import set_trace


def dict_to_array(data,n,m):
    d_ = np.zeros((n,m))
    for d in range(n*m):
        d_[int((d-d%m)/m),d%m] = data[d]
    return d_
        
def main():
    '''
    Use Q-Learning to learn a policy
    '''
    envName = 'FrozenLakeNotSlippery-v0'
    env = gym.make(envName)
    gamma = 0.95
    alpha = 0.01
    epsilon = 0.1
    num_episodes = 10000
    max_length_episode = 300
    test_every = 10
    debug = False
    agent = QLearningDiscrete(env, gamma, epsilon, alpha)
    agent.train(num_episodes, max_length_episode, test_every, debug)
    policy = agent.get_policy(range(env.observation_space.n))
    V = agent.get_value_function(range(env.observation_space.n))

    '''
    Use the learnt policy to build a posterior over the value function
    '''
    nu = 0
    sigma0 = 0
    sigmak = 0.2
    kernel = GaussianKernel(sigmak)
    gptd = GPTD(env, nu, sigma0, gamma, kernel)
    debug = True
    num_episodes = 150
    gptd.build_posterior(policy, num_episodes, max_length_episode, debug)
    V_gptd = gptd.get_value_function(range(env.observation_space.n))
    set_trace()
    '''
    Plot
    '''
    p = dict_to_array(policy,4,4)
    v = dict_to_array(V,4,4)
    v_gptd = dict_to_array(V_gptd,4,4)    

    plt.imshow(p)
    plt.colorbar()
    plt.title('Policy')
    # plt.savefig(envName+'_policy')
    plt.show()
    plt.close()

    plt.imshow(v)
    plt.colorbar()
    plt.title('Value Function - Q-Learning')
    # plt.savefig(envName+'_valueFunction')
    plt.show()
    plt.close()

    plt.imshow(v_gptd)
    plt.colorbar()
    plt.title('Value Function - GPTD')
    # plt.savefig(envName+'_valueFunctionGPTD')
    plt.show()
    plt.close()
    set_trace()

if __name__=='__main__':
    main()
