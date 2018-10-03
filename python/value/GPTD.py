import numpy as np
import gym
import GPy
from copy import deepcopy
from numpy.linalg import norm
from ipdb import set_trace

class GPTD:
    def __init__(self, env, nu, sigma0, gamma, kernel):
        
        self.env = env
        self.nu = nu
        self.gamma = gamma
        self.sigma0 = sigma0
        self.kernel = kernel.kernel
        self.D = np.array([[]])
        self.A = np.array([[]])
        self.H_ = np.array([[]])
        self.Q_ = np.array([[]])
        self.K_ = np.array([[]])
        self.K_inv = np.array([[]])
        self.alpha_ = np.array([[]])
        self.C_ = np.array([[]])

    def k_(self,x):

        if (len(x.shape)==1):
            x = x[:,np.newaxis]
        assert (len(x.shape)==2 and x.shape[1]==1), "Check state dimensions"

        k_ = []
        for d in range(self.D.shape[1]):
            k_.append(np.array([self.kernel(self.D[:,d],x[:,0])]))

        return np.array(k_)

    def update(self, xt, xt_1, r, gamma):
        """
        xt and xt_1 are numpy arrays of dimension dx1
        r is a scalar reward
        """
        # set_trace()
        if (self.D.shape[0]==1 and self.D.shape[1]==0):
            # Empty dictionary
            self.D = np.zeros((xt.shape[0],2))
            self.D[:,0] = xt_1[:,0]
            self.D[:,1] = xt[:,0]
            K_t = np.zeros((2,2))
            K_t[:,0] = self.k_(xt_1)[:,0]
            K_t[:,1] = self.k_(xt)[:,0]
            K_t_inv = np.linalg.inv(K_t)
            At = np.eye(2)
            H_t = np.dot(np.array([[1,-gamma]]),At)
            Q_t = np.linalg.inv(np.dot(H_t,np.dot(K_t,H_t.T))+np.diag([self.sigma0**2]))
            alpha_t = np.dot(H_t.T,Q_t)*r
            C_t = np.dot(H_t.T,np.dot(Q_t,H_t))
            # set_trace()
        else:

            K_t_1 = deepcopy(self.K_)
            K_t_1_inv = deepcopy(self.K_inv)
            alpha_t_1 = deepcopy(self.alpha_)
            C_t_1 = deepcopy(self.C_)
            Q_t_1 = deepcopy(self.Q_)
            At_1 = deepcopy(self.A)
            H_t_1 = deepcopy(self.H_)

            # Compute et
            k_t_1 = self.k_(xt_1)
            k_t = self.k_(xt)
            ktt = self.kernel(xt,xt)
            at_1 = np.dot(K_t_1_inv,k_t_1)
            at = np.dot(K_t_1_inv,k_t)
            et = ktt - np.dot(k_t.T,at)
            et = et[0][0]
            
            delk_t_1 = k_t_1 - gamma*k_t

            gt = np.dot(Q_t_1,np.dot(H_t_1,delk_t_1))
            if ((et - self.nu) > 10**(-4)):
                # If et > nu
                # Update D
                D = np.zeros((self.D.shape[0],self.D.shape[1]+1))
                D[:,:-1] = self.D
                D[:,-1] = xt[:,0]
                self.D = D

                # Compute alphat and Ct
                c_t = np.dot(H_t_1.T,gt) - at_1 # Added transpose... Mistake in the paper
                delktt = np.dot(at_1.T,(delk_t_1 - gamma*k_t)) + gamma**2*ktt
                s_t = self.sigma0**2 + delktt - np.dot(delk_t_1.T,np.dot(C_t_1,delk_t_1))

                K_t = np.zeros((K_t_1.shape[0]+1,K_t_1.shape[1]+1))
                K_t[:-1,:-1] = K_t_1
                K_t[-1,-1] = ktt
                K_t[-1,:-1] = k_t.T
                K_t[:-1,-1] = k_t[:,0]

                K_t_inv = np.zeros((K_t_1_inv.shape[0]+1,K_t_1_inv.shape[1]+1))
                K_t_inv[:-1,:-1] = et*K_t_1_inv+np.dot(at,at.T)
                K_t_inv[-1,-1] = 1
                K_t_inv[-1,:-1] = -at.T
                K_t_inv[:-1,-1] = -at[:,0]
                K_t_inv = 1/et*K_t_inv

                alpha_t = np.zeros((alpha_t_1.shape[0]+1,alpha_t_1.shape[1]))
                alpha_t[:-1,:] = alpha_t_1 + c_t/s_t*(np.dot(delk_t_1.T,alpha_t_1)-r)
                alpha_t[-1,:] = gamma/s_t*(np.dot(delk_t_1.T,alpha_t_1)-r)
                
                C_t = np.zeros((C_t_1.shape[0]+1,C_t_1.shape[1]+1))
                C_t[:-1,:-1] = C_t_1 + 1/s_t*np.dot(c_t,c_t.T)
                C_t[-1,-1] = gamma**2/s_t
                C_t[-1,:-1] = gamma/s_t*c_t.T
                C_t[:-1,-1] = gamma/s_t*c_t[:,0]
                
                Q_t = np.zeros((Q_t_1.shape[0]+1,Q_t_1.shape[1]+1))
                Q_t[:-1,:-1] = s_t*Q_t_1 + np.dot(gt,gt.T)
                Q_t[-1,-1] = 1
                Q_t[-1,:-1] = -gt.T
                Q_t[:-1,-1] = -gt[:,0]
                Q_t = Q_t/s_t
                
                At = np.zeros((At_1.shape[0]+1,At_1.shape[1]+1))
                At[:-1,:-1] = At_1
                At[-1,-1] = 1
                
                H_t = np.zeros((H_t_1.shape[0]+1,H_t_1.shape[1]+1))
                H_t[:-1,:-1] = H_t_1
                H_t[-1,:-1] = at_1.T
                H_t[-1,-1] = -gamma

            else:
                # else
                # D unchanged
                # Compute alphat and Ct
                # if (abs(et) > 10**(-6)):
                #     print('et : %f'%et)
                # assert abs(et)<10**(-5), "Negative projection error?"    

                h_t = at_1 - gamma*at
                # error = norm(delk_t_1-np.dot(K_t_1,h_t))
                # if (error > 10**(-6)):
                #     print('Error : %f'%error)
                # assert (error<10**(-5)), "Check delk_t_1"

                ct = np.dot(H_t_1.T,gt) - h_t # Added transpose... Mistake in the paper
                st = self.sigma0**2 - np.dot(ct.T,delk_t_1)
                
                K_t = K_t_1
                K_t_inv = K_t_1_inv

                alpha_t = alpha_t_1 + ct/st*(np.dot(delk_t_1.T,alpha_t_1) - r)
                
                C_t = C_t_1 + 1/st*np.dot(ct,ct.T)
                
                Q_t = np.zeros((Q_t_1.shape[0]+1,Q_t_1.shape[1]+1))
                Q_t[:-1,:-1] = st*Q_t_1 + np.dot(gt,gt.T)
                Q_t[-1,-1] = 1
                Q_t[-1,:-1] = -gt.T
                Q_t[:-1,-1] = -gt[:,0]
                Q_t = Q_t/st
                
                At = np.zeros((At_1.shape[0]+1,At_1.shape[1]))
                At[:-1,:] = At_1
                At[-1,:] = at.T
                
                H_t = np.zeros((H_t_1.shape[0]+1,H_t_1.shape[1]))
                H_t[:-1,:] = H_t_1
                H_t[-1,:] = h_t.T

        self.K_ = K_t
        self.K_inv = K_t_inv
        self.alpha_ = alpha_t
        self.C_ = C_t
        self.Q_ = Q_t
        self.A = At
        self.H_ = H_t

    def build_posterior(self, policy, num_episodes, max_episode_length, debug):
        """
        policy is a dictionary for discrete and a function for continuous
        """

        s = self.env.reset()
        for e in range(num_episodes):    
            is_terminal = False
            num_steps = 0
            while ((num_steps < max_episode_length) and (not is_terminal)):
                num_steps+=1
                a = policy[s] # For discrete ... Let's worry about continuous later!
                s_, r, is_terminal, debug_info = self.env.step(a)
                xt = s_
                xt_1 = s
                if (not type(xt)==type(np.array([]))):
                    xt = np.array([[s_]])
                if (not type(xt_1)==type(np.array([]))):
                    xt_1 = np.array([[s]])
                
                self.update(xt, xt_1, r, self.gamma)
                s = s_

            s_ = self.env.reset()
            xt = s_
            xt_1 = s
            if (not type(xt)==type(np.array([]))):
                xt = np.array([[s_]])
            if (not type(xt_1)==type(np.array([]))):
                xt_1 = np.array([[s]])
            self.update(xt, xt_1, 0, 0)
            s = s_

            if (debug):
                print('Episode : %d, Dictionary size : %d'%(e,self.D.shape[1]))

    def get_value_function(self, states):
        V = dict()
        for s in states:
            s_ = s
            if (not type(s)==type(np.array([]))):
                s_ = np.array([[s]])    
            V[s] = np.dot(self.k_(s_).T,self.alpha_)
        return V