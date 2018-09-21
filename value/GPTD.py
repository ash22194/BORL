import numpy as np
import gym
import GPy
from copy import deepcopy


class GPTD:
    def __init__(self, nu, sigma0, gamma, kernel):
        
        self.nu = nu
        self.gamma = gamma
        self.sigma0 = sigma0
        self.kernel = kernel
        self.D = np.array([[]])
        self.A = np.array([[]])
        self.H_ = np.array([[]])
        self.Q_ = np.array([[]])
        self.K_ = np.array([[]])
        self.K_inv = np.array([[]])
        self.alpha_ = np.array([[]])
        self.C_ = np.array([[]])
        # self.r = np.array([[]])

    def k_(self,x):

        if (len(x.shape)==1):
            x = x[:,np.newaxis]
        assert (len(x.shape)==2 and x.shape[1]==1), "Check state dimensions"

        k_ = []
        for d in self.D:
            k.append(self.kernel(d,x)[0])

        return np.array(k_)

    def update(self, xt, xt_1, r):
        """
        xt and xt_1 are numpy arrays of dimension dx1
        r is a scalar reward
        """
        if (self.D.shape[0]==1 and self.D.shape[1]==0):
            # Empty dictionary
            self.D = np.zeros(xt.shape[0],2)
            self.D[:,0] = xt_1[:,0]
            self.D[:,1] = xt[:,0]
            K_t = np.zeros(2,2)
            K_t[:,0] = self.k_(xt_1)[:,0]
            K_t[:,1] = self.k_(xt)[:,0]
            K_t_inv = np.linalg.inv(K_t)
            At = np.eye(2)
            H_t = np.dot(np.array([[1,-self.gamma]]),self.A)
            Q_t = np.linalg.inv(np.dot(H_t,np.dot(K_t,H_t.T))+np.diag([self.sigma0**2]))
            alpha_t = np.dot(H_t,Q_t)*r
            C_t = np.dot(H_t.T,np.dot(Q_t,H_t))

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
            
            delk_t_1 = k_t_1 - self.gamma*k_t
            assert (delk_t_1==np.dot(K_t_1,h_t)), "Check delk_t_1"

            gt = np.dot(Q_t_1,np.dot(H_t_1,delk_t_1))

            if (et > self.nu):
                # If et > nu
                # Update D
                D = np.zeros(self.D.shape[0],self.D.shape[1]+1)
                D[:,:-1] = self.D
                D[:,-1] = xt[:,0]
                self.D = D
                print("Dictionary updated")


                # Compute alphat and Ct
                c_t = np.dot(H_t_1,gt) - at_1
                delktt = np.dot(at_1.T,(delk_t_1 - self.gamma*k_t_1)) + self.gamma**2*ktt
                s_t = self.sigma0**2 + delktt - np.dot(delk_t_1.T,np.dot(C_t_1,delk_t_1))

                K_t = np.zeros(K_t_1.shape[0]+1,K_t_1.shape[1]+1)
                K_t[:-1,:-1] = K_t_1
                K_t[-1,-1] = ktt
                K_t[-1,:-1] = k_t.T
                K_t[:-1,-1] = k_t[:,0]

                K_t_inv = np.zeros(K_t_1_inv.shape[0]+1,K_t_1_inv.shape[1]+1)
                K_t_inv[:-1,:-1] = et*K_t_1_inv+np.dot(at,at.T)
                K_t_inv[-1,-1] = 1
                K_t_inv[-1,:-1] = -at.T
                K_t_inv[:-1,-1] = -at[:,0]
                K_t_inv = 1/et*K_t_inv

                alpha_t = np.zeros(alpha_t_1.shape[0]+1,alpha_t_1.shape[1])
                alpha_t[:-1,:] = alpha_t_1 + c_t/s_t*(np.dot(delk_t_1.T,alpha_t_1)-r)
                alpha_t[-1,:] = self.gamma/st*(np.dot(delk_t_1.T,alpha_t_1)-r)
                
                C_t = np.zeros(C_t_1.shape[0]+1,C_t_1.shape[1]+1)
                C_t[:-1,:-1] = C_t_1 + 1/s_t*np.dot(c_t,c_t.T)
                C_t[-1,-1] = self.gamma**2/s_t
                C_t[-1,:-1] = self.gamma/s_t*c_t.T
                C_t[:-1,-1] = self.gamma/s_t*c_t[:,0]
                
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
                H_t[-1,-1] = -self.gamma

            else:
                # else
                # D unchanged
                # Compute alphat and Ct

                h_t = at_1 - self.gamma*at
                ct = np.dot(H_t_1,gt) - h_t
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
        