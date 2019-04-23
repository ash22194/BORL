import numpy as np
from scipy.optimize import minimize
from ipdb import set_trace

def buildQfromV(V_interp, env, gamma, state_grid, action_grid):
    """
    V_interp    - Function handle that returns V values for input states
    env         - object of the environment class
    gamma       - discount factor
    state_grid  - tuple with grid-points in each dimension
    action_grid - ax(num_actions) array of actions
    """
    
    Q = np.zeros((action_grid.shape[1], state_grid.shape[1]))
    for a in range(action_grid.shape[1]):
        action = np.dot(action_grid[:,a][:,np.newaxis], np.ones((1, state_grid.shape[1])))
        states_next, rewards, are_terminal = env.dynamics(state_grid, action)
        Q[a,:] = rewards + gamma*V_interp(states_next)*np.logical_not(are_terminal)

    return Q

def sparsifyGP(x, y, mu_y, kernel, sigma0, compressed_size=0, additional_inputs={}):
    """
    kernel            - function handle for the kernel 
    x                 - input points (dx x n)
    y                 - scalar observations (1 x n)
    mu_y              - function handle for the mean
    compressed_size   - size of the sparsified GP  
    additional_inputs - kernel matrix (K), kernel matrix inverse (K_inv)
    """

    if (compressed_size==0):
        compressed_size = x.shape[1]
        
    K = additional_inputs.get('K')
    if (K is None):
        K = computeCovarianceMatrix(x, kernel)
    K_nn = K.diagonal()[:,np.newaxis]
    
    """
    p(y|X, X_) = N(0, K_nm(K_m_inv)K_mn + lambda + sigma^2I)
    """
    obj = lambda x_: computeNegLogProbabilityWithPseudoInputs(x, y, mu_y(x), x_, mu_y, K_nn, sigma0, kernel)
    x_0 = 3*np.random.rand(x.shape[0], compressed_size)
    res = minimize(obj, x_0, options={'maxiter':100000, 'disp':True}, method='L-BFGS-B')
    x_ = np.reshape(res.x, (x.shape[0], compressed_size))
    
    K_m, K_m_inv, K_mn = computePseudoCovarianceMatrices(x, x_, kernel)
    
    lambda_ = K_nn[:,0] + sigma0**2 - np.diag((np.dot(K_mn.T, np.dot(K_m_inv, K_mn))))
    lambda_inv = np.diag(1/lambda_)
    Q_inv = np.linalg.inv(K_m + np.dot(K_mn, np.dot(lambda_inv, K_mn.T)))
    alpha_ = np.dot(Q_inv,\
                    (-mu_y(x_).T + np.dot(K_mn, \
                                        np.dot(lambda_inv,\
                                                (y - mu_y(x)).T))\
                    ))
    C_ = K_m_inv - Q_inv
    
    return alpha_, C_, x_

def sparsifyGPApprox(x, y, mu_y, kernel, sigma0, compressed_size=0, additional_inputs={}):
    """
    kernel            - function handle for the kernel 
    x                 - input points (dx x n)
    y                 - scalar observations (1 x n)
    mu_y              - function handle for the mean
    compressed_size   - size of the sparsified GP  
    additional_inputs - kernel matrix (K), kernel matrix inverse (K_inv)
    """

    if (compressed_size==0):
        compressed_size = x.shape[1]
        
    K = additional_inputs.get('K')
    if (K is None):
        K = computeCovarianceMatrix(x, kernel)
    K_nn = K.diagonal()[:,np.newaxis]
    
    """
    p(y|X, X_) = N(0, K_nm(K_m_inv)K_mn + lambda + sigma^2I)
    """
    obj = lambda x_: computeNegLogProbabilityWithPseudoInputs(x, y, mu_y(x), x_, mu_y, K_nn, sigma0, kernel)
    x_0 = 3*np.random.rand(x.shape[0], compressed_size)
    res = minimize(obj, x_0, options={'maxiter':100000, 'disp':True}, method='L-BFGS-B')
    x_ = np.reshape(res.x, (x.shape[0], compressed_size))
    
    K_m, K_m_inv, K_mn = computePseudoCovarianceMatrices(x, x_, kernel)
    
    lambda_ = K_nn[:,0] + sigma0**2 - np.diag((np.dot(K_mn.T, np.dot(K_m_inv, K_mn))))
    lambda_inv = np.diag(1/lambda_)
    Q_inv = np.linalg.inv(K_m + np.dot(K_mn, np.dot(lambda_inv, K_mn.T)))
    mean_posterior = mu_y(x) - np.dot(K_mn.T, np.dot(K_m_inv, mu_y(x_).T)).T
    f_ = np.dot(K_m,\
                np.dot(Q_inv,\
                       np.dot(K_mn,\
                              np.dot(lambda_inv,\
                                     (y - mean_posterior).T))))
    
    return K_m_inv, f_, x_
    
def computeNegLogProbabilityWithPseudoInputs(x, y, mu_y, x_, mu_y_, K_nn, sigma0, kernel):
    """
    x      - dx x n
    y      - 1 x n
    mu_y   - mean function evaluated at x (1 x n)
    x_     - dx x m
    mu_y_  - mean function handle to evaluate at x_
    K_nn   - k(xi,xj) n x 1
    sigma0 - noise std.
    """
    if (len(x_.shape)==1):
        m = int(x_.shape[0]/x.shape[0])
        x_ = np.reshape(x_, (x.shape[0], m))
    elif (len(x_.shape)==2 and x_.shape[0]!=x.shape[0]):
        print('Check pseudo-inputs!')
    
    K_m, K_m_inv, K_mn = computePseudoCovarianceMatrices(x, x_, kernel)
    
    mean_posterior = mu_y - np.dot(K_mn.T, np.dot(K_m_inv, mu_y_(x_).T)).T
    cov_posterior = np.dot(K_mn.T, np.dot(K_m_inv, K_mn))
    np.fill_diagonal(cov_posterior, K_nn + sigma0**2)
    
    negLogProb = 0.5*(np.log(np.linalg.det(cov_posterior)) + np.dot((y - mean_posterior),\
                                                                 np.dot(np.linalg.inv(cov_posterior), (y - mean_posterior).T))[0,0])
    return negLogProb
    
def computePseudoCovarianceMatrices(x, x_, kernel):
    
    n = x.shape[1]
    m = x_.shape[1]
            
    K_m = computeCovarianceMatrix(x_, kernel)
    K_m_inv = np.linalg.inv(K_m)
    
    K_mn = np.empty((m, n), dtype=np.float64, order='C')
    for i in range(n):
        K_mn[:,i] = kernel(np.repeat(x[:,i][:,np.newaxis], m, axis=1), x_)[:,0]
    
    return K_m, K_m_inv, K_mn

def computeCovarianceMatrix(x, kernel):
    
    n = x.shape[1]
    K = np.empty((n,n), dtype=np.float64, order='C')
    for i in range(n):
        K[:,i] = kernel(np.repeat(x[:,i][:,np.newaxis], n, axis=1), x)[:,0]
        
    return K
    

    
    
    
    
    
    
    
    
    
