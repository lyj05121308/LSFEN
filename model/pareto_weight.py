from scipy.optimize import minimize
import tensorflow as tf
import numpy as np


def pareto_efficient_weights(prev_w,c,G):
    '''

    :param prev_w:[K,1]上一轮迭代各loss的权重
    :param c: [K,1]每个目标权重的下限约束
    :param G: [K,m],G[i,:]是第i个task对所有参数的梯度，m是所有待优化参数的个数
    :return:
    '''
    GGT = np.matmul(G,np.transpose(G))  #[K,K]
    e = np.ones(np.shape(prev_w))  #[k,1]
    e = e.reshape((-1,1))
    c = c.reshape((-1,1))
    # print(e.shape)
    m_up = np.hstack((GGT,e))  #[k,k+1]
    m_down = np.hstack((np.transpose(e),np.zeros((1,1))))   #[1,k+1]
    M = np.vstack((m_up,m_down))  #[k+1,k+1]

    z = np.vstack((-np.matmul(GGT,c),1-np.sum(c)))  #[k+1,1]

    MTM = np.matmul(np.transpose(M),M)
    w_hat = np.matmul(np.matmul(np.linalg.inv(MTM),M),z)  #[k+1,1]
    w_hat = w_hat[:-1] #[k,1]
    w_hat = np.reshape(w_hat,(w_hat.shape[0],))  #[k,]



    return active_set_method(w_hat,prev_w,c)[0]


def active_set_method(w_hat, prev_w, c):

    A = np.eye(len(c))
    cons = {'type':'eq','fun':lambda x:np.sum(x)-1}#等式约束
    bounds = [[0., None] for _ in range(len(w_hat))] #不等书约束，要求所有weight为非负
    result = minimize(lambda x: np.linalg.norm(A.dot(x) - w_hat),
                      x0 = prev_w,
                      method='SLSQP',
                      bounds = bounds,
                      constraints=cons)
    return result.x+c
