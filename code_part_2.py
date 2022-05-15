# -*- coding: utf-8 -*-
"""
Created on Sun May 15 10:26:47 2022

@author: matteo
"""

import numpy as np
import matplotlib.pyplot as plt



def map_step(x,y,alpha1,alpha2,beta):
    if (y<beta):
        x = alpha1*x
        y = y/beta
    else:
        x = (1-alpha2) + alpha2*x
        y = (y-beta)/(1-beta)
    return [x,y]


def generate_map(x0,y0,n_iter,alpha1,alpha2,beta):
    x=[]
    y=[]
    x.append(x0)
    y.append(y0)
    for i in range(n_iter):
        [x_new, y_new] = map_step(x[-1],y[-1],alpha1,alpha2,beta)
        x.append(x_new)
        y.append(y_new)
        
    return [x,y]
    
        


def ex3_1():
    
    alpha1 = 0.3
    alpha2 = 0.4 # alpha1+alpha2<=1
    beta = 0.6 # != 0.5 in the general case
    n_iter = 1000
    x0 = np.random.uniform()
    y0 = np.random.uniform()
    
    [x,y] = generate_map(x0,y0,n_iter,alpha1,alpha2,beta)
    
    plt.figure()
    plt.plot(x,y, linewidth = 0.1, label='Time evolution')
    plt.plot(x[0],y[0], 'x', label='Start point')
    plt.title("Generalized Baker's Map evolution sample")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    
    
def ex3_2():
    
    alpha_nom = 1
    alpha_den = 3
    alpha = alpha_nom/alpha_den
    beta = 0.6 # != 0.5 in the general case
    
    n_iter = 5
    n_realisations = 5000000
    
    x = []
    y = []
    
    for n in range(n_realisations):
        x0 = np.random.uniform()
        y0 = np.random.uniform()
        [x_n,y_n] = generate_map(x0,y0,n_iter,alpha,alpha,beta)
        x.append(x_n[-1])
        y.append(y_n[-1])
        
        
    #n_divisions = int(1/alpha1**n_iter)
    r = (1/alpha_den)**n_iter
    n_divisions = int(alpha_den**n_iter)
    count_matrix = np.zeros([n_divisions,n_divisions])
    for i in range(len(x)):
        module_x = x[i] // r
        module_y = y[i] // r
        count_matrix[int(module_x), int(module_y)] = 1
        
    
    D0 = np.log(1/np.count_nonzero(count_matrix))/np.log(r)
    print ("D0 = ", D0)
    

def ex3_3():
    alpha = 0.4
    J = np.array([[alpha, 0],[0, 2]])
    n_iter = 50
    
    h = [1,0]
    for i in range(n_iter):
        h = J @ h
    lambda_ = np.log(np.linalg.norm(h))/n_iter
    print ("Expected Lambda 1 = ", np.log(alpha), "Numerical Lambda 1 = ", lambda_)
    
    h = [0,1]
    for i in range(n_iter):
        h = J @ h
    lambda_ = np.log(np.linalg.norm(h))/n_iter
    print ("Expected Lambda 2 = ", np.log(2), "Numerical Lambda 2 = ", lambda_)
    
    
    
def main():
    #ex3_1()
    #ex3_2()
    ex3_3()
    
    
    
if __name__ == '__main__':
    main()