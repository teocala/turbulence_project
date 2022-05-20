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
    
    # POINT 1
    alpha1 = 0.3
    alpha2 = 0.2 # alpha1+alpha2<=1
    beta = 0.4 # != 0.5 in the general case
    n_sample = 1000
    
    fig, axs = plt.subplots(2, 3)
    for i in range(6):
        axs[i//3,i%3].set_xlabel("x")
        axs[i//3,i%3].set_ylabel("y")
        axs[i//3,i%3].set_title("Time-step "+str(i))
    for n in range(n_sample):
        x0 = np.random.uniform()
        y0 = np.random.uniform()
        [x,y] = generate_map(x0,y0,5,alpha1,alpha2,beta)
        if y0 < beta:
            for i in range(6):
                axs[i//3,i%3].plot(x[i],y[i], 'b.', markersize=0.8)
        else:
            for i in range(6):
                axs[i//3,i%3].plot(x[i],y[i], 'r.', markersize=0.8)
            
        
    fig.tight_layout(pad=1.5)
    
    
    
    # POINT 2
    alpha1 = 0.4
    alpha2 = 0.3 # alpha1+alpha2<=1
    beta = 0.4 # != 0.5 in the general case
    n_sample = 1000
    
    plt.figure()
    for n in range(n_sample):
        x0 = np.random.uniform()
        y0 = np.random.uniform()
        [x,y] = generate_map(x0,y0,0,alpha1,alpha2,beta)
        plt.plot(x[-1],y[-1], 'b.', markersize=0.8)
        
    plt.plot([0, 0], [0,1], 'k')   
    plt.plot([1, 1], [0,1], 'k') 
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("0 timestep distribution") 
    
    
    
    plt.figure()
    for n in range(n_sample):
        x0 = np.random.uniform()
        y0 = np.random.uniform()
        [x,y] = generate_map(x0,y0,1,alpha1,alpha2,beta)
        plt.plot(x[-1],y[-1], 'b.', markersize=0.8)
        
    plt.plot([0, 0], [0,1], 'k')   
    plt.plot([alpha1, alpha1], [0,1],'k')
    plt.plot([1-alpha2, 1-alpha2], [0,1],'k')
    plt.plot([1, 1], [0,1], 'k') 
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("1 timestep distribution") 
    
    
    
    plt.figure()
    for n in range(n_sample):
        x0 = np.random.uniform()
        y0 = np.random.uniform()
        [x,y] = generate_map(x0,y0,2,alpha1,alpha2,beta)
        plt.plot(x[-1],y[-1], 'b.', markersize=0.8)
        
    plt.plot([0, 0], [0,1], 'k')   
    plt.plot([alpha1**2, alpha1**2], [0,1], 'k')  
    plt.plot([(1-alpha2)*alpha1, (1-alpha2)*alpha1], [0,1],'k')
    plt.plot([alpha1, alpha1], [0,1],'k')
    plt.plot([1-alpha2, 1-alpha2], [0,1],'k')
    plt.plot([(1-alpha2) + alpha1*alpha2, (1-alpha2) + alpha1*alpha2], [0,1], 'k')  
    plt.plot([(1-alpha2) + (1-alpha2)*alpha2, (1-alpha2) + (1-alpha2)*alpha2], [0,1],'k')
    plt.plot([1, 1], [0,1], 'k') 
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("2 timesteps distribution") 
    
    
    
    plt.figure()
    for n in range(n_sample):
        x0 = np.random.uniform()
        y0 = np.random.uniform()
        [x,y] = generate_map(x0,y0,3,alpha1,alpha2,beta)
        plt.plot(x[-1],y[-1], 'b.', markersize=0.8)
        
    plt.plot([0, 0], [0,1], 'k')   
    plt.plot([alpha1**3, alpha1**3], [0,1], 'k')  
    plt.plot([(1-alpha2)*alpha1**2, (1-alpha2)*alpha1**2], [0,1],'k')
    plt.plot([alpha1**2, alpha1**2], [0,1],'k')
    plt.plot([(1-alpha2)*alpha1, (1-alpha2)*alpha1], [0,1],'k')
    plt.plot([(1-alpha2)*alpha1 + alpha1**2*alpha2, (1-alpha2)*alpha1 + alpha1**2*alpha2], [0,1], 'k')  
    plt.plot([(1-alpha2)*alpha1 + (1-alpha2)*alpha2*alpha1, (1-alpha2)*alpha1 + (1-alpha2)*alpha2*alpha1], [0,1],'k')
    plt.plot([alpha1, alpha1], [0,1], 'k') 
    
    plt.plot([(1-alpha2), (1-alpha2)], [0,1], 'k')   
    plt.plot([(1-alpha2) + alpha1**2*alpha2, (1-alpha2) + alpha1**2*alpha2], [0,1], 'k')  
    plt.plot([(1-alpha2) + (1-alpha2)*alpha1*alpha2, (1-alpha2) + (1-alpha2)*alpha1*alpha2], [0,1],'k')
    plt.plot([(1-alpha2) + alpha1*alpha2, (1-alpha2) + alpha1*alpha2], [0,1],'k')
    plt.plot([(1-alpha2) + (1-alpha2)*alpha2, (1-alpha2) + (1-alpha2)*alpha2], [0,1],'k')
    plt.plot([(1-alpha2) + (1-alpha2)*alpha2 + alpha1*alpha2**2, (1-alpha2) + (1-alpha2)*alpha2 + alpha1*alpha2**2], [0,1], 'k')  
    plt.plot([(1-alpha2) + (1-alpha2)*alpha2 + (1-alpha2)*alpha2**2, (1-alpha2) + (1-alpha2)*alpha2 + (1-alpha2)*alpha2**2], [0,1],'k')
    plt.plot([1, 1], [0,1], 'k') 
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("3 timesteps distribution") 

           
    
    
    
    
    
    
    
    # POINT 3
    alpha1 = 0.3
    alpha2 = 0.4 # alpha1+alpha2<=1
    beta = 0.3 # != 0.5 in the general case
    n_iter = 20
    
    x0 = 0.8
    y0 = 0.8
    [x,y] = generate_map(x0,y0,n_iter,alpha1,alpha2,beta)
    plt.figure()
    plt.plot(x,y,  'tab:blue', linewidth = 0.5, label='Time evolution 1')
    plt.plot(x[0],y[0], 'rx', label='Start point 1')
    
    x1 = 0.7999
    y1 = 0.7999
    [x,y] = generate_map(x1,y1,n_iter,alpha1,alpha2,beta)
    
    plt.plot(x,y, 'tab:orange', linewidth = 0.5, label='Time evolution 2')
    plt.plot(x[0],y[0], 'bx', label='Start point 2')
    plt.title("Evolution of trajectories from two near points")
    plt.xlabel("x")
    plt.ylabel("y")
    #plt.legend()
    
    

    n_iter = 90
    plt.figure()
    
    [x,y] = generate_map(x0,y0,n_iter,0.3,0.4,0.4)
    T1 = np.array([x,y])
    [x,y] = generate_map(x1,y1,n_iter,0.3,0.4,0.4)
    T2 = np.array([x,y])
    plt.semilogy(np.linalg.norm(T1-T2,axis=0), label=r"$\alpha_1=0.3, \alpha_2=0.4, \beta=0.4$")
    
    [x,y] = generate_map(x0,y0,n_iter,0.3,0.4,0.2)
    T1 = np.array([x,y])
    [x,y] = generate_map(x1,y1,n_iter,0.3,0.4,0.2)
    T2 = np.array([x,y])
    plt.semilogy(np.linalg.norm(T1-T2,axis=0), label=r"$\alpha_1=0.3, \alpha_2=0.4, \beta=0.2$")
    
    [x,y] = generate_map(x0,y0,n_iter,0.3,0.2,0.4)
    T1 = np.array([x,y])
    [x,y] = generate_map(x1,y1,n_iter,0.3,0.2,0.4)
    T2 = np.array([x,y])
    plt.semilogy(np.linalg.norm(T1-T2,axis=0), label=r"$\alpha_1=0.3, \alpha_2=0.2, \beta=0.4$")
    
    [x,y] = generate_map(x0,y0,n_iter,0.1,0.4,0.4)
    T1 = np.array([x,y])
    [x,y] = generate_map(x1,y1,n_iter,0.1,0.4,0.4)
    T2 = np.array([x,y])
    plt.semilogy(np.linalg.norm(T1-T2,axis=0), label=r"$\alpha_1=0.1, \alpha_2=0.4, \beta=0.4$")
    
    
    plt.xlabel("Time step n")
    plt.ylabel("$\epsilon (n)$")
    plt.title("Distance between trajectories (emergence of chaos)")
    plt.legend()
    
    
def ex3_2():
    
    alpha_nom = 1
    alpha_den = 5
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
    
    h = [0.5,0]
    
    h = h/np.linalg.norm(h)
    for i in range(n_iter):
        h = J @ h
    lambda_ = np.log(np.linalg.norm(h))/n_iter
    print ("Expected Lambda 1 = ", np.log(alpha), "Numerical Lambda 1 = ", lambda_)
    
    h = [0.1,0.05] # this vector can be defined in any way except parallel to [1,0]. Indeed, it will anyway converge to the maximal eigenvalue
    
    h = h/np.linalg.norm(h)
    for i in range(n_iter):
        h = J @ h
    lambda_ = np.log(np.linalg.norm(h))/n_iter
    print ("Expected Lambda 2 = ", np.log(2), "Numerical Lambda 2 = ", lambda_)
    
    
    
    
    
    n_iter = 80
    alpha = 0.4
    
    l1 = np.log(alpha)
    l2 = np.log(2)
    
    
    plt.figure()
    x0 = 0.8
    y0 = 0.8
    x1 = 0.7999
    y1 = 0.7999
    [x,y] = generate_map(x0,y0,n_iter,alpha,alpha,0.5)
    print (x)
    T1 = np.array([x,y])
    [x,y] = generate_map(x1,y1,n_iter,alpha,alpha,0.5)
    T2 = np.array([x,y])
    plt.semilogy(np.linalg.norm(T1-T2,axis=0), label=r"$\epsilon(n)$")
    plt.semilogy(range(13), 2e-4*np.exp(l2*range(13)), label=r"$e^{λ_2 n}$")
    plt.semilogy(range(52,80), 1e20*np.exp(l1*range(52,80)), label=r"$e^{λ_1 n}$")
    plt.xlabel("Timestep n")
    plt.ylabel(r"$\epsilon(n)$")
    plt.legend()
    plt.title(r"Distance between trajectories in the case of $\beta=0.5$")
    
    
    
def main():
    #ex3_1()
    #ex3_2()
    ex3_3()
    
    
    
if __name__ == '__main__':
    main()