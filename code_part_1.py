# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 16:51:26 2022

@author: matte
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
from scipy.signal import savgol_filter


def define_variables():
    global f 
    f = 20e3
    global viscosity
    viscosity = 1.5e-5

def get_data():
    
    print ("READING DATA")
    
    for i in tqdm(range(6)):
        new_data = pd.read_csv("data/"+str(i+1)+".txt", names=[str(i)])
        if i == 0:
            data = new_data
        else:
            data = data.join(new_data)
    return data
        
       
        
def ex1_1(data):
    
    N = 1000000
    
    # POINT 1
    plt.figure()
    for i in range(6):
        vm = np.mean(data[str(i)])
        max_index = int(f/vm)
        x = i - vm*range(max_index)/f # taylor hypothesis
        plt.plot(x,data[str(i)][N:max_index+N], label='Anemometer '+str(i+1))
    plt.xlabel('Distance (m)')
    plt.ylabel('Downstream velocity (m/s)')
    plt.legend()
    
    # POINT 2
    U = np.zeros([6])
    for i in range(6):
        U[i] = np.mean(data[str(i)])
    print ('MEAN VELOCITY: ', U)
    
    # POINT 4
    sigmau = np.zeros([6])
    for i in range(6):
        sigmau[i] = np.std(data[str(i)])
        
    I = sigmau/U
    print ('TURBULENCE INTENSITY: ', I)
    
    
    

def ex1_2(data):
    
    # POINT 1
    dx = 0.05
    global Lc
    Lc = np.zeros([6])
    print ('CORRELATION LENGTH:')
    for i in tqdm(range(6)):
        l = 0
        C = 1
        vm = np.mean(data[str(i)])
        u = np.array(data[str(i)]-vm)
        u = np.nan_to_num(u)
        while (C > 1/np.e):
            l = l + dx
            Dt = int(l*f/vm) # shift in space -> shift in time-series measurement
            C = np.mean(u[Dt:]*u[:-Dt]) / np.mean(u*u)
        Lc[i] = Dt*vm/f
    print (Lc)
    
    # POINT 3
    Lint = np.zeros([6])
    print ('INTEGRAL SCALE:')
    for i in tqdm(range(6)):
        l = 0
        C = 1
        vm = np.mean(data[str(i)])
        u = np.array(data[str(i)]-vm)
        u = np.nan_to_num(u)
        while (l < 5*Lc[i]): # it approximates the integral to infinite
            l = l + dx
            Dt = int(l*f/vm)
            C = np.mean(u[Dt:]*u[:-Dt]) / np.mean(u*u)
            Lint[i] = Lint[i] + C*dx # rectangular quadrature for integral
    print (Lint)
    
    
    
def ex1_3(data):
    
    # POINT 1
    print ("ENERGY SPECTRUM:")
    plt.figure()
    
    max_idx = int(1e6)
    N = data.shape[0]
    vm = np.mean(data['0'])
    L = N*vm/f
    E = np.zeros([N,6])
    dx = L/(N-1)
    dk = 2*np.pi/(N*dx)
    k = range(N)*dk
    
    for n in tqdm(range(6)):
        
        vm = np.mean(data[str(n)])
        u = data[str(n)] - vm
        u = np.nan_to_num(u)
        
        Ek1 = dx/np.sqrt(2*np.pi)*np.fft.fft(u)
        Ek2 = dx/np.sqrt(2*np.pi)*np.fft.ifft(u)
        Ek1 = 0.5*np.absolute(Ek1/np.sqrt(L))**2
        Ek2 = 0.5*np.absolute(Ek2/np.sqrt(L))**2
        
        E[:,n] = Ek1 + Ek2
        
        plt.loglog(k[:max_idx], savgol_filter(E[:max_idx,n], 301, 5), label='Anemometer '+str(n+1))
    
    plt.loglog(k[100:max_idx], np.power(k[100:max_idx],-5/3)*1e-4, label='$k^{-5/3}$')
    plt.legend()
    plt.xlabel("k")
    plt.ylabel("Energy Spectrum")
    
    # POINT 2
    for n in range(6):
        print (n+1, ": ", 0.5*np.var(data[str(n)]), np.sum(E[:,n])*dk)
        
    # POINT 4
    plt.loglog(1.2,0.11, 'k', marker="x") # 1 left
    plt.loglog(0.7,0.03, 'k', marker="x") # 2 left
    plt.loglog(0.5,0.026, 'k', marker="x") # 3 left
    plt.loglog(0.42,0.022, 'k', marker="x") # 4 left
    plt.loglog(0.34,0.02, 'k', marker="x") # 5 left
    plt.loglog(0.32,0.014, 'k', marker="x") # 6, left

    plt.loglog(250,0.00004, 'k', marker="x") # 1 right
    plt.loglog(170,0.000008, 'k', marker="x") # 2 right
    plt.loglog(150,0.000006, 'k', marker="x") # 3 right
    plt.loglog(140,0.000004, 'k', marker="x") # 4 right
    plt.loglog(130,0.0000024, 'k', marker="x") # 5 right
    plt.loglog(120,0.0000016, 'k', marker="x") # 6 right
    
        
 
def ex1_4(data):
    
    Lc = [0.36985652, 0.63973856, 0.77961905, 0.90965516, 1.00961929, 1.08953831] # from ex1_2, point 1
    L0 = [5.23598776, 8.97597901, 12.5663706, 14.9599650, 18.4799568, 19.6349541] # from ex1_3, point 4
    
    # POINT 1
    epsilon = np.zeros([6])
    for i in range(6):
        epsilon[i] = 0.5*np.power(np.var(data[str(i)]), 3/2)/Lc[i]
    print ('DISSIPATION RATES: ', epsilon)
    
    # POINT 2
    Rel = np.zeros([6])
    for i in range(6):
        lambda_ = np.sqrt(15*viscosity*np.var(data[str(i)])/epsilon[i])
        Rel[i] = np.sqrt(np.var(data[str(i)])) * lambda_ / viscosity
    print ('TAYLOR REYNOLDS: ', Rel)
    
    # POINT 3
    Re = np.zeros([6])
    for i in range(6):
        vm = np.mean(data[str(i)])
        Re[i] = vm*L0[i]/viscosity
    print ('REYNOLDS: ', Re)
    
    
def ex1_5(data):
    #POINT 1
    # the 3/2 comes from the sum of the 3 spatial components
    kinetic_energy = np.zeros([6])
    for i in range(6):
        kinetic_energy[i] = 3/2*np.var(data[str(i)])
    print ('KINETIC ENERGY: ', kinetic_energy)
    
    
    #POINT 2
    y = kinetic_energy
    d = range(1,7)
    n_steps = 100
    err = np.zeros([n_steps])
    k = np.zeros([n_steps])
    
    for i,d0 in enumerate(np.linspace(-4,0.9,n_steps)):
        x = np.column_stack((np.ones([6]), np.log(d-d0)))
        k = np.linalg.solve(x.T @ x, x.T @ np.log(y)) #least squares formula
        err[i] = np.sum((np.log(y)-k[1]*np.log(d-d0)-k[0])**2) #computation of error
            
 
    d0_opt = np.linspace(-4,0.9,n_steps)[np.argmin(err)]
    
    x = np.column_stack((np.ones([6]), np.log(d-d0_opt)))
    [logalpha_opt, b_opt] = np.linalg.solve(x.T @ x, x.T @ np.log(y))
    
    alpha_opt = np.exp(logalpha_opt)
    h_opt = b_opt/(2+b_opt) 
    
    x = np.linspace(1,6,100)
    plt.figure()
    plt.plot(d,y,'b*', label='Kinetic energy values')
    plt.plot(x,np.power(x-d0_opt,b_opt), label='(d-d0)^b')
    plt.plot(x,alpha_opt*np.power(x-d0_opt,b_opt), label='k1(d-d0)^b')
    plt.legend()
    plt.xlabel('Distance d')
    plt.ylabel('Kinetic energy')
    print("d0 = ", d0_opt, ", h = ", h_opt)
    
    
    # POINT 3
    
    plt.figure()
    
    for d0 in np.arange(0,1,0.1):
        plt.loglog(d-d0, kinetic_energy, label='d0 = '+'{:.1f}'.format(d0))
      
    q = -1.2
    x = np.arange(0.1,1,0.1)
    plt.loglog(x, 0.1*x**q, 'k--', label='k*d0^'+str(q))
    plt.xlabel('Distance d')
    plt.ylabel('Kinetic energy')
    plt.legend()
    
    print ('h from graphic method = '+'{:.1f}'.format( q/(q+2)))
    
    # POINT 4
    
    Lc = [0.36985652, 0.63973856, 0.77961905, 0.90965516, 1.00961929, 1.08953831] # from ex1_2, point 1
    plt.figure()
    plt.loglog(Lc, kinetic_energy, label='Experimental')
    plt.loglog(Lc, np.power(Lc,-3), label='Saffman\'s decay')
    plt.loglog(Lc, np.power(Lc,-5), label='Loitsyanskii\'s decay')
    plt.loglog(Lc, np.power(Lc,-2), label='Self similar decay')
    plt.xlabel('Lc')
    plt.ylabel('Kinetic energy')
    plt.legend()
    # Saffman's decay is the most accurate => h = -3/2 that is the same as in point 3 => same d0
    
    
    # POINT 6 - non torna perché -(1+2h)>0 ma E(k) dovrebbe diminuire
    """
    print ("ENERGY SPECTRUM:")
    plt.figure()
    
    max_idx = int(1e6)
    N = data.shape[0]
    vm = np.mean(data['0'])
    L = N*vm/f
    E = np.zeros([N,6])
    dx = L/(N-1)
    dk = 2*np.pi/(N*dx)
    k = range(N)*dk
    
    for n in tqdm(range(6)):
        
        vm = np.mean(data[str(n)])
        u = data[str(n)] - vm
        u = np.nan_to_num(u)
        
        Ek1 = dx/np.sqrt(2*np.pi)*np.fft.fft(u)
        Ek2 = dx/np.sqrt(2*np.pi)*np.fft.ifft(u)
        Ek1 = 0.5*np.absolute(Ek1/np.sqrt(L))**2
        Ek2 = 0.5*np.absolute(Ek2/np.sqrt(L))**2
        
        E[:,n] = Ek1 + Ek2
        
        plt.loglog(k[:max_idx], savgol_filter(E[:max_idx,n], 301, 5), label='Anemometer '+str(n+1))
    
    plt.loglog(k[100:max_idx], np.power(k[100:max_idx],-(1+2*h_opt))*1e-4, label='$k^{-(1+2h)}$')
    plt.legend()
    plt.xlabel("k")
    plt.ylabel("Energy Spectrum")
    """
    
    # POINT 7
    # Relation (10) with L0
    L0 = [5.23598776, 8.97597901, 12.5663706, 14.9599650, 18.4799568, 19.6349541] # from ex1_3, point 4
    y = L0
    d = range(1,7)
    n_steps = 100
    err = np.zeros([n_steps])
    k = np.zeros([n_steps])
    
    for i,d0 in enumerate(np.linspace(-4,0.9,n_steps)):
        x = np.column_stack((np.ones([6]), np.log(d-d0)))
        k = np.linalg.solve(x.T @ x, x.T @ np.log(y)) #least squares formula
        err[i] = np.sum((np.log(y)-k[1]*np.log(d-d0)-k[0])**2) #computation of error
    d0_opt = np.linspace(-4,0.9,n_steps)[np.argmin(err)] 
    # d0 is 0.207
    
    # Relation (12) with Re
    Re = [3672952.55752014, 6296359.54652325, 8814198.72317052, 10494242.76662383, 12963502.48543675, 13773078.89665386] # from ex1_4, point 3
    y = Re
    d = range(1,7)
    n_steps = 100
    err = np.zeros([n_steps])
    k = np.zeros([n_steps])
    for i,d0 in enumerate(np.linspace(-4,0.9,n_steps)):
        x = np.column_stack((np.ones([6]), np.log(d-d0)))
        k = np.linalg.solve(x.T @ x, x.T @ np.log(y)) #least squares formula
        err[i] = np.sum((np.log(y)-k[1]*np.log(d-d0)-k[0])**2) #computation of error
    d0_opt = np.linspace(-4,0.9,n_steps)[np.argmin(err)] 
    # d0 is 0.207
    
    # the two d0 are clearly different from the result in point 2, so d0 is in general different
    # however, the two d0 are here exactly the same because Re depends on L0
    
    
    
def main():
    define_variables()
    data = get_data()
    #ex1_1(data)
    #ex1_2(data)
    #ex1_3(data)
    #ex1_4(data)
    ex1_5(data)
    
    del data
    gc.collect()
    
    
    
if __name__ == '__main__':
    main()
