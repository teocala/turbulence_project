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
    
    
    
def main():
    define_variables()
    data = get_data()
    #ex1_1(data)
    #ex1_2(data)
    #ex1_3(data)
    ex1_4(data)
    
    del data
    gc.collect()
    
    
    
if __name__ == '__main__':
    main()
