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


def define_variables():
    global f 
    f = 20e3
    global L
    L = 6

def get_data():
    
    print ("READING DATA")
    
    for i in tqdm(range(1,7)):
        new_data = pd.read_csv("data/"+str(i)+".txt", names=[str(i)])
        if i == 1:
            data = new_data
        else:
            data = data.join(new_data)
    return data
        
       
        
def part1_1(data):
    
    N = 1000000
    
    # POINT 1
    plt.figure()
    for i in range(1,7):
        vm = np.mean(data[str(i)])
        max_index = int(f/vm)
        x = i - vm*range(max_index)/f # taylor hypothesis
        plt.plot(x,data[str(i)][N:max_index+N], label='Anemometer '+str(i))
    plt.xlabel('Distance (m)')
    plt.ylabel('Downstream velocity (m/s)')
    plt.legend()
    
    # POINT 2
    U = np.zeros([6])
    for i in range(1,7):
        U[i-1] = np.mean(data[str(i)])
    print ('MEAN VELOCITY: ', U)
    
    # POINT 4
    sigmau = np.zeros([6])
    for i in range(1,7):
        sigmau[i-1] = np.std(data[str(i)])
        
    I = sigmau/U
    print ('TURBULENCE INTENSITY: ', I)
    
    
    

def part1_2(data):
    
    # POINT 1
    dx = 0.05
    Lc = np.zeros([6])
    
    for i in tqdm(range(1,7)):
        l = 0
        C = 1
        vm = np.mean(data[str(i)])
        u = np.array(data[str(i)]-vm)
        u = np.nan_to_num(u)
        while (C > 1/np.e):
            l = l + dx
            Dt = int(l*f/vm) # shift in space -> shift in time-series measurement
            C = np.mean(u[Dt:]*u[:-Dt]) / np.mean(u*u)
        Lc[i-1] = C
    print ('\nCORRELATION LENGTH: ', Lc)
    
    # POINT 3
    Lint = np.zeros([6])
    
    for i in tqdm(range(1,7)):
        l = 0
        C = 1
        vm = np.mean(data[str(i)])
        u = np.array(data[str(i)]-vm)
        u = np.nan_to_num(u)
        while (l < 5*Lc[i-1]): # it approximates the integral to infinite
            l = l + dx
            Dt = int(l*f/vm)
            C = np.mean(u[Dt:]*u[:-Dt]) / np.mean(u*u)
            Lint[i-1] = Lint[i-1] + C*dx # rectangular quadrature for integral
    
    print ('\nINTEGRAL SCALE: ', Lint)
    
    

def main():
    define_variables()
    data = get_data()
    part1_2(data)
    
    del data
    gc.collect()
    
    
    
if __name__ == '__main__':
    main()
