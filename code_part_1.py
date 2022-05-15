# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 16:51:26 2022

@author: matteo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
import scipy
from scipy.signal import savgol_filter
import scipy.signal


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
    
    # POINT 1
    fig, axs = plt.subplots(2, 3)
    for i in range(6):
        vm = np.mean(data[str(i)])
        max_distance = 4 # i.e. a range of 4 meters
        max_index = int(max_distance*f/vm)
        x = i+1 - vm*range(max_index)/f # taylor hypothesis
        axs[i//3,i%3].plot(x,data[str(i)][:max_index])
        axs[i//3,i%3].set_title('Anenometer '+ str(i+1))
        axs[i//3,i%3].set_ylim([7,15])
        axs[i//3,i%3].set_xlabel('Distance (m)', fontsize=7)
        axs[i//3,i%3].set_ylabel('Velocity (m/s)', fontsize=7)
    
    fig.tight_layout(pad=1.5)
        
    
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
    
    # POINT 1,2
    
    Lc = np.zeros([6])
    Lint = np.zeros([6])
    
    print ('EX1_2')
    for i in tqdm(range(6)):
        
        vm = np.mean(data[str(i)])
        u = np.array(data[str(i)]-vm)
        u = np.nan_to_num(u)
        N = u.shape[0]
        C = scipy.signal.correlate(u,u,'same','fft')
        C = C[N//2:]
        C = C / range(N,N-C.shape[0],-1)
        C = C / np.mean(u*u)
        
        dt = np.where(C < 1/np.e)[0][0]
        Lc[i] = dt*vm/f
        
        dx = vm/f
        Lint[i] = C[:5*dt].sum()*dx
        
    print ('\nCORRELATION LENGTH:', Lc)
    print ('INTEGRAL SCALE: ', Lint)
    
    
    
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
        
        Ek1 = dx/np.sqrt(2*np.pi)*scipy.fft.fft(u)
        Ek2 = dx/np.sqrt(2*np.pi)*scipy.fft.ifft(u)
        Ek1 = 0.5*np.absolute(Ek1/np.sqrt(L))**2
        Ek2 = 0.5*np.absolute(Ek2/np.sqrt(L))**2
        
        E[:,n] = Ek1 + Ek2
        
        p = scipy.interpolate.interp1d(k[:max_idx],E[:max_idx,n])
        x = np.logspace(-2,3,10000,base=10)
        y = savgol_filter(p(x),501,1)
        plt.loglog(x[100:-100], y[100:-100], label='Anemometer '+str(n+1))
        
    
    plt.loglog(x[3000:7000], np.power(x[3000:7000],-5/3)*1e-4, label='$k^{-5/3}$')
    plt.legend()
    plt.xlabel("k (1/s)")
    plt.ylabel("Energy Spectrum ($m^2/s^2$")
    
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
    
    Lc = [0.36669985, 0.63447755, 0.77330634, 0.90386788, 1.00909318, 1.08532957] # from ex1_2, point 1
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
        l = L0[i]
        dt = int(l*f/vm)
        v = np.array(data[str(i)])
        v = np.nan_to_num(v)
        y = v[dt:] - v[:-dt]
        v0 = np.sqrt(np.mean(np.power(y,2)))
        Re[i] = v0*l/viscosity
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
    plt.plot(x, alpha_opt*np.power(x,b_opt), label='k1(d)^b')
    plt.plot(x,np.power(x-d0_opt,b_opt), label='(d-d0)^b')
    plt.plot(x,alpha_opt*np.power(x-d0_opt,b_opt), label='k1(d-d0)^b')
    plt.legend()
    plt.xlabel('Distance d (m)')
    plt.ylabel('Kinetic energy ($m^2/s^2$)')
    print("d0 = ", d0_opt, ", h = ", h_opt)
    
    
    # POINT 3
    
    plt.figure()
    
    for d0 in np.arange(0,1,0.1):
        plt.loglog(d-d0, kinetic_energy, label='d0 = '+'{:.1f}'.format(d0))
      
    q = -1.2
    x = np.arange(0.1,1,0.1)
    plt.loglog(x, 0.1*x**q, 'k--', label='k*d0^'+str(q))
    plt.xlabel('Distance d (m)')
    plt.ylabel('Kinetic energy ($m^2/s^2$)')
    plt.legend()
    
    print ('h from graphic method = '+'{:.1f}'.format( q/(q+2)))
    
    # POINT 4
    
    Lc = [0.36669985, 0.63447755, 0.77330634, 0.90386788, 1.00909318, 1.08532957] # from ex1_2, point 1
    plt.figure()
    plt.loglog(Lc, kinetic_energy, label='Experimental')
    plt.loglog(Lc, kinetic_energy[0]*np.power(Lc,-3)/np.power(Lc[0],-3), label='Saffman\'s decay')
    plt.loglog(Lc, kinetic_energy[0]*np.power(Lc,-5)/np.power(Lc[0],-5), label='Loitsyanskii\'s decay')
    plt.loglog(Lc, kinetic_energy[0]*np.power(Lc,-2)/np.power(Lc[0],-2), label='Self similar decay')
    plt.xlabel('Lc (m)')
    plt.ylabel('Kinetic energy ($m^2/s^2$)')
    plt.legend()
    # Saffman's decay is the most accurate => h = -3/2 that is the same as in point 3 => same d0
    
    
    # POINT 6
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
        
        Ek1 = dx/np.sqrt(2*np.pi)*scipy.fft.fft(u)
        Ek2 = dx/np.sqrt(2*np.pi)*scipy.fft.ifft(u)
        Ek1 = 0.5*np.absolute(Ek1/np.sqrt(L))**2
        Ek2 = 0.5*np.absolute(Ek2/np.sqrt(L))**2
        
        E[:,n] = Ek1 + Ek2
        
        p = scipy.interpolate.interp1d(k[:max_idx],E[:max_idx,n])
        x = np.logspace(-2,3,10000,base=10)
        y = savgol_filter(p(x),501,1)
        plt.loglog(x[100:-100], y[100:-100], label='Anemometer '+str(n+1))
        
    plt.loglog(x[:5000], np.power(x[:5000],-(1+2*h_opt))*5, label='$k^{-(1+2h)}$')
    plt.legend()
    plt.xlabel("k (1/s)")
    plt.ylabel("Energy Spectrum ($m^2/s^2$)")
    
    # POINT 7
    # Relation (10) with L0
    L0 = [5.23598776, 8.97597901, 12.5663706, 14.9599650, 18.4799568, 19.6349541] # from ex1_3, point 4
    y = L0
    d = range(1,7)
    n_steps = 100
    err = np.zeros([n_steps])
    k = np.zeros([n_steps])
    b = 1/(1-h_opt)
    
    for i,d0 in enumerate(np.linspace(-4,0.9,n_steps)):
        x = np.column_stack((np.ones([6]))).T
        ym = np.log(y)-b*np.log(d-d0)
        k = np.linalg.solve(x.T @ x, x.T @ ym) #least squares formula
        err[i] = np.sum((np.log(y)-b*np.log(d-d0)-k)**2) #computation of error
    d0_opt = np.linspace(-4,0.9,n_steps)[np.argmin(err)] 
    x = np.column_stack((np.ones([6]))).T
    ym = np.log(y)-b*np.log(d-d0_opt)
    logalpha_opt = np.linalg.solve(x.T @ x, x.T @ ym)
    alpha_opt = np.exp(logalpha_opt)
    
    print ("From L0: d0=",d0_opt, ", alpha=",alpha_opt,", h=",h_opt)
    x = np.linspace(1,6,100)
    plt.figure()
    plt.plot(d,y,'b*', label='L0 values')
    plt.plot(x,alpha_opt*np.power(x-d0_opt,b), label='k1(d-d0)^b')
    plt.legend()
    plt.xlabel('Distance d (m)')
    plt.ylabel('L0 (m)')
    
    # Relation (12) with Re
    Re = [646287.15855169, 500801.04127716, 499838.32346323, 481861.24611346,
     505926.82006497, 476626.71192208] # ex1_4, point 3
    y = Re
    d = range(1,7)
    n_steps = 100
    err = np.zeros([n_steps])
    k = np.zeros([n_steps])
    b = (1+h_opt)/(1-h_opt)
    
    for i,d0 in enumerate(np.linspace(-4,0.9,n_steps)):
        x = np.column_stack((np.ones([6]))).T
        ym = np.log(y)-b*np.log(d-d0)
        k = np.linalg.solve(x.T @ x, x.T @ ym) #least squares formula
        err[i] = np.sum((np.log(y)-b*np.log(d-d0)-k)**2) #computation of error
    d0_opt = np.linspace(-4,0.9,n_steps)[np.argmin(err)] 
    x = np.column_stack((np.ones([6]))).T
    ym = np.log(y)-b*np.log(d-d0_opt)
    logalpha_opt = np.linalg.solve(x.T @ x, x.T @ ym)
    alpha_opt = np.exp(logalpha_opt)
    
    print ("From Re: d0=",d0_opt, ", alpha=",alpha_opt,", h=",h_opt)
    x = np.linspace(1,6,100)
    plt.figure()
    plt.plot(d,y,'b*', label='Re values')
    plt.plot(x,alpha_opt*np.power(x-d0_opt,b), label='k1(d-d0)^b')
    plt.legend()
    plt.xlabel('Distance d (m)')
    plt.ylabel('Reynolds number (adim.)')
    
    # the two d0 are clearly different from the result in point 2, so d0 is in general different
    # however, the two d0 are here exactly the same because Re depends on L0



def ex1_6(data):
    # POINT 1
    dist = [0.001, 0.01, 0.1, 10]
    fig, axs = plt.subplots(2, 2)
    
    for i,l in enumerate(dist):
        vm = np.mean(data[str(0)])
        max_distance = 4
        max_index = int(max_distance*f/vm)
        dt = int(l*f/vm)
        x = 1 - vm*range(max_index)/f # taylor hypothesis
        y = np.array(data[str(0)][dt:dt+max_index]) - np.array(data[str(0)][:max_index])
        axs[i//2,i%2].plot(x,y)
        axs[i//2,i%2].set_title('l = '+str(l))
        axs[i//2,i%2].set_xlabel('Distance (m)', fontsize=9)
        axs[i//2,i%2].set_ylabel('$\delta_u$ (m/s)', fontsize=9)
    
    fig.tight_layout(pad=1.5)
    
    
    # POINT 4
    plt.figure()
    for i,l in enumerate(dist):
        N = data[str(0)].shape[0]
        vm = np.mean(data[str(0)])
        dt = int(l*f/vm)
        y = np.array(data[str(0)][dt:]) - np.array(data[str(0)][:-dt])
        num = np.mean(np.power(y,4))
        den = np.mean(np.power(y,2))**2
        plt.plot(l,num/den, 'x', label = 'l = '+str(l))
    
    plt.plot([0,10],[3,3], label='Gaussian flatness')
    plt.legend()
    
    

def ex1_7(data):
    # POINT 1, 2, 3
    n_ref = 40
    dist = np.logspace(-3,0,n_ref) #for the four-fifth law, we need small l
    N = data[str(0)].shape[0]
    vm = np.mean(data[str(0)])
    
    S2 = np.zeros([n_ref])
    S3 = np.zeros([n_ref])
    
    print ("S2 and S3: ")
    for i in tqdm(range(n_ref)):
        dt = int(dist[i]*f/vm)
        y = np.array(data[str(0)][:-dt]) - np.array(data[str(0)][dt:])
        S2[i] = np.mean(np.power(y,2))
        S3[i] = np.mean(np.power(y,3))
    
    plt.figure()
    plt.loglog(dist, S2, label='S2')
    plt.loglog(dist, np.power(dist,2/3), label ='$x^{2/3}$')
    plt.title('S2')
    plt.xlabel('l (m)')
    plt.ylabel('S2 ($m^2/s^2$)')
    plt.legend()
    # A clear scaling is for l in [10^-2, 0.5] 
    
    plt.figure()
    plt.loglog(dist, -S3, label = '-S3')
    plt.loglog(dist, dist, label ='$y=x$')
    plt.title('S3')
    plt.xlabel('l (m)')
    plt.ylabel('S3 ($m^3/s^3$)')
    plt.legend()
    # A clear scaling is for l in [10^-2, 0.2] 
    
    
    # POINT 4
    l = 0.07 # I take a value in the middle of the above ranges
    N = data[str(0)].shape[0]
    vm = np.mean(data[str(0)])

    dt = int(l*f/vm)
    y = np.array(data[str(0)][:-dt]) - np.array(data[str(0)][dt:])
    S2 = np.mean(np.power(y,2))
    S3 = np.mean(np.power(y,3))
    
    e2 = S2**(3/2)/(2.2**(3/2)*l)
    e3 = -(5/4)*S3/l
    
    Lc = 0.36669985 # from ex1_2, point 1
    epsilon = 0.5*np.power(np.var(data[str(0)]), 3/2)/Lc
    print ("Original rate: ", epsilon)
    print ("Rate from S2: ", e2)
    print ("Rate from S3: ", e3)
    
    
    
def main():
    define_variables()
    data = get_data()
    #ex1_1(data)
    #ex1_2(data)
    #ex1_3(data)
    #ex1_4(data)
    #ex1_5(data)
    #ex1_6(data)
    ex1_7(data)
    
    del data
    gc.collect()
    
    
    
if __name__ == '__main__':
    main()
