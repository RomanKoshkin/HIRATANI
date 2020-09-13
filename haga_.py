import numpy as np
import matplotlib.pyplot as plt

import numba

from numba import jit
print('Numba ver. {}'.format(numba.__version__))


@jit(nopython=True, fastmath=True) #, parallel=True)
def batch_step():
    
    # params:
    epsilon = 0.5
    rho = 0.0025

    eta = 20
    tauw = 1000

    tauexc = 10
    tauinh = 10
    winh = 1
    wmax = 27
    d = 5
    U = 0.6

    N = 500

    tauSTD = 500
    tauSTF = 200
    dt = 0.01
    T = 4000


    # __init__
    W = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            W[i,j] = wmax * np.exp(-np.abs(i-j)/d)
        W[i,i] = 0
    
    W_ = np.copy(W)
    
    r = np.zeros((N,))
    D = np.zeros((N,))
    F = np.zeros((N,))
    
    D = np.ones((N,)) * 1
    F = np.ones((N,)) * U
    
    Iexc = np.zeros((N,))
    Iext = np.zeros((N,))
    Iinh = np.zeros((N,))
    delta = np.zeros_like(W)
    
    out = np.zeros((len(np.arange(0, T, dt)), 6, N))
    
    iteration = 0
    
    for t in np.arange(0, T, dt):
        if t >= 0 and t < 10:
            Iext[:10] = 5    
        elif t >= 3000 and t < 3010:
            Iext[245:250] = 5
        else:
            Iext *= 0
            
        I = Iexc - Iinh + Iext
        for i in range(N):
            aa = rho*(I[i]-epsilon)
            r[i] = aa if aa > 0 else 0

        for i in range(N):
            Sum = 0
            for j in range(N):
                Sum += W[i,j] * r[j] * D[j] * F[j]
            Iexc[i] += (-Iexc[i]/tauexc + Sum)*dt

        Sum = 0
        for j in range(N):
            Sum += r[j] * D[j] * F[j]
            Iinh[j] += (-Iinh[j]/tauinh + winh*Sum)*dt

        for j in range(len(D)):
            F[j] += ((U - F[j])/tauSTF + U*(1 - F[j])*r[j])*dt

        for j in range(N):
            D[j] += ((1 - D[j])/tauSTD - r[j] * D[j] * F[j])*dt

        for i in range(N):
            for j in range(N):
                delta[i,j] += (-delta[i,j] + eta*r[i]*r[j]*D[j]*F[j])/tauw * dt
            delta[i,i] = 0

        W += delta# * d
        
        
        if iteration%10000==0:
            print('Sim time:\t', np.round(t))
            
        out[iteration,0,:] = r
        out[iteration,1,:] = D
        out[iteration,2,:] = F
        out[iteration,3,:] = Iexc
        out[iteration,4,:] = Iinh
        out[iteration,5,:] = Iext
        
        iteration += 1


    return N, W_, W, out

N, W_, W, out = batch_step()