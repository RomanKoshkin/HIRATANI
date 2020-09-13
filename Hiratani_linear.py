
import time
import numpy as np
import matplotlib.pyplot as plt
from utils import printProgressBar

import numba
from numba import jit
from numba import jitclass          # import the decorator
from numba import int32, float64    # import the types

Time = time.time()

@jit(nopython=True, fastmath=True)
def run(Nexc, Ninh, T, H, stimStart, stimStop, c_EE, c_IE, c_EI, c_II):
    
    t_Eud = 5.0        # average time that an excitatory neuron's state is updated 
    t_Iud = 2.5        # average time that an inhibitory neuron's state is updated

    times = np.round(np.arange(0, T, H), 2)
    print(times)
    step = 0

    out_E = np.zeros((Nexc, len(times)))
    out_I = np.zeros((Ninh, len(times)))
    y_t = np.zeros((Nexc, len(times)))
    
    h_E = 1.0
    h_I = 1.0
    m_ex = 0.3
    I_Eex = 2.0
    I_Iex = 0.5
    I_p = np.zeros((Nexc, ))
    m_ex = 0.3
    sigma_ex = 0.1
    t_Eud = 5.0
    t_Iud = 2.5
    Cp = 0.01875
    Cd = 0.0075

    tauP = 20/H # ms/dt
    tauD = 40/H # ms/dt
    
    tau_sd = 600
    u_sd = 0.2

    y = np.ones((Nexc, )) * 0.5
    Y = 0
    X = 0
    st_neur = int(Nexc*0.8)
    
    J_EE = 0.3 * np.random.randn(Nexc,Nexc) + 0.18
    for post in range(Nexc):
        for pre in range(Nexc):
            if J_EE[post, pre] < 0.0:
                J_EE[post, pre] = 0.0
            if J_EE[post, pre] > 0.75:
                J_EE[post, pre] = 0.75

    J_EI = np.ones((Nexc, Ninh)) * 0.15
    J_II = np.ones((Ninh, Ninh)) * 0.06
    J_IE = np.ones((Ninh, Nexc)) * 0.15

    x_E = np.zeros((Nexc, ))
    x_I = np.zeros((Ninh, ))
    lastSpikeTime = np.ones((Nexc, )) * -5e7
    alpha = 50.0
    JEE = 0.15


    for t in times:
        Iex_ud = np.arange(0, Nexc, 1)
        Iinh_ud = np.arange(0, Ninh, 1)
        if t == stimStart:
            print('yeah')
            I_p[st_neur:] = 1
        if t == stimStop:
            print('oh')
            I_p = np.zeros((Nexc, ))

        for i in Iex_ud: #range(Nexc):
            xi_E = np.random.randn()
            acc = 0.0
            for j in range(Nexc):
                if j!=i:
                    acc += c_EE[i,j] * J_EE[i,j] * y[j] * x_E[j]
            for j in range(Ninh):
                acc -= c_EI[i,j] * J_EI[i,j] * x_I[j]  
            acc += I_Eex * (m_ex + sigma_ex * xi_E) + I_p[i] - h_E
            x_E[i] = 1 if acc > 0 else 0
        
        """updated STP"""
        for j in Iex_ud:
            y[j] += ((1-y[j])/tau_sd - u_sd*y[j] * x_E[j])*H
        
        for i in Iinh_ud: #range(Ninh):
            xi_I = np.random.randn()
            acc = 0.0
            for j in range(Nexc):
                acc += c_IE[i,j] * J_IE[i,j] * x_E[j]
            for j in Iinh_ud: #range(Ninh):
                if j!=i:
                    acc -= c_II[i,j] * J_II[i,j] * x_I[j]
            acc += I_Iex * (m_ex + sigma_ex * xi_I) - h_I
            x_I[i] = 1 if acc > 0 else 0
        
        
        """STDP"""
        for post in Iex_ud:
            if x_E[post] == 1:
                lastSpikeTime[post] = t
            Y = Cd * np.exp(-(t-lastSpikeTime[post])/tauD)  # fix tau and Fd !!!! (Log-STDP!!!!)

            for pre in Iex_ud:
                if x_E[pre] == 1:
                    lastSpikeTime[pre] = t
                X = Cp * np.exp(-(t-lastSpikeTime[pre])/tauP) # fix tau!!!!


                fd = np.log(1 + alpha * J_EE[post, pre]/JEE)/np.log(1 + alpha)

                dw = X * x_E[post] - fd * Y * x_E[pre] # f_d(post, pre)
                J_EE[post, pre] += dw * c_EE[post, pre]

                # clip the the weights:
                if J_EE[post, pre] < 0.0:
                    J_EE[post, pre] = 0.0
                if J_EE[post, pre] > 0.75:
                    J_EE[post, pre] = 0.75

        out_E[:,step] = x_E
        out_I[:,step] = x_I
        y_t[:,step] = y
        step += 1
    return out_E, out_I, J_EE, y_t

        
        
        
def initw(size, prob):
    arr = np.zeros((size[0]*size[1], ))
    arr[:int(prob*size[0]*size[1])] = 1
    np.random.shuffle(arr)
    arr = arr.reshape(size[0], size[1])
    return arr


Nexc = 250         # number of excitatory neurons in the network
Ninh = 50          # number of inhibitory neurons in the network
T = 2               # simulation time, ms
H = 0.01            # dt, ms
stimStart = 1.0
stimStop = 2.0

c_EE = initw(size=(Nexc,Nexc), prob=0.2)
c_IE = initw(size=(Ninh,Nexc), prob=0.2) 
c_EI = initw(size=(Nexc,Ninh), prob=0.5) 
c_II = initw(size=(Ninh,Ninh), prob=0.5) 

out_E, out_I, J_EE, y_t = run(Nexc, Ninh, T, H, stimStart, stimStop, c_EE, c_IE, c_EI, c_II)
print(time.time()-Time)