import time
import numpy as np
import matplotlib.pyplot as plt
from utils import printProgressBar

def run(T = 10, tpost = 5, tpre = 10):
    Nexc = 2
    H = 0.01
    Cp = 0.01875
    Cd = 0.0075
    tauP = 20 #20/H # ms/dt
    tauD = 40 # 40/H # ms/dt
    alpha = 50.0
    JEE = 0.15


    # diff_step = 0
    # DIFF_STEPS = np.linspace(-10, 10, 30)
    times = np.round(np.arange(0,T, H), 2)
    W = np.zeros((len(times), ))

    x_E = np.zeros((Nexc, ))
    lastSpikeTime = np.ones((Nexc, )) * -5e7

    J_EE = np.array([[0.0, 0.0],[0.5, 0.0]])
    c_EE = np.array([[0.0, 0.0],[0.5, 0.0]])

    XX = np.zeros((Nexc, int(T/H)))
    XX[0,int(tpre/H)] = 1
    XX[1,int(tpost/H)] = 1

    DW = []
    step = 0
    for t in times:

        x_E = XX[:, step]

        if x_E[1] == 1:
            lastSpikeTime[1] = t
        Y = Cd * np.exp(-(t-lastSpikeTime[1])/tauD)  # fix tau and Fd !!!! (Log-STDP!!!!)


        if x_E[0] == 1:
            lastSpikeTime[0] = t
        X = Cp * np.exp(-(t-lastSpikeTime[0])/tauP) # fix tau!!!!

#         fd = np.log(1 + alpha * J_EE[1, 0]/JEE)/np.log(1 + alpha)

        dw = X * x_E[1] - Y * x_E[0]
        J_EE[1, 0] += dw * c_EE[1, 0]
        W[step] = J_EE[1, 0]
        step += 1
        DW.append(dw)
    return times, XX, W, DW[np.argmax(np.abs(DW))]


T = 100
initial_weight = 0.5
tdiff = []
dw = []
tpre = 50
for tpost in np.arange(1,90, 0.1):
    times, XX, W, DW = run(T=T, tpost=tpost, tpre=tpre)
    tdiff.append(tpost-tpre)
    print(tpost-tpre)
    dw.append(W[-1]-initial_weight)
plt.plot(tdiff, dw)
