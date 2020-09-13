# toy models of dual-Hebbian learning rule and the approximated rule
#
# full-model used in the simulations depicted in results is found in sample_code2.cpp
#
# Naoki Hiratani, The University of Tokyo
#
from math import *
import numpy as np
import random as rnd
import matplotlib.pyplot as plt

T = 100*1000 #simulation time

p = 10 #number of external states
M = 5 #number of pre-neurons 
gm = 0.4 #sparseness parameter
sigmaw = 1.0 #noise amplitude
rXo = 1.0 #pre-firing rate constant

trec = 10*1000 #rewiring time scale
etaX = 0.1 #learning rate of weight
etaR1 = 0.01 #learning rate of connection probability
etaR2 = 0.001

rhomin = 0.0
rhomax = 1.0
wmin = 0.0

rmeans = [] #mean responses
for q in range(p):
    rmeans.append([])
    for j in range(M):
        rmeans[q].append( rnd.gauss(1.0,1.0) )
        while rmeans[q][j] < 0.0:
            rmeans[q][j] = rnd.gauss(1.0,1.0)

rhomean = gm/(sigmaw*sigmaw)
wmean = 1.0/gm
wsig = 0.1
sigmay = 0.1

dt = 100

#dual-Hebbian
w1s = []
c1s = []
rho1s = []

#approx-dual-Hebbian
w2s = []
c2s = []
rho2s = []

#initial conditions
for j in range(M):
    w1s.append( wmean*(1.0 + rnd.gauss(0.0,wsig)) )
    rho1s.append( rhomean )
    if rnd.random() < rho1s[j]:
        c1s.append(1.0)
    else:
        c1s.append(0.0)

    w2s.append( wmean*(1.0 + rnd.gauss(0.0,wsig)) )
    rho2s.append( rhomean )
    if rnd.random() < rho2s[j]:
        c2s.append(1.0)
    else:
        c2s.append(0.0)

ts = []
w1ss = []
rho1ss = []
w2ss = []
rho2ss = []
for j in range(M):
    w1ss.append([])
    rho1ss.append([])
    w2ss.append([])
    rho2ss.append([])

st = 0
rXs = []
for j in range(M):
    rXs.append(0.0)
rY = 0.0

for t in range(T):
    st = rnd.choice(range(p))
    for j in range(M):
        rXs[j] = rmeans[st][j] + sigmaw*rnd.random()
    if st == 0:
        rY = 0.1*(1.0 + rnd.gauss(0.0,sigmay))
    else:
        rY = 0.01*(1.0 + rnd.gauss(0.0,sigmay))

    #dual-Hebbian rule
    for j in range(M):
        if c1s[j] > 0.5:
            w1s[j] += etaX*rY*(rXs[j] - sigmaw*sigmaw*rhomean*w1s[j])
            if w1s[j] < wmin:
                w1s[j] = wmin
        rho1s[j] += etaR1*rY*(rXs[j] - sigmaw*sigmaw*rho1s[j]*wmean)
        if rho1s[j] < rhomin:
            rho1s[j] = rhomin
        if rho1s[j] > rhomax:
            rho1s[j] = rhomax
        if c1s[j] < 0.5:
            if rnd.random() < rho1s[j]/trec:
                c1s[j] = 1.0
        else:
            if rnd.random() < (1.0 - rho1s[j])/trec:
                c1s[j] = 0.0
                w1s[j] = wmean*(1.0 + rnd.gauss(1.0,wsig))
                if w1s[j] < wmin:
                    w1s[j] = wmin
    
    #approx-dual Hebbian rule
    for j in range(M):
        if c2s[j] > 0.5:
            w2s[j] += etaX*rY*(rXs[j] - sigmaw*sigmaw*rhomean*w2s[j])
            if w2s[j] < wmin:
                w2s[j] = wmin
            rho2s[j] += etaR2*( w2s[j]*gm*gm - rho2s[j])
        if rho2s[j] < rhomin:
            rho2s[j] = rhomin
        if rho2s[j] > rhomax:
            rho2s[j] = rhomax
        if c2s[j] < 0.5:
            if rnd.random() < rho2s[j]/trec:
                c2s[j] = 1.0
        else:
            if rnd.random() < (1.0 - rho2s[j])/trec:
                c2s[j] = 0.0
                rho2s[j] = gm
                w2s[j] = wmean*(1.0 + rnd.gauss(1.0,wsig))
                if w2s[j] < wmin:
                    w2s[j] = wmin
    #recording
    if t%dt == 0:
        ts.append(t)
        for j in range(M):
            w1ss[j].append( c1s[j]*w1s[j] )
            rho1ss[j].append( rho1s[j] )
            w2ss[j].append( c2s[j]*w2s[j] )
            rho2ss[j].append( rho2s[j] )

#plotting
fig = plt.figure()
for j in range(M):
    ax1 = fig.add_subplot(2,5,j+1)
    ax1.plot(ts,w1ss[j],color='blue',linewidth=3.0)
    #ax1.set_yticks([0.0,3,6])
    ax1.set_ylim(0,1.1/(gm*gm))
    for tl in ax1.get_xticklabels():
        tl.set_fontsize(20)
    for tl in ax1.get_yticklabels():
        tl.set_color('blue')
        tl.set_fontsize(20)

    ax2 = ax1.twinx()
    ax2.plot(ts,rho1ss[j],color='green',linewidth=3.0)
    ax2.set_xticks([])
    ax2.set_yticks([0.0,0.5,1.0])
    #ax2.set_xlim(0,5000000)
    ax2.set_ylim(0,1.0)
    for tl in ax2.get_xticklabels():
        tl.set_fontsize(20)
    for tl in ax2.get_yticklabels():
        tl.set_color('green')
        tl.set_fontsize(20)

for j in range(M):
    ax1 = fig.add_subplot(2,5,5+j+1)
    ax1.plot(ts,w2ss[j],color='blue',linewidth=3.0)
    #ax1.set_yticks([0.0,3,6])
    ax1.set_ylim(0,1.1/(gm*gm))
    for tl in ax1.get_xticklabels():
        tl.set_fontsize(20)
    for tl in ax1.get_yticklabels():
        tl.set_color('blue')
        tl.set_fontsize(20)

    ax2 = ax1.twinx()
    ax2.plot(ts,rho2ss[j],color='green',linewidth=3.0)
    ax2.set_xticks([])
    ax2.set_yticks([0.0,0.5,1.0])
    #ax2.set_xlim(0,5000000)
    ax2.set_ylim(0,1.0)
    for tl in ax2.get_xticklabels():
        tl.set_fontsize(20)
    for tl in ax2.get_yticklabels():
        tl.set_color('green')
        tl.set_fontsize(20)

fig.subplots_adjust(left=0.1,right=0.85)
fig.suptitle('Synaptic weight',fontsize=20,color='blue',x=0.05,y=0.6,rotation=90)
fig.suptitle('Connection probability',fontsize=20,color='green',x=0.95,y=0.6,rotation=270)
fig.suptitle('Time',fontsize=20,color='black',x=0.5,y=0.08,rotation=0)

plt.show()
