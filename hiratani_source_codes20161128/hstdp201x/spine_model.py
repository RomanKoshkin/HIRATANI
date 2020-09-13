# Dendritic spine model
#
# Created by Naoki Hiratani (N.Hiratani@gmail.com)
#

from math import *
import numpy as np
import matplotlib.pyplot as plt
from pylab import cm

dt = 0.1 #ms

N = 2
clrs = ['0.7','0.1','0.4']
#for q in range(5):
#    clrs.append( cm.jet( q/float(5+1) ) )


tauc = 18.0#30.0#30.0#30.0
taum = 3.0#5.0#5.0#5.0#15.0
tauN = 15.0#25.0#25.0#25.0#25.0#50.0#25.0
tauA = 3.0#5.0#5.0#10.0#5.0
tauBP = 3.0#5.0#5.0#7.0#5.0
tauG = 3.0#5.0
tauE = 6.0#10.0

adelay = 7.5#15.0
#Idelay = 5.0
postdelay = 10.0
Idelay = 0.0
Edelay = 1.0#10.0

alphaN = 1.0 #fixed
betaN = 0.0#1.0#1.0
alphaV = 2.0
gammaA = 1.0 #fixed
gammaN = 0.20#0.15#0.10
gammaBP = 8.5#8.0#8.0
gammaG = 3.0#0.5#2.0
gammaE = 0.0#0.4#2.5 #0.0

thetap = 70#200
thetad = 35#100
thetaN = 0.0
Cp = 2.3#1.9#1.75#1.75
Cd = 1.0

yth = 5.0

def h(ytmp):
    if ytmp > yth:
        return ytmp - yth
    elif ytmp < -yth:
        return ytmp + yth
    else:
        return 0.0

def rk(xtmp,tautmp):
    kx1 = -xtmp/tautmp
    x1 = xtmp + kx1*0.5*dt

    kx2 = -x1/tautmp
    x2 = xtmp + kx2*0.5*dt

    kx3 = -x2/tautmp
    x3 = xtmp + kx3*dt

    kx4 = -x3/tautmp
    x4 = xtmp + dt*(kx1 + 2.0*kx2 + 2.0*kx3 + kx4)/6.0

    return x4

def gN(utmp):
    return alphaN*utmp + betaN

def gV(utmp):
    return alphaV*utmp

def rk_cu(ctmp,utmp,xA,xN,xBP,xG):
    kc1 = -ctmp/tauc + gN(utmp)*xN + gV(utmp)
    ku1 = -utmp/taum + gammaA*xA + gammaN*gN(utmp)*xN + gammaBP*xBP - gammaG*xG
    c1 = ctmp + kc1*0.5*dt
    u1 = utmp + ku1*0.5*dt

    kc2 = -c1/tauc + gN(u1)*xN + gV(u1)
    ku2 = -u1/taum + gammaA*xA + gammaN*gN(u1)*xN + gammaBP*xBP - gammaG*xG
    c2 = ctmp + kc2*0.5*dt
    u2 = utmp + ku2*0.5*dt

    kc3 = -c2/tauc + gN(u2)*xN + gV(u2)
    ku3 = -u2/taum + gammaA*xA + gammaN*gN(u2)*xN + gammaBP*xBP - gammaG*xG
    c3 = ctmp + kc3*dt
    u3 = utmp + ku3*dt

    kc4 = -c3/tauc + gN(u3)*xN + gV(u3)
    ku4 = -u3/taum + gammaA*xA + gammaN*gN(u3)*xN + gammaBP*xBP - gammaG*xG
    c4 = ctmp + dt*(kc1 + 2.0*kc2 + 2.0*kc3 + kc4)/6.0
    u4 = utmp + dt*(ku1 + 2.0*ku2 + 2.0*ku3 + ku4)/6.0

    return c4,u4

Ts = np.arange(0,800.0,dt)

sptposts = [200,410]
sptpres = [100,310,400]
sptIs = [300]

c = 0.0
u = 0.0
y = 0.0
xA = 0.0
xN = 0.0
xBP = 0.0
xG = 0.0
xE = 0.0

Us = []
Cs = []
ys = []
for t in Ts:
    for sptpre in sptpres:
        if abs(t - sptpre) < 0.5*dt:
            xA += 1.0
            xN += 1.0
    xA = rk(xA,tauA)
    xN = rk(xN,tauN)

    for sptpost in sptposts:
        if abs(t - sptpost) < 0.5*dt:
            xBP += 1.0
    xBP = rk(xBP,tauBP)

    for sptI in sptIs:
        if abs(t - sptI) < 0.5*dt:
            xG += 1.0
    xG = rk(xG,tauG)

    c,u = rk_cu(c,u,xA,xN,xBP,xG)
    if c > thetap:# and xN > thetaN:
        y += Cp*dt
    if c > thetad:
        y -= Cd*dt
    Us.append(u)
    Cs.append(c)
    ys.append(y)

plt.subplot(3,1,1)
plt.axhline(0.0,color='gray')
for sptpre in sptpres:
    plt.axvline(sptpre,color='green',linewidth=3.0)
for sptpost in sptposts:
    plt.axvline(sptpost,color='purple',linewidth=3.0)
for sptI in sptIs:
    plt.axvline(sptI,color='red',linewidth=3.0)
plt.plot(Ts,Us,color='k',linewidth=5.0)
plt.xticks([])
plt.yticks([0,4,8,12],fontsize=30)
plt.ylim(-4,12)
plt.xlim(0,550)

plt.subplot(3,1,2)
plt.axhline(thetad,color='cyan',linewidth=5.0,ls='--')
plt.axhline(thetap,color='orange',linewidth=5.0,ls='--')
plt.axhline(0.0,color='gray')
for sptpre in sptpres:
    plt.axvline(sptpre,color='green',linewidth=3.0)
for sptpost in sptposts:
    plt.axvline(sptpost,color='purple',linewidth=3.0)
for sptI in sptIs:
    plt.axvline(sptI,color='red',linewidth=3.0)
plt.plot(Ts,Cs,color='k',linewidth=5.0)
plt.xticks([])
plt.yticks([0,100,200],fontsize=30)
plt.ylim(-50,200)
plt.xlim(0,550)

plt.subplot(3,1,3)
plt.axhline(0.0,color='gray')
for sptpre in sptpres:
    plt.axvline(sptpre,color='green',linewidth=3.0)
for sptpost in sptposts:
    plt.axvline(sptpost,color='purple',linewidth=3.0)
for sptI in sptIs:
    plt.axvline(sptI,color='red',linewidth=3.0)
plt.plot(Ts,ys,color='k',linewidth=5.0)
plt.xticks([0,100,200,300,400,500],fontsize=30)
plt.yticks([0,20,40],fontsize=30)
plt.ylim(-5,40)
plt.xlim(0,550)
plt.show()
