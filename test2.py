import time, sys
import numpy as np
from cClasses import cClassOne
import matplotlib.pyplot as plt


params = {
    "alpha": 50.0,    # Degree of log-STDP (50.0)
    "usd": 0.1,       # Release probability of a synapse (0.05 - 0.5)
    "JEI": 0.15,      # 0.15 or 0.20
    "ita": 2,           # stim A duration, s (90)
    "itb": 2,           # stim B duration, s (30)

    "T": 1800*1000.0,   # simulation time, ms
    "h": 0.01,          # time step, ms ??????

    # probability of connection
    "cEE": 0.2, # 
    "cIE": 0.2, #
    "cEI": 0.5, #
    "cII": 0.5, #

    # Synaptic weights
    "JEE": 0.15, #
    "JEEinit": 0.18, # ?????????????
    "JIE": 0.15, # 
    "JII": 0.06, #
    
    #initial conditions of synaptic weights
    "JEEh": 0.15, # Standard synaptic weight E-E
    "sigJ": 0.3,  #

    "Jtmax": 0.25, # J_maxˆtot
    "Jtmin": 0.01, # J_minˆtot # ??? NOT IN THE PAPER

    # Thresholds of update
    "hE": 1.0, # Threshold of update of excitatory neurons
    "hI": 1.0, # Threshold of update of inhibotory neurons

    "IEex": 2.0, # Amplitude of steady external input to excitatory neurons
    "IIex": 0.5, # Amplitude of steady external input to inhibitory neurons
    "mex": 0.3,        # mean of external input
    "sigex": 0.1,      # variance of external input

    # Average intervals of update, ms
    "tmE": 5.0,  #t_Eud EXCITATORY
    "tmI": 2.5,  #t_Iud INHIBITORY
    
    #Short-Term Depression
    "trec": 600.0,     # recovery time constant (tau_sd, p.13 and p.12)
    "Jepsilon": 0.001, # ????????
    
    # Time constants of STDP decay
    "tpp": 20.0,  # tau_p
    "tpd": 40.0,  # tau_d
    "twnd": 500.0, # STDP window lenght, ms
    
    "g": 1.25,        # ??????
    
    #homeostatic
    "itauh": 100,		# decay time of homeostatic plasticity, (100s)
    "hsd": 0.1,
    "hh": 10.0,  # SOME MYSTERIOUS PARAMETER
    "Ip": 1.0, # External current applied to randomly chosen excitatory neurons
    "a": 0.20, # Fraction of neurons to which this external current is applied
    
    "xEinit": 0.02, # the probability that an excitatory neurons spikes at the beginning of the simulation
    "xIinit": 0.01, # the probability that an inhibitory neurons spikes at the beginning of the simulation
    "tinit": 1.00, # period of time after which STDP kicks in (100.0)
    "tdur": 1000.0,
    "t1": 1.0   # stim 1 onset (s)
} 

NE = 1800
NI = 360

params['t1'] = 0.5
params['ita'] = 1
params['itb'] = 1

# params['SNE'] = 100.0
# params['SNI'] = 40.0

# params['cEI'] = 0.00
# params['cIE'] = 0.00
# params['cII'] = 0.8
# params['cEE'] = 0.8

# params['IEex'] = 40.00
# params['IIex'] = 0.0
# params['Ip'] = 5.0

m = cClassOne(NE, NI)
m.setParams(params)
ret = m.getState()

# check if the parameters have been set:
for var_name in params.keys():
    if params[var_name] != getattr(ret, var_name):
        print("{} doesn't match".format(var_name))
m.setParams(params)

ret = m.getState()
[(attr, getattr(ret, attr)) for attr in dir(ret) if attr[0]!="_"]

sim_t = 5
snapshotEveryMs = 1000

W = []
W.append(m.getWeights())
for i in range(int(sim_t*1000/snapshotEveryMs)):
    t = time.time()
    # simulate for the number of steps that are in 1000 ms (1 s)
    m.sim(int(snapshotEveryMs / params['h']))
    
    # get the states
    ret = m.getState()
    print('Elapsed {:.3f} s. per 1 s. of simulated time | t = {:.2f}'.format(time.time() - t, ret.t))
    t = time.time()
    
    # take a snapshot of the weight matrix
    W.append(m.getWeights())
print("Memory footprint of W is {:.2f} MB".format(
    np.sum([W[i].size * W[i].itemsize / 1024**2 for i in range(len(W))])))