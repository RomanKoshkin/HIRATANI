from cClasses import cClassOne

params = {
    "alpha": 50.0,    # Degree of log-STDP (50.0)
    "usd": 0.1,       # Release probability of a synapse (0.05 - 0.5)
    "JEI": 0.15,      # 0.15 or 0.20
    "ita": 90,           # stim A duration, s
    "itb": 30,           # stim B duration, s

    "T": 1800*1000.0,   # simulation time, ms
    "h": 0.01,          # time step, ms ??????

    "NE": 2500,    # Nexc
    "NI": 500,     # Ninh

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
    "hI": 1.0,       # Threshold of update of inhibotory neurons

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
    "tinit": 100.0, # period of time after which STDP kicks in
    "tdur": 1000.0
} 

m = cClassOne(params["NE"] + params["NI"])
# m = cClassOne(3)
m.setParams(params)
ret = m.getState()
ret_params = {}
for var_name in dir(ret):
    ret_params[var_name] = getattr(ret, var_name)
# print(ret_params)

wts = m.getWeights()

pass

# m.sim(300000)