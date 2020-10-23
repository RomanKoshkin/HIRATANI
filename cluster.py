import time, sys, os
import numpy as np
from cClasses import cClassOne
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import shutil
import pickle
from pathlib import Path
home = str(Path.home())

HAGA = bool(int(sys.argv[1]))
asym = bool(int(sys.argv[2]))
U = float(sys.argv[3])
sim_t = float(sys.argv[4])

print(f'\n STARTING  HAGA: {HAGA}\nasym: {asym}\nU: {U}\n sim_t: {sim_t}\n')

params = {
    "alpha": 50.0,    # Degree of log-STDP (50.0)
    "usd": 0.1,       # Release probability of a synapse (0.05 - 0.5)
    "JEI": 0.15,      # 0.15 or 0.20

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
    "itauh": 100,       # decay time of homeostatic plasticity, (100s)
    "hsd": 0.1,
    "hh": 10.0,  # SOME MYSTERIOUS PARAMETER
    "Ip": 1.0, # External current applied to randomly chosen excitatory neurons
    "a": 0.20, # Fraction of neurons to which this external current is applied
    
    "xEinit": 0.02, # the probability that an excitatory neurons spikes at the beginning of the simulation
    "xIinit": 0.01, # the probability that an inhibitory neurons spikes at the beginning of the simulation
    "tinit": 1.00, # period of time after which STDP kicks in (100.0)
    "U": 0.6,
    "taustf": 200,
    "taustd": 500,
    "HAGA": True,
    "asym": True,
    "nstim": 1,
    "sm": -1
} 

dirname = f'/scratch/HAGA_{HAGA}_asym_{asym}_U_{U}'

if not os.path.exists(dirname):
    os.makedirs(dirname)
else:
    shutil.rmtree(dirname)           # Removes all the subdirectories!
    os.makedirs(dirname)

os.chdir(dirname)
wd = os.getcwd()
imdir = wd + "/img"
os.mkdir(imdir)

NE = 2500
NI = 500

params['asym'] = asym
params['HAGA'] = HAGA

# stimulus matrix (each row is a definition of stimulus. onset/offset, ms ; neur_id/neur_id range)
params['sm'] = np.array([1000, 3000, 000, 500, 
                         3000, 6000, 500, 1000,
                         6000, 8000, 2000, 2400], 
                         dtype='float')
params['nstim'] = len(params['sm'])//4

if params['HAGA'] == True:
    params['g'] = 2.5
    params['U'] = U
else:
    params['usd'] = U


m = cClassOne(NE, NI)
m.setParams(params)
ret = m.getState()

# check if the parameters have been set:
for var_name in params.keys():
    try:
        if params[var_name] != getattr(ret, var_name):
            print("{} doesn't match".format(var_name))
    except Exception as e: 
        print(e)
        
m.setParams(params)

ret = m.getState()
[(attr, getattr(ret, attr)) for attr in dir(ret) if attr[0]!="_"]

snapshotEveryMs = 1000

A, B, C, D, A0, A1, CA2CA, BG2CA, CA2BG, BG2BG = [],[],[],[],[],[],[],[],[],[]

    
def wtSnapshot(m, i):
    global A, B, C, D, A0, A1, dirname, CA2CA, BG2CA, CA2BG, BG2BG
#     fig = plt.figure()
    W = m.getWeights()
#     ax = plt.Axes(fig, [0., 0., 1., 1.])
#     ax.set_axis_off()
#     fig.add_axes(ax)
#     plt.imshow(W, origin='upper', vmin=-0.1, vmax=0.2)
    print(f'Saving to {imdir}')
#     fig.savefig(f'{imdir}/fig{i}.png', dpi=300)
#     plt.close(fig)

    a = W[:NE, :NE].mean() # EE
    b = W[:NE, NE:].mean() # IE
    c = W[NE:, NE:].mean() # II
    d = W[NE:, :NE].mean() # EI
    
    ca2ca = W[:500,:500].mean()
    bg2ca = W[:500,500:2500].mean()
    ca2bg = W[500:2500, :500].mean()
    bg2bg = W[500:2500, 500:2500].mean()
    as0 = W[:500, :500].mean() # assebly 0
    as1 = W[500:1000, 500:1000].mean() # assebly 1
    A.append(a)
    B.append(b)
    C.append(c)
    D.append(d)
    A0.append(as0)
    A1.append(as1)
    CA2CA.append(ca2ca)
    BG2CA.append(bg2ca)
    CA2BG.append(ca2bg)
    BG2BG.append(bg2bg)

interstitial_o = 1000.0
with open(f'{home}/H2/SETTLED_WTS/HAGA_{HAGA}_asym_{asym}_U_{U}_interstitial_{interstitial_o}/settled_wts.pickle', 'rb') as f:
    WS = pickle.load(f)
    m.setWeights(WS)
    
with open(f'{home}/H2/SETTLED_WTS/HAGA_{HAGA}_asym_{asym}_U_{U}_interstitial_{interstitial_o}/settled_F.pickle', 'rb') as f:
    FS = pickle.load(f)
    m.setF(FS)
    
with open(f'{home}/H2/SETTLED_WTS/HAGA_{HAGA}_asym_{asym}_U_{U}_interstitial_{interstitial_o}/settled_D.pickle', 'rb') as f:
    DS = pickle.load(f)
    m.setD(DS)
    
with open(f'{home}/H2/SETTLED_WTS/HAGA_{HAGA}_asym_{asym}_U_{U}_interstitial_{interstitial_o}/settled_ys.pickle', 'rb') as f:
    ys = pickle.load(f)
    m.setys(ys)


wtSnapshot(m, 0)
for i in range(int(sim_t*1000/snapshotEveryMs)):
    t = time.time()
    
    # simulate for the number of steps that are in 1000 ms (1 s)
    m.sim(int(snapshotEveryMs / params['h']))
   
    # get the states
    ret = m.getState()
    print('Elapsed {:.3f} s. per 1 s. of simulated time | t = {:.2f}'.format(time.time() - t, ret.t))
    t = time.time()
    
    wtSnapshot(m, i)

    tt = np.round(ret.t/1000, 0).astype('int')
    with open(f'weight_dynamics_T_{tt}.pickle', 'wb') as f:
        pickle.dump({"EE": A,
                     "IE": B,
                     "II": C,
                     "EI": D,
                     "A0": A0,
                     "A1": A1,
                     "CA2CA": CA2CA,
                     "BG2CA": BG2CA,
                     "CA2BG": CA2BG,
                     "BG2BG": BG2BG},
                    f)

    prev = f'weight_dynamics_T_{(tt - 1)}.pickle'
    if prev in os.listdir():
        os.remove(prev)

print(len(A), len(B), len(C), len(D))

# dump settled weights
W = m.getWeights()
with open(f'settled_wts.pickle', 'wb') as f:
    pickle.dump(W, f)
    
F = m.getF()
with open(f'settled_F.pickle', 'wb') as f:
    pickle.dump(F, f)

D = m.getD()
with open(f'settled_D.pickle', 'wb') as f:
    pickle.dump(D, f)

ys = m.getys()
with open(f'settled_ys.pickle', 'wb') as f:
    pickle.dump(ys, f)
    



    

