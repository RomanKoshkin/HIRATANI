This is the readme file for simulation codes associated with:

Hiratani N, Fukai T (2016) Hebbian wiring plasticity generates efficient network structure for robust inference with synaptic weight plasticity. Front Neural Circuits 10:41. doi: 10.3389/fncir.2016.00041

The codes were contributed by Naoki Hiratani.
Questions should be addressed to N.Hiratani"at"gmail.com (replace "at" to @).

-sample_code1.py
This python file is the simplified version of a simulation code for explanation of the implementation of the dual-Hebbian and the approximated dual-Hebbian rules.
The program depicts temporal dynamics of synaptic weights(blue lines) and connection probabilities(green lines), for given activity of presynaptic neurons and the postsynaptic neuron. Upper graphs are the behavior under the dual-Hebbian rule, and lower graphs are for the approximated dual-Hebbian rule. Note that in the actual simulation, the output neuron changes its response through learning, while the response is fixed in here.


-sample_code2.cpp
This c++ file is a simulation code for learning through the dual-Hebbian mechanism under a dynamic environment, which is depicted in Figure 6 in the paper.
The program receives eight variables from the standard input as below.
Ly : redundancy in the output population 
gm : sparseness parameter
sigmaw : noise level
bh : strength of homeostatic plasticity
etar : learning rate of connection probability
treco : rewiring time scale (trec = treco*1000.0)
kappa : degree of the change in input structure
ik : simulation ID
 
Output files of the programs are following text files
sample_code2p*.txt : temporal change of task performance
sample_code2q*.txt : input selectivity in the simulation
sample_code2s*.txt : trajectory of the external state
sample_code2rx*.txt : firing rates of input neurons
sample_code2ry*.txt : firing rates of output neurons

Note that the simulation typically takes several hours in a starndard computer at Ly=10.
