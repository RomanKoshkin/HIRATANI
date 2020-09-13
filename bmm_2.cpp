// Reduced binary model of short- and long-term synaptic plasticity
//
// Original code by Created by Naoki Hiratani (N.Hiratani@gmail.com)
// Comments and Python-friendly interactive implementation 
// by Roman Koshkin (roman.koshkin@gmail.com)

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <deque>
#include <set>
#include <algorithm>
#include <cmath>
#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <chrono>
#include <algorithm> 

using namespace std;

// to complile
// g++ -std=gnu++11 -O3 -dynamiclib -ftree-vectorize -march=native -mavx bmm_2.cpp -o ./bmm.dylib


struct Timer
{
	std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
	std::chrono::duration<float> duration;

	Timer() {
		start = std::chrono::high_resolution_clock::now();
	}

	/* when the function where this object is created returns,
	this object must be destroyed, hence this destructor is called */
	~Timer() {
		end = std::chrono::high_resolution_clock::now();
		duration = end - start;
		float ms = duration.count() * 1000.0f;
		std::cout << "Elapsed: " << ms << " ms." << std::endl;
	}
};



// class definition
class Model {
    public:
        // input struct
        typedef struct {
            double alpha;
            double usd; 
            double JEI ;
            int ita;
            int itb;

            double T;
            double h;

            // int NE;
            // int NI;

            // probability of connection
            double cEE;
            double cIE;
            double cEI;
            double cII;

            // Synaptic weights
            double JEE;
            double JEEinit;
            double JIE;
            double JII;

            //initial conditions of synaptic weights
            double JEEh;
            double sigJ;

            double Jtmax;
            double Jtmin;

            // Thresholds of update
            double hE;
            double hI;

            double IEex;
            double IIex;
            double mex;
            double sigex;

            // Average intervals of update, ms
            double tmE;
            double tmI;

            //Short-Term Depression
            double trec;
            double Jepsilon;

            // Time constants of STDP decay
            double tpp;
            double tpd;
            double twnd;

            // Coefficients of STDP
            double g;

            //homeostatic
            int itauh;

            double hsd;
            double hh;

            double Ip;
            double a;

            double xEinit;
            double xIinit;
            double tinit;
            double tdur;   
            double t1;        
        } ParamsStructType;

        // returned struct
        typedef struct {
            double alpha;
            double usd; 
            double JEI ;
            int ita;
            int itb;

            double T;
            double h;

            int NE;
            int NI;

            // probability of connection
            double cEE;
            double cIE;
            double cEI;
            double cII;

            // Synaptic weights
            double JEE;
            double JEEinit;
            double JIE;
            double JII;

            //initial conditions of synaptic weights
            double JEEh;
            double sigJ;

            double Jtmax;
            double Jtmin;

            // Thresholds of update
            double hE;
            double hI;

            double IEex;
            double IIex;
            double mex;
            double sigex;

            // Average intervals of update, ms
            double tmE;
            double tmI;

            //Short-Term Depression
            double trec;
            double Jepsilon;

            // Time constants of STDP decay
            double tpp;
            double tpd;
            double twnd;

            // Coefficients of STDP
            double g;

            //homeostatic
            int itauh;

            double hsd;
            double hh;

            double Ip;
            double a;

            double xEinit;
            double xIinit;
            double tinit;
            double tdur;

            double Jmin;
            double Jmax;
            double Cp;
            double Cd;
            int SNE;
            int SNI;
            int NEa;
            double t;
            double t1;
            
        } retParamsStructType;

		int NE, NI, N;
		int SNE, SNI; //how many neurons get updated per time step
		int NEa; // Exact number of excitatory neurons stimulated externally
		int pmax;

		vector< vector<double>> Jo;
		vector<vector<double>> Ji;

        double alpha = 50.0;    // Degree of log-STDP (50.0)
        double usd = 0.1;       // Release probability of a synapse (0.05 - 0.5)
        double JEI = 0.15;      // 0.15 or 0.20
        int ita = 90;           // stim A duration, s
        int itb = 30;           // stim B duration, s

        double pi = 3.14159265;
        double e = 2.71828182;

        double T = 1800*1000.0;   // simulation time, ms
        double h = 0.01;          // time step, ms ??????

        // probability of connection
        double cEE = 0.2; // 
        double cIE = 0.2; //
        double cEI = 0.5; //
        double cII = 0.5; //

        // Synaptic weights
        double JEE = 0.15; //
        double JEEinit = 0.18; // ?????????????
        double JIE = 0.15; // 
        double JII = 0.06; //
        //initial conditions of synaptic weights
        double JEEh = 0.15; // Standard synaptic weight E-E
        double sigJ = 0.3;  //

        double Jtmax = 0.25; // J_maxˆtot
        double Jtmin = 0.01; // J_minˆtot // ??? NOT IN THE PAPER

        // WEIGHT CLIPPING     // ???
        double Jmax = 5.0*JEE; // ???
        double Jmin = 0.01*JEE; // ????

        // Thresholds of update
        double hE = 1.0; // Threshold of update of excitatory neurons
        double hI = 1.0;       // Threshold of update of inhibotory neurons

        double IEex = 2.0; // Amplitude of steady external input to excitatory neurons
        double IIex = 0.5; // Amplitude of steady external input to inhibitory neurons
        double mex = 0.3;        // mean of external input
        double sigex = 0.1;      // variance of external input

        // Average intervals of update, ms
        double tmE = 5.0;  //t_Eud EXCITATORY
        double tmI = 2.5;  //t_Iud INHIBITORY

        //Short-Term Depression
        double trec = 600.0;     // recovery time constant (tau_sd, p.13 and p.12)
        //double usyn = 0.1;
        double Jepsilon = 0.001; 	// BEFORE UPDATING A WEIGHT, WE CHECK IF IT IS GREATER THAN
									// Jepsilon. If smaller, we consider this connection as
									// non-existent, and do not update the weight.

        // Time constants of STDP decay
        double tpp = 20.0;  // tau_p
        double tpd = 40.0;  // tau_d
        double twnd = 500.0; // STDP window lenght, ms

        // Coefficients of STDP
        double Cp = 0.1*JEE; // must be 0.01875 (in the paper)
        double Cd = Cp*tpp/tpd; // must be 0.0075 (in the paper)
        double g = 1.25;        // ??????

        //homeostatic
        //double hsig = 0.001*JEE/sqrt(10.0);
        double hsig = 0.001*JEE; 	// i.e. 0.00015 per time step (10 ms)
        int itauh = 100;			// decay time of homeostatic plasticity, (100s)

        double hsd = 0.1;  // release probability
        double hh = 10.0;  // SOME MYSTERIOUS PARAMETER

        double Ip = 1.0; // External current applied to randomly chosen excitatory neurons
        double a = 0.20; // Fraction of neurons to which this external current is applied

        double xEinit = 0.02; // the probability that an excitatory neurons spikes at the beginning of the simulation
        double xIinit = 0.01; // the probability that an inhibitory neurons spikes at the beginning of the simulation
        double tinit = 100.0; // period of time after which STDP kicks in
        double tdur = 1000.0; // time between the stimulations (of which there are two)

        vector<double> dvec;
        vector<int> ivec;
        deque<int> ideque;


        vector< deque<int> > dspts;
        vector<int> x;
        set<int> spts;

        double t = 0;
        int tidx = -1;
        bool trtof = true; // ?????? some flag
        double u;
        int j; 
        vector<int> smpld;
        set<int>::iterator it;
        double k1,k2,k3,k4; 
        bool Iptof = true;

        vector< vector<int> > Jinidx; /* list (len=3000) of lists. Each
        list lists indices of excitatory neurons whose weights are > Jepsilon */

        // we initialize (only excitatory) neurons with values of synaptic efficiency
        vector<double> ys;

        vector< vector<double> > wqqcnt;
        vector<int> ptn_inv; // Excitatory neurons are split into 3 types 0 (500), 1 (500), 2 (1500)

        int ialpha;
        int iusd;
        int iJEI;

        ostringstream ossr; 
        ostringstream ossw; // make a string stream
        ostringstream ossd;

        string fstrr;
        string fstrw;
        string fstrd;

		// classes to stream data to files
        ofstream ofsr;
        ofstream ofsw;
        ofstream ofsd;
        
        double tauh; 	// decay time of homeostatic plasticity, in ms
        double t1;		// stimulus onset, ms
        double t2;	// stimulus offset, ms
        double t3;	// one sec after stim offset, ms
        double t4;	// itb offset
        double t5;	// itb offset + 100 s, ms

        // method declarations
        Model(int, int); // construction
        double dice();
        double ngn();
        void sim(int);
        void setParams(ParamsStructType);
		retParamsStructType getState();
        vector< vector<double> > calc_J(double, double);
        vector<int> rnd_sample(int, int);
        double fd(double, double);
    private:
        double xx;
};

double Model::dice(){
	return rand()/(RAND_MAX + 1.0);
}

double Model::ngn(){
	// sample from a normal distribution based on two uniform distributions
	// WHY IS IT SO?
	double u = Model::dice();
	double v = Model::dice();
	return sqrt(-2.0*log(u))*cos(2.0*pi*v);
}

// choose the neuron ids that will be updated at the current time step
vector<int> Model::rnd_sample(int ktmp, int Ntmp){ // when ktmp << Ntmp
	vector<int> smpld; 
	int xtmp;
	bool tof;
	while( smpld.size() < ktmp ){
		xtmp = (int)floor( Ntmp*Model::dice() ); 
		tof = true;
		// make sure that the sampled id isn't the same as any of the previous ones
		for(int i = 0; i < smpld.size(); i++ ){
			if( xtmp == smpld[i] ){
				tof = false;
			}
		}
		if(tof) smpld.push_back(xtmp);
	}
	return smpld;
}

double Model::fd(double x, double alpha){
	return log(1.0 + alpha*x)/log(1.0 + alpha);
}

void Model::sim(int interval){
	Timer timer;
	// HAVING INITIALIZED THE NETWORK, WE go time step by time step
	while (interval > 0){
		t += h;
		interval -= 1;
				
		// we decide which EXCITATORY neurons will be updated
		// they may or may not be spiking at the current step
		smpld = rnd_sample(SNE, NE);
		
		// we cycle through those chosen neurons
		for(int iidx = 0; iidx < smpld.size(); iidx++){
			int i = smpld[iidx];
			
			// STP (empty square, eq. 6 p.12)
			// if a chosen neuron is ALREADY on
			if( x[i] == 1 ){  
				// then we decrease synaptic efficiency of that neuron by 10% (!!!??? IS THIS STD ???)
				ys[i] -= usd*ys[i]; 
				// remove it from the set of spiking neurons
				it = spts.find( i );
				if( it != spts.end() ) {
					spts.erase( it++ );
				}
				// and turn it OFF
				x[i] = 0;
			}
			// either way

			// WE update the membrane potential of the chosen excitatory neuron (eq.4, p.12)
			// -threshold of update + steady exc input * mean/var of external stim input
			u = -hE + IEex*(mex + sigex*ngn());  // pentagon, p.12
			it = spts.begin();
			//we go over all POSTsynaptic neurons that are spiking now
			while( it != spts.end() ){
				//if a postsynaptic spiking neuron happens to be excitatory, 
				if( *it < NE){
					u += ys[*it]*Jo[i][ *it ];
				//if a postsynaptic spiking neuron happens to be inhibitory, 
				}else{
					u += Jo[i][ *it ];
				}
				++it;
			}

			// at t1 < t < t2 we inject an external stimulus current into the first group of excitatory neurons 
			if( (t1 < t && t < t2) && ptn_inv[i]==1 ) {
				u += Ip;
			}
			
			// ??? this is some post-stimulus period during which u reaches 1/2 of tdur. WHY?
			if( (t2 < t && t < t2+tdur) && ptn_inv[i]==1 ) {
				u += Ip*(t2+tdur - t)/tdur;
			}

			// at t3 < t < t4 we inject an external stimulus current into the second group of excitatory neurons 
			if( (t3 < t && t < t4) && ptn_inv[i]==2 ) {
				u += Ip;
			}
			
			// ??? this is some post-stimulus period during which u reaches 1/2 of tdur. WHY?
			if( (t4 < t && t < t4+tdur) && ptn_inv[i]==2 ) {
				u += Ip*(t4+tdur - t)/tdur;
			} 

			// WE PERFORM AN STDP UPDATE IF u > 0, I.E. IF THE CURRENTLY CHOSEN (i-th) neuron spikes.
			if( u > 0 ){
				// if the POSTsynaptic neuron chosen for update exceeds the threshold, we save its ID in the set "spts"
				spts.insert(i);

				/* dspts saves the TIME of this spike to a DEQUE, such that each row id of this deque
				corresponds to the id of the spiking neuron. The row records the times at which that
				neuron emitted at spike. */
				dspts[i].push_back(t); // SHAPE: (n_postsyn x pytsyn_sp_times)
				x[i] = 1; 
				if( trtof ){
					ofsr << t << " " << i << endl; // record a line to file
				}
				

				// BEGIN STDP ******************************************************************************
				// **************** CHECK THIS IMPLEMENTATION, I DON'T LIKE IT ****************************
				// *****************************************************************************************
				/* First (LTD), we treat the chosen neuron as PREsynaptic and loop over all the POSTSYNAPTIC excitatory 
				neurons that THE CHOSEN NEURON synapses on. Since we're at time t (and this is the latest time),
				the spikes recorded on those "POSTsynaptic" neurons will have an earlier timing than the spike
				recorded on the currently chosen neuron (that we treat as PREsynaptic). This indicates that
				the synaptic weight between this chosen neuron (presynaptic) and all the other neurons
				(postsynaptic) will decrease.  */
				for(int ip = 0; ip < NE; ip++){
					if( Jo[ip][i] > Jepsilon && t > tinit ){							
						// dspts is a deque of spiking times on the ith POSTSYNAPTIC neurons
						for(int sidx = 0; sidx < dspts[ip].size(); sidx++) {
							// { ?????? Eq.7 p.12 }
							/* depression happens if presynaptic spike time (t) happens AFTER
							a postsynaptic spike time (dspts[ip][sidx])
							BY THE WAY: HERE WE HAVE ASYMMETRIC STDP */
							
							Jo[ip][i] -= Cd*fd(Jo[ip][i]/JEE, alpha)*exp( -(t-dspts[ip][sidx])/tpd );
							// Jo[ip][i] += Cd*fd(Jo[ip][i]/JEE, alpha)*exp((t-dspts[ip][sidx])/tpd );
						}
						// we force the weights to be no less than Jmin, THIS WAS NOT IN THE PAPER
						if( Jo[ip][i] < Jmin ) Jo[ip][i] = Jmin;
					}
				}
				
				// (LTP)

				/* Jinidx is a list of lists (shape (2500 POSTsyn, n PREsyn)). E.g. if in row 15 we have number 10,
				it means that the weight between POSTsynaptic neuron 15 and presynaptc neuron 10 is greater than Jepsilon */

				/* we treat the currently chosen neuron as postsynaptic, and we loop over all the presynaptic
				neurons that synapse on the current postsynaptic neuron. At time t (the latest time, and we don't
				yet know any spikes that will happen in the future) all the spikes on the presynaptic neurons with
				id j will have an earlier timing that the spike on the currently chosen neuron i (that we treat
				as postsynaptic for now). This indicates that the weights between the chosen neuron treated as post-
				synaptic for now and all the other neurons (treated as presynaptic for now) will be potentiated.  */
				for(int jidx = 0; jidx < Jinidx[i].size(); jidx++){
					j = Jinidx[i][jidx]; // at each loop we get the id of the jth presynaptic neuron with J > Jepsilon
					if( t > tinit){
						for(int sidx = 0; sidx < dspts[j].size(); sidx++){
							// we loop over all the spike times on the jth PRESYNAPTIC neuron
							Jo[i][j] += g*Cp*exp( -(t-dspts[j][sidx])/tpp );
							// as per eq. (7) p.12, it actually should be 
							// Jo[i][j] += g * Cp * exp((dspts[j][sidx] - t)/tpp)
							// because dspts[j][sidx] is t_pre, and t is t_post
						}
						// we force the weights to be no more than Jmax, THIS WAS NOT IN THE PAPER
						if( Jo[i][j] > Jmax ) Jo[i][j] = Jmax;
					}
				}

				// END STDP *******************************************************************************************
			}
		}

		// we sample INHIBITORY neurons to be updated at the current step
		smpld = rnd_sample(SNI,NI);
		for(int iidx = 0; iidx < smpld.size(); iidx++){
			int i = NE + smpld[iidx];
			/* if this inhibitory neuron is spiking we set it to zero
			in the binary vector x and remove its index from the set
			of currently spiking neurons */
			
			// crazy optimization, but in fact eq.(5) p.12 (filled circle)
			if( x[i] == 1 ){
				it = spts.find( i );
				if( it != spts.end() ) {
					// removing a spike time from the SET of spikes on inhibitory neurons is the same as
					// subtracting them
					spts.erase( it++ ); 
				}
				x[i] = 0;
			}
			
			//update the membrane potential on a chosen inhibitory neuron
			u = -hI + IIex*(mex + sigex*ngn()); // hexagon, eq.5, p.12
			it = spts.begin();
			while( it != spts.end() ){
				u += Jo[i][ *it ];
				++it;
			}

			// if the membrane potential on the currently chosen INHIBITORy neuron 
			// is greater than the threshold, we record a spike on this neuron.
			if( u > 0 ){
				spts.insert(i);
				x[i] = 1; 
				if( trtof ) {
					// if trtof, we log a spike ot the ith postsynaptic neuron
					ofsr << t << " " << i << endl;
				}
			}
		}

		// EVERY 10 ms Becuase STD is slow, we can apply STD updates every 10th step
		if( ( (int)floor(t/h) )%10 == 0 ){
			
			// ??????????????? WHY ?????????????????
			for(int i = 0; i < NE; i++){
				k1 = (1.0 - ys[i])/trec; 
				k2 = (1.0 - (ys[i]+0.5*hsd*k1))/trec;
				k3 = (1.0 - (ys[i]+0.5*hsd*k2))/trec; 
				k4 = (1.0 - (ys[i]+hsd*k3))/trec;
				ys[i] += hsd*(k1 + 2.0*k2 + 2.0*k3 + k4)/6.0;
			}
		}

		// EVERY 1000 ms Homeostatic plasticity, weight clipping, boundary conditions, old spike removal
		if( ( (int)floor(t/h) )%1000 == 0 ){
			
			//Homeostatic Depression
			for(int i = 0; i < NE; i++){
				for(int jidx = 0; jidx < Jinidx[i].size(); jidx++){
					j = Jinidx[i][jidx];
					// ?????????? THAT'S NOT EXACTLY WHAT THE PAPER SAYS
					k1 = (JEEh - Jo[i][j])/tauh;
					k2 = (JEEh - (Jo[i][j]+0.5*hh*k1))/tauh;
					k3 = (JEEh - (Jo[i][j] + 0.5*hh*k2))/tauh;
					k4 = (JEEh - (Jo[i][j] + hh*k3))/tauh;
					Jo[i][j] += hh*(k1 + 2.0*k2 + 2.0*k3 + k4)/6.0 + hsig*ngn();
					// we clip the weights from below and above
					if( Jo[i][j] < Jmin ) Jo[i][j] = Jmin; // ????? Jmin is zero, not 0.0015, as per Table 1
					if( Jo[i][j] > Jmax ) Jo[i][j] = Jmax;
				}
			}
			
			//boundary condition
			for(int i = 0; i < NE; i++){
				double Jav = 0.0;
				for(int jidx = 0; jidx < Jinidx[i].size(); jidx++){
					// find the total weight per each postsynaptic neuron
					Jav += Jo[i][ Jinidx[i][jidx] ];
				}
				// find mean weight per each postsynaptic neuron
				Jav = Jav/( (double)Jinidx[i].size() );
				if( Jav > Jtmax ){
					for(int jidx = 0; jidx < Jinidx[i].size(); jidx++) {
						j = Jinidx[i][jidx];
						// if the total weight exceeds Jtmax, we subtract the excess value
						Jo[i][j] -= (Jav-Jtmax);
						// but if a weight is less that Jmin, we set it to Jmin (clip from below)
						if( Jo[i][j] < Jmin ) {
							Jo[i][j] = Jmin;
						}
					}
				
				// if the total weight is less that Jtmin
				}else if( Jav < Jtmin ){
					for(int jidx = 0; jidx < Jinidx[i].size(); jidx++) {
						j = Jinidx[i][jidx];
						/* ???????? we top up each (!!!???) weight by the difference 
						between the total min and current total weight */
						Jo[i][j] += (Jtmin-Jav);
						// but if a weight is more that Jmax, we clip it to Jmax
						if( Jo[i][j] > Jmax ) {
							Jo[i][j] = Jmax;
						}
					}
				}
			}
			
			// remove spikes older than 500 ms
			for(int i = 0; i < NE; i++){
				for(int sidx = 0; sidx < dspts[i].size(); sidx++){
					//if we have spike times that are occured more than 500 ms ago, we pop them from the deque
					if( t - dspts[i][0] > twnd ) {
						dspts[i].pop_front();
					}
				}
			}
		}

		// EVERY 1s 
		if( ( (int)floor(t/h) )%(1000*100) == 0 ){

			tidx += 1; // we count the number of 1s cycles
			if( tidx%100 < 10 || (t4 < t && t < t5) ){ // if the time is less than 1000s, OR t4 < t < t5
				trtof = true;
			}else{
				trtof = false;
			}
			// every 10s
			if( ( (int)floor(t/h) )%(10000*100) == 0 ){

				// std::cout << "t: " << t << std::endl;

				// create a 3 x 3 matrix of doubles that will hold sumtotals of weights from one type to another
				vector< vector<double> > wqq;
				for(int i = 0; i < 3; i++){
					wqq.push_back( dvec );
					for( j = 0; j < 3; j++) {
						wqq[i].push_back(0.0);
					}
				}
				for(int i = 0; i < NE; i++){
					for(j = 0; j < NE; j++){
						if( abs(Jo[i][j]) > Jepsilon ) {
							// compute the total weights by type of neuron (1,2,0) in the 3 x 3 matrix
							wqq[ ptn_inv[i] ][ ptn_inv[j] ] += Jo[i][j];
						}
					}
				}
				for(int i = 0; i < 3; i++){
					for( j = 0; j < 3; j++) {
						// report the average weights by type
						ofsd << wqq[i][j]/wqqcnt[i][j] << " ";
					}
				}
				ofsd << endl;
			}

			// we log the weight matrix @ 151, 900, 1200 and 1799 s into the sim
			if( tidx == (int)floor(t4/1000.0 + 1.01) || tidx == 900 || tidx == 1200 || tidx == 1799 ){
				std::cout << "logging weights " << std::endl;
				for(int i = 0; i < N; i++){
					for(j = 0; j < N; j++){
						if( abs(Jo[i][j]) > Jepsilon ) {
							// write weights
							ofsw << Jo[i][j] << " " << j << " ";
						} 
					}
					ofsw << endl;
				}
			}
			
			// report sim time every 0.1s of sim (not astronomical) time
			if( ( (int)floor(t/h) )%(100*100) == 0 ){
				// cout << t/1000.0 << endl;
				std::cout << "t: " << t << " ms." << std::endl;
			}
			int s = 0;
			it = spts.begin();
			while( it != spts.end() ){
				++s;
				++it;
			}
			//cout << s << endl;

			// exit if either no neurons are spiking or too many spiking after t > 200 ms
			if( s == 0 || (s > 1.0*NE && t > 200.0) ) {
				std::cout << "Exiting because either 0 or too many spikes at t =" << t << std::endl;
				break;
			}
		}
	}

}

// initialize the weight matrix
vector< vector<double> > Model::calc_J(double JEEinit, double JEI){
	vector< vector<double> > J;	
	int mcount = 0; 
	for(int i = 0; i < NE; i++){
		J.push_back(dvec);
		for(int j = 0; j < NE; j++){
			J[i].push_back(0.0);
			// first E-E weights consistent with the E-E connection probability
			if( i != j && dice() < cEE ){
				J[i][j] += JEEinit*(1.0 + sigJ*ngn());
				// if some weight is out of range, we clip it
				if( J[i][j] < Jmin ) J[i][j] = Jmin;
				if( J[i][j] > Jmax ) J[i][j] = Jmax;
			}
		}
		// then the E-I weights
		for(int j = NE; j < N; j++){
			J[i].push_back(0.0); // here the matrix J is at first of size 2500, we extend it
			if( dice() < cEI ) {
				J[i][j] -= JEI;  /* becuase jth presynaptic inhibitory synapsing on
				an ith excitatory postsynaptic neuron should inhibit it. Hence the minus */
			}
		}
	}
	
	// then the I-E and I-I weights
	for(int i = NE; i < N; i++){
		J.push_back(dvec);
		for(int j = 0; j < NE; j++){
			J[i].push_back(0.0);
			if( dice() < cIE ) {
				J[i][j] += JIE;
			}
		}
		for(int j = NE; j < N; j++){
			J[i].push_back(0.0);
			if( i != j && dice() < cII ) {
				J[i][j] -= JII;
			}
		}
	}

	return J;	
}

// class construction _definition_. Requires no type specifiction.
Model::Model(int _NE, int _NI){
	NE = _NE;
	NI = _NI;
	N = NE + NI;  //
	//how many neurons get updated per time step
	SNE = (int)floor(NE*h/tmE + 0.001);
	SNI = (int)floor(NI*h/tmI + 0.001);

	NEa = (int)floor(NE*a+0.01); // Exact number of excitatory neurons stimulated externally
	pmax = NE/NEa;

    srand((unsigned int)time(NULL));

    ialpha = (int)floor(alpha + 0.01);
	iusd = (int)floor(usd*100.1);
	iJEI = (int)floor(JEI*1000.01);

	// ossr << "binary_model_r_al" << ialpha <<"_u"<< iusd <<"_i"<< iJEI <<"_a"<< ita <<"_b"<< itb << ".txt";
	// fstrr = ossr.str();
	// ofsr.open( fstrr.c_str() );
	ofsr.open( "spike_times.txt" );
	ofsr.precision(10);
	
	// make a string
	// ossw << "binary_model_w_al" << ialpha <<"_u"<< iusd <<"_i"<< iJEI <<"_a"<< ita <<"_b"<< itb << ".txt";
	// fstrw = ossw.str();
	// ofsw.open( fstrw.c_str() );
	ofsw.open("weights.txt");
	
	ossd << "binary_model_d_al" << ialpha <<"_u"<< iusd <<"_i"<< iJEI <<"_a"<< ita <<"_b"<< itb << ".txt"; 
	
	fstrd = ossd.str();
	ofsd.open( fstrd.c_str() );

	tauh = itauh*1000.0; 	// decay time of homeostatic plasticity, in ms
	t1 = 30*1000.0;			// stimulus onset, ms
	t2 = t1 + ita*1000.0;	// stimulus offset, ms
	t3 = t2 + 3.0*1000.0;	// one sec after stim offset, ms
	t4 = t3 + itb*1000.0;	// itb offset
	t5 = t4 + 100*1000.0;	// itb offset + 100 s, ms
	
	
	for(int i = 0; i < NEa; i++) ptn_inv.push_back(1);
	for(int i = NEa; i < 2*NEa; i++) ptn_inv.push_back(2);
	for(int i = 2*NEa; i < NE; i++) ptn_inv.push_back(0);
	
	for(int i = 0; i < 3; i++){
		wqqcnt.push_back(dvec);
		for(int i2 = 0; i2 < 3; i2++) wqqcnt[i].push_back(0.0);
	}

	for (int i = 0; i < NE; i++) {
		ys.push_back(1.0/(1.0 + usd*0.05*trec/tmE));
	}
	
    // initialize the weight matrix Jo
    Jo = calc_J(JEEinit, JEI);

	for(int i = 0; i < NE; i++){
		Jinidx.push_back(ivec); // shape = (3000, max3000)
		for(int i2 = 0; i2 < NE; i2++){
			if( Jo[i][i2] > Jepsilon ){
				Jinidx[i].push_back( i2 );
				wqqcnt[ ptn_inv[i] ][ ptn_inv[i2] ] += 1.0; /* a 3 (type0-2) x 3 (type0-2) matrix whose
				entries are the counts of weights > Jepsilon */
			}
		}
	}

	// create a vector size N and fill it with zeros
	// this vector says if a neuron is spiking or not
	for(int i = 0; i < N; i++){
		x.push_back(0);
	}

	// we remember in spts the ids of neurons that are spiking at the current step
	// and set neurons with these ids to 1
	// at the beginning of the simulation, some neurons have to spike, so we
	// initialize some neurons to 1 to make them spike
	for(int i = 0; i < N; i++){
		// elements corresponding to excitatory neurons are filled with 
		// ones with probability xEinit (0.02)
		if( i < NE && dice() < xEinit ){
			spts.insert(i); 
			x[i] = 1;
		}
		// elements corresponding to inhibitory neurons are filled with 
		// ones with probability xIinit (0.01)
		if( i >= NE && dice() < xIinit ){
			spts.insert(i); 
			x[i] = 1;
		}
	}

	for(int i = 0; i < NE; i++){
		dspts.push_back( ideque );
	}
}

void Model::setParams(Model::ParamsStructType params){
	alpha = params.alpha;   // Degree of log-STDP (50.0)
	usd = params.usd;       // Release probability of a synapse (0.05 - 0.5)
	JEI = params.JEI;       // 0.15 or 0.20
	ita = params.ita;       // stim A duration, s
	itb = params.itb;       // stim B duration, s
	T = params.T;       // simulation time, ms (1800*1000.0)
	h = params.h;       // time step, ms ??????
	// NE = params.NE;     // Nexc
	// NI = params.NI;     // Ninh
	cEE = params.cEE; // 0.2
	cIE = params.cIE; // 0.2
	cEI = params.cEI; // 0.5
	cII = params.cII; // 0.5
	JEE = params.JEE;                // 0.15
	JEEinit = params.JEEinit;        // 0.18
	JIE = params.JIE;                // 0.15
	JII = params.JII;                // 0.06
	JEEh = params.JEEh;      // Standard synaptic weight E-E 0.15
	sigJ = params.sigJ;      // 0.3
	Jtmax = params.Jtmax; // J_maxˆtot (0.25)
	Jtmin = params.Jtmin; // J_minˆtot // ??? NOT IN THE PAPER (0.01)
	hE = params.hE; // Threshold of update of excitatory neurons 1.0
	hI = params.hI; // Threshold of update of inhibotory neurons 1.0
	IEex = params.IEex;     // Amplitude of steady external input to excitatory neurons 2.0
	IIex = params.IIex;     // Amplitude of steady external input to inhibitory neurons 0.5
	mex = params.mex;       // mean of external input 0.3
	sigex = params.sigex;   // variance of external input 0.1
	tmE = params.tmE;  //t_Eud EXCITATORY 5.0
	tmI = params.tmI;  //t_Iud INHIBITORY 2.5
	trec = params.trec;              // recovery time constant (tau_sd, p.13 and p.12) 600.0
	Jepsilon = params.Jepsilon;      // ???????? 0.001
	tpp = params.tpp;          // tau_p 20.0
	tpd = params.tpd;          // tau_d 40.0
	twnd = params.twnd;        // STDP window lenght, ms 500.0
	g = params.g;              // ???? 1.25
	itauh = params.itauh;           // decay time of homeostatic plasticity,s (100)
	hsd = params.hsd;                // 0.1 release probability
	hh = params.hh;                  // SOME MYSTERIOUS PARAMETER 10.0
	Ip = params.Ip;                  // External current applied to randomly chosen excitatory neurons 1.0
	a = params.a;                    // Fraction of neurons to which this external current is applied 0.20
	xEinit = params.xEinit; // prob that an exc neurons spikes at the beginning of the simulation 0.02
	xIinit = params.xIinit; // prob that an inh neurons spikes at the beginning of the simulation 0.01
	tinit = params.tinit;   // period of time after which STDP kicks in 100.0
	tdur = params.tdur;     // time between the stimulations (of which there are two) 1000.0
    t1 = params.t1;         // stim 1 onset

	// recalculate values that depend on the parameters 
	// N = NE + NI;        //
	SNE = (int)floor(NE*h/tmE + 0.001);
	SNI = (int)floor(NI*h/tmI + 0.001);
	Jmax = 5.0 * JEE;         // ???
	Jmin = 0.01 * JEE;        // ????
	Cp = 0.1*JEE;              // must be 0.01875 (in the paper)
	Cd = Cp*tpp/tpd;           // must be 0.0075 (in the paper)
	hsig = 0.001*JEE;           // i.e. 0.00015 per time step (10 ms)
	NEa = (int)floor(NE*a+0.01);    // Exact number of excitatory neurons stimulated externally
	pmax = NE/NEa;

	tauh = itauh*1000.0; 	// decay time of homeostatic plasticity, in ms
	t1 = t1*1000.0;			// stimulus onset, ms
	t2 = t1 + ita*1000.0;	// stimulus offset, ms
	t3 = t2 + 1.0*1000.0;	// one sec after stim offset, ms
	t4 = t3 + itb*1000.0;	// itb offset
	t5 = t4 + 100*1000.0;	// itb offset + 100 s, ms

}

Model::retParamsStructType Model::getState(){
	Model::retParamsStructType ret_struct;
	ret_struct.alpha = alpha;
	ret_struct.usd = usd;
	ret_struct.JEI = JEI;
	ret_struct.ita = ita;
	ret_struct.itb = itb;
	ret_struct.T = T;
	ret_struct.h = h;
	ret_struct.NE = NE;
	ret_struct.NI = NI;
	ret_struct.cEE = cEE;
	ret_struct.cIE = cIE;
	ret_struct.cEI = cEI;
	ret_struct.cII = cII;
	ret_struct.JEE = JEE;
	ret_struct.JEEinit = JEEinit;
	ret_struct.JIE = JIE;
	ret_struct.JII = JII;
	ret_struct.JEEh = JEEh;
	ret_struct.sigJ = sigJ;
	ret_struct.Jtmax = Jtmax;
	ret_struct.Jtmin = Jtmin;
	ret_struct.hE = hE;
	ret_struct.hI = hI;
	ret_struct.IEex = IEex;
	ret_struct.IIex = IIex;
	ret_struct.mex = mex;
	ret_struct.sigex = sigex;
	ret_struct.tmE = tmE;
	ret_struct.tmI = tmI;
	ret_struct.trec = trec;
	ret_struct.Jepsilon = Jepsilon; 
	ret_struct.tpp = tpp;
	ret_struct.tpd = tpd;
	ret_struct.twnd = twnd;
	ret_struct.g = g;
	ret_struct.itauh = itauh;
	ret_struct.hsd = hsd;
	ret_struct.hh = hh;
	ret_struct.Ip = Ip;
	ret_struct.a = a;
	ret_struct.xEinit = xEinit;
	ret_struct.xIinit = xIinit;
	ret_struct.tinit = tinit;
	ret_struct.tdur = tdur;

    ret_struct.Jmin = Jmin;
    ret_struct.Jmax = Jmax;
    ret_struct.Cp = Cp;
    ret_struct.Cd = Cd;
    ret_struct.SNE = SNE;
    ret_struct.SNI = SNI;
    ret_struct.NEa = NEa;
    ret_struct.t = t;
    ret_struct.t1 = t1;

	return ret_struct;
}

extern "C"
{   
    // we create a pointer to the object of type Net. The pointer type must be of the same type as the 
    // object/variable this pointer points to
    Model* createModel(int NE, int NI) {
        return new Model(NE, NI);
    } 

    // this func takes a pointer to the object of type Net and calls the bar() method of that object 
    // because this is a pointer, to access a member of the class, you use an arrow, not dot
    void sim(Model* m, int steps) {
		std::cout << "Simulating " << steps << " time steps" << std::endl;
        m->sim(steps);
    }

	double* getWeights(Model* m) {
		const int x = m->N;
		double* arrayOfPointers = new double[x*x]; // ?
		vector<vector<double>>* arrayOfPointersVec = &(m->Jo);
		for (int i=0; i<x; i++) {
			for (int j=0; j<x; j++) {
				arrayOfPointers[j + x*i] = (double)(*arrayOfPointersVec)[i][j];
			}
		}
		return arrayOfPointers;
    }

    void setParams(Model* m, Model::ParamsStructType params) {
        m->setParams(params);
    }
	
	Model::retParamsStructType getState(Model* m) {	
        return m->getState();
    }
}
