// Fully-spiking neuron model of short- and long-term synaptic plasticity
//
// Created by Naoki Hiratani (N.Hiratani@gmail.com)
//

#include <iostream>
#include <vector>
#include <string>
#include <deque>
#include <set>
#include <algorithm>
#include <cmath>
#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <sstream>

using namespace std;
double const pi = 3.14159265;
double const e = 2.71828182;

double const T = 601*1000.0; //3000*1000.0;
double const h = 0.1;

double tauEa = 1.0;
double tauEb = 5.0;

double tauIa = 1.0;
double tauIb = 2.5;

int const NE = 300;
int const NI = 60;
int const N = NE+NI;

int const p = 3;
int NEa = NE/p;

double cEEo = 0.5;
double cLo = 1.0;
int const K = (int)floor(NE*cEEo + 0.01);

double const wEEo = 100.0/((double)K);
double const sigE = 0.1;
double const wIEo = 2.0*wEEo;
//double const wEIo = 0.5*wEEo;
double const wIIo = 0.5*wEEo;

double const wEEoinit = 1.15*wEEo;

double Imin = 0.00001;

//STD
//double const usd = 0.2;
double const tsd = 600.0; //[ms]

//STDP
//double const ssig = 0.5;
double const eta = wEEo*0.1;

double const tpp = 20.0;
double const tpd = 40.0;
double const Ap = 1.25;
double const Apo = 1.0;
double const Ad = Apo*tpp/tpd;
double const wEmin = 0.001*wEEo;
double const wEmax = 10.0*wEEo; //5.0*wEEo

double alpha = 50.0;

double tpmax = 1000.0;

//homeostatic
double const tauh = 100.0*1000.0;
double const sigmah = wEEo*0.0001;

//delay 
int const dmin = (int)floor(0.5/h + 0.01); //0.5ms
int const dmax = (int)floor(1.5/h + 0.01); //1.5ms

double const hE = 0.5;
double const hI = 2.0;
double const rhoampE = 100.0;
double const rhoampI = 200.0;
double const rhop = 10.0; //[Hz]

double const T1 = 0.1*1000.0;
//double const T2 = 10*1000.0;

vector<double> dvec;
vector<int> ivec;
deque<double> ddeque;
deque<int> ideque;

double dice(){
	return rand()/(RAND_MAX + 1.0);
}

double ngn(){
	double u = dice(); double v = dice();
	return sqrt(-2.0*log(u))*cos(2.0*pi*v);
}

int poisson(double fr){
	if( dice() < fr/(1000.0/h) ){
		return 1;
	}else{
		return 0;
	}
}

int bp(double ptmp){
	if( dice() < ptmp ){
		return 1;
	}else{
		return 0;
	}
}

double rk(double Itmp, double ttmp){
	if( Itmp > Imin ){
		double k1,k2,k3,k4;
		k1 = -Itmp/ttmp; k2 = -(Itmp + 0.5*h*k1)/ttmp;
		k3 = -(Itmp + 0.5*h*k2)/ttmp; k4 = -(Itmp + 1.0*h*k3)/ttmp;
		Itmp += h*(k1 + 2.0*k2 + 2.0*k3 + k4)/6.0;
	}else{
		Itmp = Imin;
	}
	return Itmp;
}

double y_rk(double ytmp, double ttmp){
	double k1,k2,k3,k4;
	k1 = (1.0-ytmp)/ttmp; k2 = (1.0-ytmp + 0.5*h*k1)/ttmp;
	k3 = (1.0-ytmp + 0.5*h*k2)/ttmp; k4 = (1.0-ytmp + 1.0*h*k3)/ttmp;
	ytmp += h*(k1 + 2.0*k2 + 2.0*k3 + k4)/6.0;
	return ytmp;
}

double h_rk(double wtmp, double ttmp){
	double k1,k2,k3,k4;
	k1 = (wEEo-wtmp)/ttmp; k2 = (wEEo-wtmp + 0.5*h*k1)/ttmp;
	k3 = (wEEo-wtmp + 0.5*h*k2)/ttmp; k4 = (wEEo-wtmp + 1.0*h*k3)/ttmp;
	wtmp += h*(k1 + 2.0*k2 + 2.0*k3 + k4)/6.0;
	return wtmp + sigmah*ngn();
}

double fd(double wtmp){
	if(wtmp > 0.0){
		return Ad*log(1.0 + alpha*wtmp/wEEo)/log(1.0 + alpha);
	}else{
		return 0.0;
	}
}

double fp(double wtmp){
	return Ap;
}

double Wp(double dtmp){
	return exp(-dtmp/tpp);
}

double Wd(double dtmp){
	return exp(-dtmp/tpd);
}

double calc_LTP(double dtmp, double wtmp, double ssig){
	return eta*(1.0 + ssig*ngn())*Wp(dtmp)*fp(wtmp);
}

double calc_LTD(double dtmp, double wtmp, double ssig){
	return eta*(1.0 + ssig*ngn())*Wd(dtmp)*fd(wtmp);
}

double fIr(double rtmp){
	return 1.0/(1.0 + exp(-rtmp));
}

void calc(double usd, double wEIoo, double rhobeta, double ssig, int ik){
	int iusd = (int)floor(usd*100.01);
	int iwEIoo = (int)floor(wEIoo*100.01);
	int irhobeta = (int)floor(rhobeta + 0.01);
	int issig = (int)floor(ssig*100.01);

	ostringstream ossd; 
	ossd << "spiking_model_d_u" << iusd << "_w" << iwEIoo << "_b" << irhobeta << "_s" << issig << "_k" << ik << ".txt";
	string fstrd = ossd.str(); ofstream ofsd; ofsd.open( fstrd.c_str() );
	ostringstream ossr; 
	ossr << "spiking_model_r_u" << iusd << "_w" << iwEIoo << "_b" << irhobeta << "_s" << issig << "_k" << ik << ".txt";
	string fstrr = ossr.str(); ofstream ofsr; ofsr.open( fstrr.c_str() );
	ofsr.precision(10);
	ostringstream ossw; 
	ossw << "spiking_model_w_u" << iusd << "_w" << iwEIoo << "_b" << irhobeta << "_s" << issig << "_k" << ik << ".txt";
	string fstrw = ossw.str(); ofstream ofsw; ofsw.open( fstrw.c_str() );

	double wEIo = wEIoo*wEEo;

	//excitatory-to-excitatory
	vector< vector<int> > cEEs;
	for(int i = 0; i < NE; i++){
		cEEs.push_back(ivec);
		for(int j = 0; j < NE; j++){
			cEEs[i].push_back(0);
			if(dice() < cEEo) cEEs[i][j] = 1;
		}
	}
		
	vector< vector<double> > ws;
	vector< vector<int> > ds;
	for(int i = 0; i < N; i++){
		ws.push_back(dvec); ds.push_back(ivec);
		for(int j = 0; j < N; j++){
			ws[i].push_back(0.0);
			if(i < NE && j < NE){
				if(cEEs[i][j] == 1){
					if(i/NEa == j/NEa && j/NEa < 2){
						ws[i][j] = 4.0*(1.0 + sigE*ngn())*wEEoinit;
						if(ws[i][j] < wEmin) ws[i][j] = 4.0*1.0*wEEoinit;
					}else{
						ws[i][j] = (1.0 + sigE*ngn())*wEEoinit;
						if(ws[i][j] < wEmin) ws[i][j] = 1.0*wEEoinit;
					}
				}
			}else if(i < NE && j >= NE){
				ws[i][j] = wEIo;
			}else if(i >= NE && j < NE){
				ws[i][j] = wIEo;
			}else{
				ws[i][j] = wIIo;
			}
			ds[i].push_back( dmin+(int)floor( (dmax-dmin)*dice() ) );
		}
	}

	vector<double> hthrs;
	for(int i = 0; i < NE; i++) hthrs.push_back(hE);
	for(int i = 0; i < NI; i++) hthrs.push_back(hI);
	vector<double> rhoexs;
	for(int i = 0; i < NE; i++) rhoexs.push_back(0.0);
	for(int i = 0; i < NI; i++) rhoexs.push_back(0.0);
	vector<double> rhoamps;
	for(int i = 0; i < NE; i++) rhoamps.push_back(rhoampE);
	for(int i = 0; i < NI; i++) rhoamps.push_back(rhoampI);

	vector< deque<double> > uE,uI; //input queue
	vector<double> rhos; //firing probability
	vector<double> IEa,IEb,IIa,IIb; //input current
	for(int j = 0; j < N; j++){
		uE.push_back(ddeque);
		for(int di = 0; di < dmax+2; di++) uE[j].push_back(0.0);
		uI.push_back(ddeque);
		for(int di = 0; di < dmax+2; di++) uI[j].push_back(0.0);
		rhos.push_back(0.0);
		IEa.push_back(0.0); IEb.push_back(0.0);
		IIa.push_back(0.0); IIb.push_back(0.0);
	}
	vector<double> ys; //normalized release probability
	for(int j = 0; j < NE; j++) ys.push_back(0.75);

	vector< deque<double> > ddequevec;
	vector< vector< deque<double> > > pre_spikes;
	for(int i = 0; i < NE; i++){
		pre_spikes.push_back(ddequevec);
		for(int j = 0; j < NE; j++) pre_spikes[i].push_back(ddeque);
	}
	vector< vector< deque<double> > > post_spikes;
	for(int j = 0; j < NE; j++){
		post_spikes.push_back(ddequevec);
		for(int i = 0; i < NE; i++) post_spikes[j].push_back(ddeque);
	}

	bool rtof = true;
	for(double t = 0; t < T; t += h){
		for(int i = 0; i < N; i++){ //excitatory update
			IEa[i] += uE[i][0]; IEb[i] += uE[i][0];
			IEa[i] = rk(IEa[i],tauEa); IEb[i] = rk(IEb[i],tauEb);
			IIa[i] += uI[i][0]; IIb[i] += uI[i][0];
			IIa[i] = rk(IIa[i],tauIa); IIb[i] = rk(IIb[i],tauIb);
			
			rhos[i] = (IEb[i]-IEa[i])/(tauEb-tauEa) - (IIb[i]-IIa[i])/(tauIb-tauIa);
			if(t < T1 && i < NE) rhoexs[i] = rhop; 
			//if(T1 <= t && t < T2 && i < Na) rhoE[i] += rhop; 
			uE[i].pop_front(); uE[i].push_back(0.0);
			uI[i].pop_front(); uI[i].push_back(0.0);
			//if( dice() < 0.00001 ) cout << IEb[i] << " " << IEa[i] << " " << IIb[i] << " " << IIa[i] << " " << rhos[i] << endl;
			if( poisson(rhoexs[i] + rhoamps[i]*fIr(rhobeta*rhos[i]-hthrs[i])) == 1 ){
				if(rtof) ofsr << t << " " << i << endl; 
				if(i < NE){
					for(int j = 0; j < N; j++){
						if(i < NE){
							if(cEEs[i][j] == 1) uE[j][ds[j][i]] += ys[i]*ws[j][i];
						}else{
							uE[j][ds[j][i]] += ws[j][i];
						}
					}
					//STD
					ys[i] -= usd*ys[i];
					//STDP
					for(int j = 0; j < NE; j++){
						if(cEEs[j][i] == 1) pre_spikes[j][i].push_back( t+ds[j][i]*h );
						if(cEEs[i][j] == 1) post_spikes[i][j].push_back(t);
					}
				}else{
					for(int j = 0; j < N; j++) uI[j][ds[j][i]] += ws[j][i];	
				}
			}
		}
		for(int i = 0; i < N; i++) rhoexs[i] = 0.0;

		//STD
		for(int i = 0; i < NE; i++) ys[i] = y_rk(ys[i],tsd);
		
		//weight update
		double wEtmp = 0.0;
		//E-LTP
		for(int i = 0; i < NE; i++){
			for(int j = 0; j < NE; j++){
				if( cEEs[i][j] == 1 && post_spikes[i][j].size() > 0 ){
					if( abs(post_spikes[i][j][post_spikes[i][j].size()-1] - t) < 0.1*h ){
						wEtmp = ws[i][j];
						for(int tidx = 0; tidx < pre_spikes[i][j].size(); tidx++){
							if( pre_spikes[i][j][tidx] < t ){
								ws[i][j] += calc_LTP(t-pre_spikes[i][j][tidx],wEtmp,ssig);
								if( ws[i][j] < wEmin ) ws[i][j] = wEmin;
								if( ws[i][j] > wEmax ) ws[i][j] = wEmax;
							}
						}
					}
					if( t - post_spikes[i][j][0] > tpmax ) post_spikes[i][j].pop_front();
				}				
			}
		}
		
		//E-LTD
		for(int i = 0; i < NE; i++){	
			for(int j = 0; j < NE; j++){
				if( cEEs[i][j] == 1 && pre_spikes[i][j].size() > 0 ){
					if( abs(pre_spikes[i][j][pre_spikes[i][j].size()-1] - t) < 0.1*h ){
						wEtmp = ws[i][j];
						for(int tidx = 0; tidx < post_spikes[i][j].size(); tidx++){
							if( post_spikes[i][j][tidx] < t ){
								ws[i][j] -= calc_LTD(t-post_spikes[i][j][tidx],wEtmp,ssig);
								if( ws[i][j] < wEmin ) ws[i][j] = wEmin;
								if( ws[i][j] > wEmax ) ws[i][j] = wEmax;
							}
						}
					}
					if( t - pre_spikes[i][j][0] > tpmax ) pre_spikes[i][j].pop_front();
				}				
			}
		}
		//homeostatic plasticity
		for(int i = 0; i < NE; i++){	
			for(int j = 0; j < NE; j++){
				if(cEEs[i][j] == 1){
					ws[i][j] = h_rk(ws[i][j],tauh);
					if( ws[i][j] < wEmin ) ws[i][j] = wEmin;
					if( ws[i][j] > wEmax ) ws[i][j] = wEmax;
				}
			}
		}
		
		if( ((int)floor(t/h))%(100*10) == 0 ){
			vector< vector<double> > wEms;
			for(int q1 = 0; q1 < p; q1++){
				wEms.push_back(dvec);
				for(int q2 = 0; q2 < p; q2++) wEms[q1].push_back(0.0);
			}
			for(int i = 0; i < NE; i++){
				for(int j = 0; j < NE; j++) wEms[i/NEa][j/NEa] += ws[i][j]/(cEEo*NEa*NEa);
			}
			for(int q1 = 0; q1 < p; q1++){
				for(int q2 = 0; q2 < p; q2++) ofsd << wEms[q1][q2] << " ";
			}
			ofsd << endl;
		}
		if( ((int)floor(t/h))%(10*1000*10) == 0 ){
			if( t < 19*1000.0 || t > 579*1000.0 ){
				rtof= true;
			}else{
				rtof = false;
			}
			cout << t/1000.0 << endl;
		}
		if( ((int)floor(t/h))%(100*1000*10) == 0 ){//measurement
			for(int i = 0; i < N; i++){
				for(int j = 0; j < N; j++) ofsw << ws[i][j] << " ";
				ofsw << endl;
			}	
		}
	}
}

void simul(double usd, double wEIoo, double rhobeta, double ssig, int ik){
	calc(usd, wEIoo, rhobeta, ssig, ik);
}

int main(int argc, char **argv){
	double usd = 0.0; double wEIoo = 0.0; double rhobeta = 0.0; 
	double ssig = 0.0; int ik = 0;
	if(argc > 1) usd = atof(argv[1]);
	if(argc > 2) wEIoo = atof(argv[2]);
	if(argc > 3) rhobeta = atof(argv[3]);
	if(argc > 4) ssig = atof(argv[4]);
	if(argc > 5) ik = atoi(argv[5]);
	srand((unsigned int)time(NULL));
	simul(usd,wEIoo,rhobeta,ssig,ik);
	return 0;
}
