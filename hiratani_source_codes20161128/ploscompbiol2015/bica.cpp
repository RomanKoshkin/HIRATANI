// Model of Bayesian ICA
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

double const T = 1000.0*1000.0;//4000.0*1000.0; //3000*1000.0; //3000*1000.0;
double const h = 0.05; //0.05; //0.10[ms] f stands for fast-temporal simulation

double tauEa = 1.0;
double tauEb = 5.0;
double tauIa = 0.5;
double tauIb = 2.5;
double tauLa = 0.8;
double tauLb = 4.0;

int const L = 400;
int const La = 100;
int const p = 4;
int const M = 40; //M=50
int const N = 40; //N=50

int Ma = M/p;
int Na = N/p;

double const wEo = 2.5; //1.0
double const sigE = 0.1;
double const sigI = 0.1;

double Imin = 0.00001;

//STDP
double const ssig = 0.3;
double const eta = wEo*0.05;

//STDP : E-to-E : log-Hebbian
double const tpp = 17.0;
double const tpd = 34.0;
double const Ap = 1.0;
double const Ad = Ap*tpp/tpd;
double const wEmin = 0.001*wEo;
double const wEmax = 100.0*wEo;

double alpha = 20.0;
double beta = 50.0;

//STDP : E-to-I (L) : Hebbian
//double const wLoinit = 20.0;
//double const wLo = 80.0;
//double const cL = 0.5;

//STDP : I-to-E (I) : Hebbian
//double const wIoinit = 20.0;
//double const wIo = 80.0;
//double const cI = 0.5;

double const wLo = 100.0;
//double const wIo = 50.0;

//delay 
int const dEamin = (int)floor(2.0/h + 0.01); //2.0ms
int const dEamax = (int)floor(4.0/h + 0.01); //4.0ms
int const dEdmin = (int)floor(0.5/h + 0.01); //0.5ms
int const dEdmax = (int)floor(1.5/h + 0.01); //1.5ms
int const dEmax = dEamax + dEdmax;
int const dlmin = (int)floor(0.2/h + 0.01); //0.2ms
int const dlmax = (int)floor(1.2/h + 0.01); //1.2ms

double gtmax = 50.0; //40.0;//20.0; //[ms]
int igtmax = (int)floor(gtmax/h);

double stw = 50.0; //20.0; //[ms]
int istw = (int)floor(stw/h + 0.01);

int kg = 3;
double ttheta = 2.0; //[ms]

//double dt1 = 5.0; //10.0;//1.0;
//int idt1 = (int)floor(dt1/h + 0.01);

double dt2 = 10.0;
int idt2 = (int)floor(dt2/h + 0.01);

double const nuo = 10.0;
double const nuIo = 0.0;//Hz
double const nuXo = 2.0;

//double const nuSo = nuo*dt1/1000.0;

double qtinit = 0.1;
double qtinitsigma = 0.3;

double dq = 0.001;

//double etaq = 0.01; //0.001;//0.001; //0.01;

double D[16][4] = {{0,0,0,0},
    		       {0,0,0,1},
     		       {0,0,1,0},
     		       {0,0,1,1},
     		       {0,1,0,0},
     		       {0,1,0,1},
     		       {0,1,1,0},
     		       {0,1,1,1},
     		       {1,0,0,0},
     		       {1,0,0,1},
     		       {1,0,1,0},
     		       {1,0,1,1},
     		       {1,1,0,0},
     		       {1,1,0,1},
     		       {1,1,1,0},
     		       {1,1,1,1}};

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

double gdist(double ttmp, int kg, double ttheta){
	return pow(ttmp,kg-1)*exp(-ttmp/ttheta)/(pow(ttheta,kg)*tgamma(kg));
}

double Wd(double wtmp){
	if(wtmp > 0.0){
		return Ad*log(1.0 + alpha*wtmp/wEo)/log(1.0 + alpha);
	}else{
		return 0.0;
	}
}

double Wp(double wtmp){
	return Ap*exp(-wtmp/(wEo*beta));
}

int get_dkidx(vector<long double> pysk){
	vector<double> cumpysk; cumpysk.push_back(0.0);
	for(int didx = 0; didx < pysk.size(); didx++) cumpysk.push_back( cumpysk[didx] + pysk[didx] );

	double rtmp = dice();
	int dkidx = 0;
	for(int didx = 0; didx < 16; didx++){
		if( cumpysk[didx] <= rtmp && rtmp < cumpysk[didx+1] ){
			dkidx = didx;
		}
	}
	return dkidx;
}

void calc(double qS, double qN, double dt1, double etaq, int ik){
	int iqS = (int)floor(qS*100.1);
	int iqN = (int)floor(qN*100.1);
	int idt1 = (int)floor(dt1/h + 0.01);
	int ietaq = (int)floor(etaq*1000.0 + 0.01);
	
	ostringstream ossd; 
	ossd << "bica_d_s" << iqS << "_n" << iqN << "_t" << idt1 << "_e" << ietaq << "_k" << ik << ".txt";
	string fstrd = ossd.str(); ofstream ofsd; ofsd.open( fstrd.c_str() );	
	ostringstream osss; 
	osss << "bica_s_s" << iqS << "_n" << iqN << "_t" << idt1 << "_e" << ietaq << "_k" << ik << ".txt";
	string fstrs = osss.str(); ofstream ofss; ofss.open( fstrs.c_str() );	
	ofss.precision(10);	
	ostringstream ossy; 
	ossy << "bica_y_s" << iqS << "_n" << iqN << "_t" << idt1 << "_e" << ietaq << "_k" << ik << ".txt";
	string fstry = ossy.str(); ofstream ofsy; ofsy.open( fstry.c_str() );
	ostringstream ossq; 
	ossq << "bica_q_s" << iqS << "_n" << iqN << "_t" << idt1 << "_e" << ietaq << "_k" << ik << ".txt";
	string fstrq = ossq.str(); ofstream ofsq; ofsq.open( fstrq.c_str() );

	double const nuSo = nuo*dt1/1000.0;
	double Qss[4][4] = {{qS,qN,0.0,qN}, {qN,qS,qN,0.0},{0.0,qN,qS,qN},{qN,0.0,qN,qS} };
	
	vector<double> gmpdf;
	for(int tidx = 0; tidx < igtmax; tidx++){
		gmpdf.push_back( 1000.0*gdist(tidx*h,kg,ttheta) );
	}
	vector< deque<double> > Igs;
	for(int q = 0; q < p; q++){
		Igs.push_back(ddeque);
		for(int tidx = 0; tidx < igtmax; tidx++) Igs[q].push_back(0.0);
	}
	vector<double> gks;
	for(int tidx = 0; tidx < istw/idt1; tidx++){
		gks.push_back( 0.0 );
		for(int dtidx = 0; dtidx < idt1; dtidx++){
			gks[tidx] += gdist(tidx*dt1+dtidx*h, kg, ttheta)*h;
		}
		cout << gks[tidx] << " ";
	}
	cout << endl;

	vector<double> nucs; //correlation matrix
	for(int q1 = 0; q1 < L/La; q1++) nucs.push_back(nuXo);

	vector<int> so;
	for(int q = 0; q < p; q++) so.push_back(0);
	vector<double> rhos;
	for(int q = 0; q < p; q++) rhos.push_back(0.0);

	vector<double> xs;
	for(int i  = 0; i < L; i++) xs.push_back(0.0);

	vector< deque<double> > yss;
	for(int q = 0; q < p; q++){
		yss.push_back(ddeque);
		for(int tidx = 0; tidx < istw/idt1; tidx++){
			yss[q].push_back(0.0);
			if( dice() < nuSo ) yss[q][tidx] = 1.0;
		}
	}

	vector< vector<double> > Qt;
	for(int q = 0; q < p; q++){
		Qt.push_back(dvec);
		for(int i = 0; i < L; i++){
			Qt[q].push_back( qtinit*(1.0 + qtinitsigma*ngn()) );
			if( Qt[q][i] > 1.0 - dq ) Qt[q][i] = 1.0-dq;
			if( Qt[q][i] < dq ) Qt[q][i] = dq;
		}
	}	

	bool rtof = true; double nutmp;
	vector<int> spt;
	for(double t = 0; t < T; t += h){
		for(int q = 0; q < p; q++){
			if( poisson(nuo)==1 ){
				for(int tidx = 0; tidx < igtmax; tidx++) Igs[q][tidx] += gmpdf[tidx];
				if( rtof ) ofss << t << " " << q << endl;
			}
		}
		for(int q1 = 0; q1 < L/La; q1++){
			for(int l = 0; l < La; l++){
				nutmp = nucs[q1];
				for(int q2 = 0; q2 < p; q2++) nutmp += Qss[q2][q1]*Igs[q2][0];
				if( poisson(nutmp) == 1 ){
					spt.push_back( q1*La+l );
					if( rtof ) ofsd << t << " " << q1*La+l << endl;
				}
			}
		}

		for(int iidx = 0; iidx < spt.size(); iidx++){
			xs[ spt[iidx] ] = 1.0; 
		}
		spt.clear();
		for(int q = 0; q < p; q++){
			Igs[q].pop_front(); Igs[q].push_back(0.0);
		}

		if( ((int)floor(t/h))%idt1 == 0 ){
			vector<long double> pysk; long double lpysktmp; long double Zpysk = 0.0;
			for(int didx = 0; didx < 16; didx++){
				for(int q = 0; q < p; q++) yss[q][0] = D[didx][q];
				vector<double> gstmps;
				for(int q = 0; q < p; q++){
					gstmps.push_back(0.0);
					for(int tidx = 0; tidx < yss[q].size(); tidx++){
						gstmps[q] += gks[tidx]*yss[q][tidx];
					}
				}	
				vector<long double> pik;
				for(int i = 0; i < L; i++){
					long double piktmp = 1.0 - nuSo;
					for(int q = 0; q < p; q++) piktmp *= (1.0 - Qt[q][i]*gstmps[q]);
					pik.push_back(1.0-piktmp);
				}
				lpysktmp = 0.0; 
				for(int q = 0; q < p; q++){
					if( D[didx][q] == 1 ){
						lpysktmp += log(nuSo);
					}else{
						lpysktmp += log(1.0 - nuSo);
					}
				}
				for(int i = 0; i < L; i++) lpysktmp += log(xs[i]*pik[i] + (1.0-xs[i])*(1.0-pik[i]));
				pysk.push_back(exp(lpysktmp));
				//if( dice() < 0.001 ) cout << pysk[didx] << endl;
				Zpysk += exp(lpysktmp);
				
			}
			for(int didx = 0; didx < 16; didx++) pysk[didx] = pysk[didx]/Zpysk;

			int dkidx = get_dkidx(pysk);
			for(int q = 0; q < p; q++){
				yss[q][0] = D[dkidx][q];
				if(rtof) ofsy << t << " " << dkidx << endl;
			}
			vector<double> gstmp2s;
			for(int q = 0; q < p; q++){
				gstmp2s.push_back(0.0);
				for(int tidx = 0; tidx < yss[q].size(); tidx++){
					gstmp2s[q] += gks[tidx]*yss[q][tidx];
				}
			}
			vector<long double> pik2;
			for(int i = 0; i < L; i++){
				long double piktmp = 1.0 - nuSo;
				for(int q = 0; q < p; q++) piktmp *= (1.0 - Qt[q][i]*gstmp2s[q]);
				pik2.push_back(1.0-piktmp);
			}

			for(int q = 0; q < p; q++){
				for(int i = 0; i < L; i++){
					Qt[q][i] += etaq*(2.0*xs[i]-1.0)*gstmp2s[q]/( ( xs[i]*pik2[i]/(1.0-pik2[i]) + (1.0-xs[i]) )*( 1.0 - Qt[q][i]*gstmp2s[q] ) );
					if( Qt[q][i] > 1.0 - dq ) Qt[q][i] = 1.0-dq;
					if( Qt[q][i] < dq ) Qt[q][i] = dq;
				}
			}

			for(int q = 0; q < p; q++){
				yss[q].pop_back(); yss[q].push_front(0.0);
			}
			for(int i = 0; i < L; i++) xs[i] = 0.0;
		}
		
		if( ((int)floor(t/h))%100 == 0 ){//measurement
			if( ((int)floor(t/h))%(1*1000*100) == 0 ){
				for(int q = 0; q < p; q++){
					for(int i = 0; i < L; i++) ofsq << Qt[q][i] << " ";
					ofsq << endl;
				}
				cout << t/1000.0 << endl;
			}
			if( t < 10*1000.0 || t > 990*1000.0 ){
				rtof= true;
			}else{
				rtof = false;
			}
		}
	}
}

void simul(double qS, double qN, double dt1, double etaq, int ikmax){
	for(int ik = 0; ik < ikmax; ik++){
		calc(qS, qN, dt1, etaq, ik);
	}
}

int main(int argc, char **argv){
	double qS = 0.0; double qN = 0.0;
	double dt1 = 0.0; double etaq = 0.0; int ikmax = 0;
	if(argc > 1) qS = atof(argv[1]);
	if(argc > 2) qN = atof(argv[2]);
	if(argc > 3) dt1 = atof(argv[3]);
	if(argc > 4) etaq = atof(argv[4]);
	if(argc > 5) ikmax = atoi(argv[5]);
	srand((unsigned int)time(NULL));
	simul(qS,qN,dt1,etaq,ikmax);
	return 0;
}
