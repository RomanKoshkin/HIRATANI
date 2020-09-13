// Reduced binary model of short- and long-term synaptic plasticity
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

double const T = 1800*1000.0;
double const h = 0.01;

int const NE = 2500;
int const NI = 500;
int const N = NE + NI;

double const cEE = 0.2;
double const cIE = 0.2;
double const cEI = 0.5;
double const cII = 0.5;

double JEE = 0.15;
double JEEh = 0.15;
//double JEI = 0.22;
//double JEIo = 0.15;
double JIE = 0.15;
double JII = 0.06;
double sigJ = 0.3;

double Jtmax = 0.25; //0.25
double Jtmin = 0.01;

double Jmax = 5.0*JEE;
double Jmin = 0.01*JEE;

double const hE = 1.0;
double hI = 1.0;

double const IEex = 2.0;
double const IIex = 0.5;
double mex = 0.3;
double sigex = 0.1;

double const tmE = 5.0;  //t_Eud
double const tmI = 2.5;  //t_Iud

int const SNE = (int)floor(NE*h/tmE + 0.001);
int const SNI = (int)floor(NI*h/tmI + 0.001);

//Short-Term Depression
double trec = 600.0;
//double usyn = 0.1;
double Jepsilon = 0.001;

//STDP
double const tpp = 20.0;
double const tpd = 40.0;
double const twnd = 500.0;
double const Cp = 0.1*JEE;
double const Cd = Cp*tpp/tpd;
double const g = 1.25;

//homeostatic
//double hsig = 0.001*JEE/sqrt(10.0);
double hsig = 0.001*JEE;
int itauh = 100;

double hsd = 0.1;
double hh = 10.0;

double Ip = 1.0; //1.0;
double a = 0.20;
int NEa = (int)floor(NE*a+0.01);
int pmax = NE/NEa;

double o1th = 0.01;

//initial input
double xEinit = 0.02;
double xIinit = 0.01;
double tinit = 100.0;

double tdur = 1000.0;

vector<double> dvec;
vector<int> ivec;
deque<int> ideque;

double dice(){
	return rand()/(RAND_MAX + 1.0);
}

double ngn(){
	double u = dice(); double v = dice();
	return sqrt(-2.0*log(u))*cos(2.0*pi*v);
}

vector<int> rnd_sample(int ktmp, int Ntmp){ // when ktmp << Ntmp
	vector<int> smpld; int xtmp; bool tof;
	while( smpld.size() < ktmp ){
		xtmp = (int)floor( Ntmp*dice() ); tof = true;
		for(int i = 0; i < smpld.size(); i++ ){
			if( xtmp == smpld[i] ) tof = false;
		}
		if(tof) smpld.push_back(xtmp);
	}
	return smpld;
}

double fd(double x, double alpha){
	return log(1.0 + alpha*x)/log(1.0 + alpha);
}

void calc(vector< vector<double> > Jo, double alpha, double usd, double JEI, int ita,int itb,int ik){
	int ialpha = (int)floor(alpha + 0.01);
	int iusd = (int)floor(usd*100.1);
	int iJEI = (int)floor(JEI*1000.01);

	ostringstream ossr; 
	ossr << "binary_model_r_al" << ialpha <<"_u"<< iusd <<"_i"<< iJEI <<"_a"<< ita <<"_b"<< itb <<"_k"<< ik << ".txt";
	string fstrr = ossr.str(); ofstream ofsr; ofsr.open( fstrr.c_str() );
	ofsr.precision(10);
	ostringstream ossw;
	ossw << "binary_model_w_al" << ialpha <<"_u"<< iusd <<"_i"<< iJEI <<"_a"<< ita <<"_b"<< itb <<"_k"<< ik << ".txt";
	string fstrw = ossw.str(); ofstream ofsw; ofsw.open( fstrw.c_str() );
	ostringstream ossd;
	ossd << "binary_model_d_al" << ialpha <<"_u"<< iusd <<"_i"<< iJEI <<"_a"<< ita <<"_b"<< itb <<"_k"<< ik << ".txt"; 
	string fstrd = ossd.str(); ofstream ofsd; ofsd.open( fstrd.c_str() );

	double tauh = itauh*1000.0; 
	double t1 = 30*1000.0;
	double t2 = t1 + ita*1000.0;
	double t3 = t2 + 1.0*1000.0;
	double t4 = t3 + itb*1000.0;
	double t5 = t4 + 100*1000.0;
	cout << t1 << " " << t2 << " " << t3 << " " << t4 << " " << t5 << endl;

	vector<int> ptn_inv;
	for(int i = 0; i < NEa; i++) ptn_inv.push_back(1);
	for(int i = NEa; i < 2*NEa; i++) ptn_inv.push_back(2);
	for(int i = 2*NEa; i < NE; i++) ptn_inv.push_back(0);
	vector< vector<double> > wqqcnt;
	for(int i = 0; i < 3; i++){
		wqqcnt.push_back(dvec);
		for(int i2 = 0; i2 < 3; i2++) wqqcnt[i].push_back(0.0);
	}

	vector<double> ys;
	for(int i = 0; i < NE; i++) ys.push_back( 1.0/(1.0 + usd*0.05*trec/tmE) );

	vector< vector<int> > Jinidx;
	for(int i = 0; i < NE; i++){
		Jinidx.push_back(ivec);
		for(int i2 = 0; i2 < NE; i2++){
			if( Jo[i][i2] > Jepsilon ){
				Jinidx[i].push_back( i2 );
				wqqcnt[ ptn_inv[i] ][ ptn_inv[i2] ] += 1.0;
			}
		}
	}

	vector<int> x;
	for(int i = 0; i < N; i++) x.push_back(0);
	set<int> spts;
	for(int i = 0; i < N; i++){
		if( i < NE && dice() < xEinit ){
			spts.insert(i); x[i] = 1;
		}
		if( i >= NE && dice() < xIinit ){
			spts.insert(i); x[i] = 1;
		}
	}

	vector< deque<int> > dspts;
	for(int i = 0; i < NE; i++) dspts.push_back( ideque );

	int tidx = -1;
	bool trtof = true;
	double u;
	int j; 
	vector<int> smpld;
	set<int>::iterator it;
	double k1,k2,k3,k4; 
	bool Iptof = true;
	
	for(double t = 0; t < T+h; t += h){
		smpld = rnd_sample(SNE,NE);
		for(int iidx = 0; iidx < smpld.size(); iidx++){
			int i = smpld[iidx];
			if( x[i] == 1 ){
				ys[i] -= usd*ys[i];
				it = spts.find( i );
				if( it != spts.end() ) spts.erase( it++ );
				x[i] = 0;
			}
			u = -hE + IEex*(mex + sigex*ngn()); 
			it = spts.begin();
			while( it != spts.end() ){
				if( *it < NE){
					u += ys[*it]*Jo[i][ *it ];
				}else{
					u += Jo[i][ *it ];
				}
				++it;
			}
			if( (t1 < t && t < t2) && ptn_inv[i]==1 ) u += Ip;
			if( (t2 < t && t < t2+tdur) && ptn_inv[i]==1 ) u += Ip*(t2+tdur - t)/tdur;
			if( (t3 < t && t < t4) && ptn_inv[i]==2 ) u += Ip;
			if( (t4 < t && t < t4+tdur) && ptn_inv[i]==2 ) u += Ip*(t4+tdur - t)/tdur;
			if( u > 0 ){
				spts.insert(i); dspts[i].push_back(t); x[i] = 1; 
				if( trtof ) ofsr << t << " " << i << endl;
				//E-pre
				for(int ip = 0; ip < NE; ip++){
					if( Jo[ip][i] > Jepsilon && t > tinit ){
						for(int sidx = 0; sidx < dspts[ip].size(); sidx++){
							Jo[ip][i] -= Cd*fd(Jo[ip][i]/JEE,alpha)*exp( -(t-dspts[ip][sidx])/tpd );
						}
						if( Jo[ip][i] < Jmin ) Jo[ip][i] = Jmin;
					}
				}
				//E-post
				for(int jidx = 0; jidx < Jinidx[i].size(); jidx++){
					j = Jinidx[i][jidx];
					if( t > tinit){
						for(int sidx = 0; sidx < dspts[j].size(); sidx++){
							Jo[i][j] += g*Cp*exp( -(t-dspts[j][sidx])/tpp );
						}
						if( Jo[i][j] > Jmax ) Jo[i][j] = Jmax;
					}
				}
			}
		}
	
		smpld = rnd_sample(SNI,NI);
		for(int iidx = 0; iidx < smpld.size(); iidx++){
			int i = NE + smpld[iidx];
			if( x[i] == 1 ){
				it = spts.find( i );
				if( it != spts.end() ) spts.erase( it++ );
				x[i] = 0;
			}
			u = -hI + IIex*(mex + sigex*ngn()); 
			it = spts.begin();
			while( it != spts.end() ){
				u += Jo[i][ *it ]; ++it;
			}
			if( u > 0 ){
				spts.insert(i); x[i] = 1; 
				if( trtof ) ofsr << t << " " << i << endl;
			}
		}

		// STD ???????????????/
		if( ( (int)floor(t/h) )%10 == 0 ){
			//STD
			for(int i = 0; i < NE; i++){
				k1 = (1.0 - ys[i])/trec; 
				k2 = (1.0 - (ys[i]+0.5*hsd*k1))/trec;
				k3 = (1.0 - (ys[i]+0.5*hsd*k2))/trec; 
				k4 = (1.0 - (ys[i]+hsd*k3))/trec;
				ys[i] += hsd*(k1 + 2.0*k2 + 2.0*k3 + k4)/6.0;
			}
		}


		if( ( (int)floor(t/h) )%1000 == 0 ){
			//Homeostatic Depression
			for(int i = 0; i < NE; i++){
				for(int jidx = 0; jidx < Jinidx[i].size(); jidx++){
					j = Jinidx[i][jidx];
					k1 = (JEEh - Jo[i][j])/tauh; k2 = (JEEh - (Jo[i][j]+0.5*hh*k1))/tauh;
					k3 = (JEEh - (Jo[i][j] + 0.5*hh*k2))/tauh; k4 = (JEEh - (Jo[i][j] + hh*k3))/tauh;
					Jo[i][j] += hh*(k1 + 2.0*k2 + 2.0*k3 + k4)/6.0 + hsig*ngn();
					if( Jo[i][j] < Jmin ) Jo[i][j] = Jmin;
					if( Jo[i][j] > Jmax ) Jo[i][j] = Jmax;
				}
			}
			//boundary condition
			for(int i = 0; i < NE; i++){
				double Jav = 0.0;
				for(int jidx = 0; jidx < Jinidx[i].size(); jidx++) Jav += Jo[i][ Jinidx[i][jidx] ];
				Jav = Jav/( (double)Jinidx[i].size() );
				if( Jav > Jtmax ){
					for(int jidx = 0; jidx < Jinidx[i].size(); jidx++){
						j = Jinidx[i][jidx]; Jo[i][j] -= (Jav-Jtmax);
						if( Jo[i][j] < Jmin ) Jo[i][j] = Jmin;
					}
				}else if( Jav < Jtmin ){
					for(int jidx = 0; jidx < Jinidx[i].size(); jidx++){
						j = Jinidx[i][jidx]; Jo[i][j] += (Jtmin-Jav);
						if( Jo[i][j] > Jmax ) Jo[i][j] = Jmax;
					}
				}
			}
			//STDP
			for(int i = 0; i < NE; i++){
				for(int sidx = 0; sidx < dspts[i].size(); sidx++){
					if( t - dspts[i][0] > twnd ) dspts[i].pop_front();
				}
			}
		}

		if( ( (int)floor(t/h) )%(1000*100) == 0 ){
			tidx += 1;
			if( tidx%100 < 10 || (t4 < t && t < t5) ){
				trtof = true;
			}else{
				trtof = false;
			}
			if( ( (int)floor(t/h) )%(10000*100) == 0 ){
				vector< vector<double> > wqq;
				for(int i = 0; i < 3; i++){
					wqq.push_back( dvec );
					for( j = 0; j < 3; j++) wqq[i].push_back(0.0);
				}
				for(int i = 0; i < NE; i++){
					for(j = 0; j < NE; j++){
						if( abs(Jo[i][j]) > Jepsilon ) wqq[ ptn_inv[i] ][ ptn_inv[j] ] += Jo[i][j];
					}
				}
				for(int i = 0; i < 3; i++){
					for( j = 0; j < 3; j++) ofsd << wqq[i][j]/wqqcnt[i][j] << " ";
				}
				ofsd << endl;
			}
			if( tidx == (int)floor(t4/1000.0 + 1.01) || tidx == 900 || tidx == 1200 || tidx == 1799 ){
				for(int i = 0; i < N; i++){
					for(j = 0; j < N; j++){
						if( abs(Jo[i][j]) > Jepsilon ) ofsw << Jo[i][j] << " " << j << " ";
					}
					ofsw << endl;
				}
			}
			
			if( ( (int)floor(t/h) )%(10000*100) == 0 ) cout << t/1000.0 << endl;
			int s = 0; it = spts.begin();
			while( it != spts.end() ){
				++s; ++it;
			}
			//cout << s << endl;
			if( s == 0 || (s > 1.0*NE && t > 200.0) ) break;
		}	
	}

}

vector< vector<double> > calc_J(double JEEinit, double JEI){
	vector< vector<double> > J;	
	int mcount = 0; 
	for(int i = 0; i < NE; i++){
		J.push_back(dvec);
		for(int j = 0; j < NE; j++){
			J[i].push_back(0.0);
			if( i != j && dice() < cEE ){
				J[i][j] += JEEinit*(1.0 + sigJ*ngn());
				if( J[i][j] < Jmin ) J[i][j] = Jmin;
				if( J[i][j] > Jmax ) J[i][j] = Jmax;
			}
		}
		for(int j = NE; j < N; j++){
			J[i].push_back(0.0);
			if( dice() < cEI ) J[i][j] -= JEI;
		}
	}
	for(int i = NE; i < N; i++){
		J.push_back(dvec);
		for(int j = 0; j < NE; j++){
			J[i].push_back(0.0);
			if( dice() < cIE ) J[i][j] += JIE;
		}
		for(int j = NE; j < N; j++){
			J[i].push_back(0.0);
			if( i != j && dice() < cII ) J[i][j] -= JII;
		}
	}

	return J;	
}

void simul(double alpha, double usd, double JEI, int ita, int itb, int k){	
	double JEEinit = 0.18;
	vector< vector<double> > J = calc_J(JEEinit, JEI);
	calc(J,alpha,usd,JEI,ita,itb,k);
}

int main(int argc, char **argv){
	cout << SNE << " " << SNI << endl;
	double alpha = 0.0; double usd = 0.0; double JEI = 0.0; int ita = 0; int itb = 0; int k = 0;
	if(argc > 1) alpha = atof(argv[1]);
	if(argc > 2) usd = atof(argv[2]);
	if(argc > 3) JEI = atof(argv[3]);
	if(argc > 4) ita = atoi(argv[4]);
	if(argc > 5) itb = atoi(argv[5]);
	if(argc > 6) k = atoi(argv[6]);
	cout << alpha << " " << usd << " " << JEI << " " << ita << " " << itb << " " << k << endl;
	srand((unsigned int)time(NULL));
	simul(alpha,usd,JEI,ita,itb,k);
	return 0;
}
