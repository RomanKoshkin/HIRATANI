// Dendritic hotspot model
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

double const T = 5300*1000.0;//1500*1000.0;
double const To = 5000*1000.0;//1000*1000.0; //ms
double const T2 = 10*1000.0;
double const dt = 0.1; //ms

int const p = 5;
int const N = 10; //number of E-spines

//spine parameters [ms]
double const tauc = 18.0;
double const taum = 3.0;
double const tauN = 15.0;
double const tauA = 3.0;
double const tauBP = 3.0;
double const tauG = 3.0;

double const tauY = 50.0*1000.0;

double const alphaN = 1.0; //fixed
double const betaN = 1.0;//0.0;
double const alphaV = 2.0;//2.0;
double const gammaA = 1.0; //fixed

double const gammaN = 0.20; 
double const gammaBP = 8.0;
//double const gammaG = 0.5;

double const thetap = 70;
double const thetad = 35; 
double const thetaN = 0.0;
//double const Cp = 2.1;
double const Cd = 1.0;

double const yth = 5*50;//20*50;
double const Yp = 1.0*0.001;
double const Yd = 0.5*0.001;
double const hth = 0.01;

//stimuli
double taus = 10.0; //[ms]
double alphas = exp(-dt/taus);
double rsEo = 500.0;
double rxEo = 1.0; //Hz

double rsIo = 1000.0;
double rxIo = 2.0;
double rpost = 5.0;

double umax = 100.0;
double cmax = 10000.0;

double wo = 100.0;
double wmax = 500.0;
double wmin = 0.0;

//delay
//double const idspt = 10.0; //10.0; //5.0//pre-inhibitory delay (heterosynaptic)
double const edspt = 0.0; //pre-excitatory delay (homosynaptic)

//int iidspt = (int)floor(idspt/dt);
int iedspt = (int)floor(iedspt/dt);

deque<double> ddeque;

double dice(){
	return rand()/(RAND_MAX + 1.0);
}

double ngn(){
	double u = dice(); double v = dice();
	return sqrt(-2.0*log(u))*cos(2.0*pi*v);
}

bool Poisson(double rtmp){
	if( dice() < rtmp*0.001*dt ){
		return true;
	}else{
		return false;
	}
}

double rk_y(double wtmp, double tautmp){
	double kw1 = -wtmp/tautmp;
    double w1 = wtmp + kw1*0.5*dt;

    double kw2 = -w1/tautmp;
    double w2 = wtmp + kw2*0.5*dt;

    double kw3 = -w2/tautmp;
    double w3 = wtmp + kw3*dt;

    double kw4 = -w3/tautmp;
    double w4 = wtmp + dt*(kw1 + 2.0*kw2 + 2.0*kw3 + kw4)/6.0;

    return w4;
}

double gN(double utmp){
    return alphaN*utmp + betaN;
}

double gV(double utmp){
    return alphaV*utmp;
}

vector<double> rk_cu(double ctmp, double utmp, double xA, double xN, double xBP, double xG,  double gammaG){
    double kc1 = -ctmp/tauc + gN(utmp)*xN + gV(utmp);
    double ku1 = -utmp/taum + gammaA*xA + gammaN*gN(utmp)*xN + gammaBP*xBP - gammaG*xG;
    double c1 = ctmp + kc1*0.5*dt;
    double u1 = utmp + ku1*0.5*dt;

    double kc2 = -c1/tauc + gN(u1)*xN + gV(u1);
    double ku2 = -u1/taum + gammaA*xA + gammaN*gN(u1)*xN + gammaBP*xBP - gammaG*xG;
    double c2 = ctmp + kc2*0.5*dt;
    double u2 = utmp + ku2*0.5*dt;

    double kc3 = -c2/tauc + gN(u2)*xN + gV(u2);
    double ku3 = -u2/taum + gammaA*xA + gammaN*gN(u2)*xN + gammaBP*xBP - gammaG*xG;
    double c3 = ctmp + kc3*dt;
    double u3 = utmp + ku3*dt;

    double kc4 = -c3/tauc + gN(u3)*xN + gV(u3);
    double ku4 = -u3/taum + gammaA*xA + gammaN*gN(u3)*xN + gammaBP*xBP - gammaG*xG;
    double c4 = ctmp + dt*(kc1 + 2.0*kc2 + 2.0*kc3 + kc4)/6.0;
    double u4 = utmp + dt*(ku1 + 2.0*ku2 + 2.0*ku3 + ku4)/6.0;

	vector<double> cutmp;
	cutmp.push_back(c4); cutmp.push_back(u4);

    return cutmp;
}

double gp(double stmp){
	//if(stmp > 0.0){
	return stmp;
	//}else{
	//	return 0.0;
	//}
}

double gb(double stmp, double Ubzero){
	if(stmp > -Ubzero){
		return stmp;
	}else{
		return -Ubzero;
	}
}

void calc(double Idelay, double gammaG, double Cp, double Ubzero, int ik){
	int iIdelay = (int)floor( Idelay/dt + 0.01 ); //CAUTION normalized
	int igammaG = (int)floor( gammaG*100.0 + 0.01 );
	int iCp = (int)floor( Cp*100.0 + 0.01 );
	int iUbzero = (int)floor( Ubzero*100.0 + 0.01 );
	ostringstream ossy; 
	ossy << "cdpy6a-7-1_id" << iIdelay << "_gg" << igammaG << "_cp" << iCp << "_uz" << iUbzero << "_k" << ik << ".txt";
	string fstry = ossy.str(); ofstream ofsy; ofsy.open( fstry.c_str() );
	ostringstream ossw; 
	ossw << "cdpw6a-7-1_id" << iIdelay << "_gg" << igammaG << "_cp" << iCp << "_uz" << iUbzero << "_k" << ik << ".txt";
	string fstrw = ossw.str(); ofstream ofsw; ofsw.open( fstrw.c_str() );
	ostringstream ossu; 
	ossu << "cdpu6a-7-1_id" << iIdelay << "_gg" << igammaG << "_cp" << iCp << "_uz" << iUbzero << "_k" << ik << ".txt";
	string fstru = ossu.str(); ofstream ofsu; ofsu.open( fstru.c_str() );

	vector<double> sts;
	for(int q = 0; q < p; q++) sts.push_back(0.0);

	vector<double> Cs,Us,ys,ws,xAs,xNs;
	for(int i = 0; i < N; i++){
		Cs.push_back(0.0); Us.push_back(0.0); ys.push_back(0.0); ws.push_back(wo);
		xAs.push_back(0.0); xNs.push_back(0.0); 
	}
	double xBP = 0.0; double xG = 0.0; 

	double prespike = 0.0;
	vector<double> prespikes;
	for(int i = 0; i < N; i++) prespikes.push_back(0.0);
	double postspike = 0.0;
	//assumption: E spike comes earlier
	double inhspike = 0.0;
	deque<double> inhspikehist; 
	for(int tidx = 0; tidx < Idelay/dt + 2; tidx++) inhspikehist.push_back(0.0);

	double mUb = 0.0; double sigUb = 0.0;
	vector<double> msts, sigsts, covs;
	for(int q = 0; q < p; q++){
		msts.push_back(0.0); sigsts.push_back(0.0); covs.push_back(0.0);
	}

	vector<double> cutmps; cutmps.push_back(0.0); cutmps.push_back(0.0);
	double rtmp; double Ub = 0.0;
	for(double t = 0.0; t < T; t += dt){
		//stimulus
		for(int q = 0; q < p; q++) sts[q] = (dice() - 0.5)*(1.0-alphas) + sts[q]*alphas;
		for(int i = 0; i < N; i++){			
			prespikes[i] = 0.0;
			if( t < To && Poisson( rxEo + rsEo*sts[i/2] ) ) prespikes[i] = 1.0;
		}
		postspike = 0.0;
		if( t < To && Poisson(rpost) ) postspike = 1.0;
		inhspike = 0.0;
		if( t < To && Poisson( rxIo + rsIo*sts[0] ) ) inhspikehist[iIdelay] += 1.0;
		inhspike = inhspikehist[0];
		inhspikehist.pop_front(); inhspikehist.push_back(0.0);

		//state update
		xBP += postspike; xBP = rk_y(xBP, tauBP);
		xG += inhspike; xG = rk_y(xG, tauG);
		for(int i = 0; i < N; i++){
			xAs[i] += prespikes[i]; xAs[i] = rk_y(xAs[i], tauA); 
			xNs[i] += prespikes[i]; xNs[i] = rk_y(xNs[i], tauN);
			
			cutmps = rk_cu(Cs[i], Us[i], xAs[i], xNs[i], xBP, xG, gammaG);
        	Cs[i] = cutmps[0]; Us[i] = cutmps[1];
			if( Us[i] > umax ) Us[i] = umax;
			if( Cs[i] > cmax ) Cs[i] = cmax;
        	if(Cs[i] > thetad) ys[i] -= Cd*dt;
        	if(Cs[i] > thetap) ys[i] += Cp*dt;

			ys[i] = rk_y(ys[i], tauY);
			if(ys[i] > yth) ws[i] += Yp*dt;
			if(ys[i] < -yth) ws[i] -= Yd*dt;
			if( ws[i] > wmax ) ws[i] = wmax;
			if( ws[i] < wmin ) ws[i] = wmin;
        }
		Ub = 0.0;
		for(int i = 0; i < N; i++) Ub += (ws[i]/wo)*Us[i]/((double)N);
		Ub = gb(Ub, Ubzero);

		//recording
		mUb += Ub/(T2/dt);
		sigUb += Ub*Ub/(T2/dt);
		for(int q = 0; q < p; q++){
			msts[q] += gp(sts[q])/(T2/dt);
			sigsts[q] += gp(sts[q])*gp(sts[q])/(T2/dt);
		}
		for(int q = 0; q < p; q++) covs[q] += Ub*gp(sts[q])/(T2/dt);

		if( ((int)floor(t/dt))%100000 == 0 ){
			for(int i = 0; i < N; i++) ofsy << ys[i] << " ";
			ofsy << endl;
			for(int i = 0; i < N; i++) ofsw << ws[i] << " ";
			ofsw << endl;

			if(t > 0 && t < To){
				sigUb = sqrt( sigUb - mUb*mUb ); 
				for(int q = 0; q < p; q++) sigsts[q] = sqrt( sigsts[q] - msts[q]*msts[q] );
				for(int q = 0; q < p; q++){
					covs[q] = covs[q] - mUb*msts[q];
					ofsu << covs[q]/(sigUb*sigsts[q]) << " ";
				}
				ofsu << endl;			

				mUb = 0.0; sigUb = 0.0;
				for(int q = 0; q < p; q++){
					msts[q] = 0.0; sigsts[q] = 0.0; covs[q] = 0.0;
				}
			}
		}
	}
}

void simul(double Idelay, double gammaG, double Cp, double Ubzero, int ik){
	calc(Idelay, gammaG, Cp, Ubzero, ik);
}

int main(int argc, char **argv){
	double Idelay = 0.0; double gammaG = 0.0; double Cp = 0.0; double Ubzero = 0.0; int ik = 0; 
	if(argc > 1) Idelay = atof(argv[1]);
	if(argc > 2) gammaG = atof(argv[2]);
	if(argc > 3) Cp = atof(argv[3]);
	if(argc > 4) Ubzero = atof(argv[4]);
	if(argc > 5) ik = atoi(argv[5]);

	srand((unsigned int)(time(NULL)+ik));
	
	simul(Idelay, gammaG, Cp, Ubzero, ik);
	return 0;
}
