// Model of excitatory STDP
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

double const T = 3000*1000.0; //3000*1000.0;
double const h = 0.05; //0.10[ms]

double tauEa = 1.0;
double tauEb = 5.0;
double tauIa = 0.5;
double tauIb = 2.5;
double tauLa = 0.8;
double tauLb = 4.0;

int const L = 400;
int const La = 100;
int const p = 2;
int const M = 20; //M=50
int const N = 20; //N=50

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

double gtmax = 20.0; //[ms]
int igtmax = (int)floor(gtmax/h);

//int kg = 5;
//double ttheta = 2.0; //[ms]

double const nuo = 10.0;
double const nuIo = 0.0;//Hz

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

void calc(double qA, double qB, double wIo, int kg, double ttheta, int ik){
	int iqA = (int)floor(qA*100.1);
	int iqB = (int)floor(qB*100.1);
	int iwIo = (int)floor(wIo + 0.01);
	int ittheta = (int)floor( ttheta*100.01 );
	
	ostringstream ossd; 
	ossd << "estdp_d_a" << iqA << "_b" << iqB << "_i" << iwIo << "_g" << kg << "_t" << ittheta << "_k" << ik << ".txt";
	string fstrd = ossd.str(); ofstream ofsd; ofsd.open( fstrd.c_str() );	
	ostringstream osss; 
	osss << "estdp_s_a" << iqA << "_b" << iqB << "_i" << iwIo << "_g" << kg << "_t" << ittheta << "_k" << ik << ".txt";
	string fstrs = osss.str(); ofstream ofss; ofss.open( fstrs.c_str() );	
	ofss.precision(10);	
	ostringstream ossr; 
	ossr << "estdp_r_a" << iqA << "_b" << iqB << "_i" << iwIo << "_g" << kg << "_t" << ittheta << "_k" << ik << ".txt";
	string fstrr = ossr.str(); ofstream ofsr; ofsr.open( fstrr.c_str() );
	ofsr.precision(10);
	ostringstream ossw; 
	ossw << "estdp_w_a" << iqA << "_b" << iqB << "_i" << iwIo << "_g" << kg << "_t" << ittheta << "_k" << ik << ".txt"; 
	string fstrw = ossw.str(); ofstream ofsw; ofsw.open( fstrw.c_str() );

	double Qss[2][4] = {{qA,0.0,0.0,0.0}, {0.0,qB,0.0,0.0}};
	
	vector<double> gmpdf;
	for(int tidx = 0; tidx < igtmax; tidx++){
		gmpdf.push_back( 1000.0*gdist(tidx*h,kg,ttheta) );
	}
	vector< deque<double> > Igs;
	for(int q = 0; q < p; q++){
		Igs.push_back(ddeque);
		for(int tidx = 0; tidx < igtmax; tidx++) Igs[q].push_back(0.0);
	}

	//input-to-excitatory
	vector< vector<double> > wEs;
	vector< vector<int> > dEas; vector< vector<int> > dEds; vector< vector<int> > dEtots; 
	for(int j = 0; j < M; j++){
		wEs.push_back(dvec);
		dEas.push_back(ivec); dEds.push_back(ivec); dEtots.push_back(ivec);
		for(int i = 0; i < L; i++){
			wEs[j].push_back( (1.0 + sigE*ngn())*wEo );
			dEas[j].push_back( dEamin+(int)floor( (dEamax-dEamin)*dice() ) );
			dEds[j].push_back( dEdmin+(int)floor( (dEdmax-dEdmin)*dice() ) );
			dEtots[j].push_back( dEas[j][i] + dEds[j][i] );
		}
	}
	
	//excitatory-to-inhibitory
	vector< vector<double> > wLs; vector< vector<int> > dLs; 
	for(int k = 0; k < N; k++){
		wLs.push_back(dvec);
		for(int j = 0; j < M; j++){
			wLs[k].push_back( 0.0 );
			if( k/Na == j/Ma ){
				wLs[k][j] = (1.0 + sigI*ngn())*wLo;
			}
		}
		dLs.push_back(ivec);
		for(int j = 0; j < M; j++) dLs[k].push_back( dlmin+(int)floor( (dlmax-dlmin)*dice() ) );
	}

	//inhibitory-to-excitatory
	vector< vector<double> > wIs; vector< vector<int> > dIs; 
	for(int j = 0; j < M; j++){
		wIs.push_back(dvec);
		for(int k = 0; k < N; k++){
			wIs[j].push_back( 0.0 );
			if( j/Ma != k/Na ){
				wIs[j][k] = (1.0 + sigI*ngn())*wIo;
			}
		}
		dIs.push_back(ivec);
		for(int k = 0; k < N; k++) dIs[j].push_back( dlmin+(int)floor( (dlmax-dlmin)*dice() ) );
	}
	
	vector< vector<int> > wIEoutidx,wEIinidx;
	for(int j = 0; j < M; j++){
		wIEoutidx.push_back(ivec); wEIinidx.push_back(ivec);
		for(int k = 0; k < N; k++){
			if( wLs[k][j] > 0.0 ) wIEoutidx[j].push_back(k);
			if( wIs[j][k] > 0.0 ) wEIinidx[j].push_back(k);
		}
	}
	vector< vector<int> > wEIoutidx,wIEinidx;
	for(int k = 0; k < N; k++){
		wEIoutidx.push_back(ivec); wIEinidx.push_back(ivec);
		for(int j = 0; j < M; j++){
			if( wIs[j][k] > 0.0 ) wEIoutidx[k].push_back(j);
			if( wLs[k][j] > 0.0 ) wIEinidx[k].push_back(j);
		}
	}

	vector<double> nucs; //correlation matrix
	for(int q1 = 0; q1 < L/La; q1++){
		nucs.push_back( nuo );
		for(int q2 = 0; q2 < p; q2++) nucs[q1] -= nuo*Qss[q2][q1];
	}

	vector< deque<double> > uE,uI,uL; //input queue
	vector<double> rhoE,rhoI; //firing probability
	vector<double> IEa,IEb,IIa,IIb,ILa,ILb; //input current
	for(int j = 0; j < M; j++){
		uE.push_back(ddeque); uI.push_back(ddeque);
		for(int di = 0; di < dEmax+2; di++) uE[j].push_back(0.0);
		for(int di = 0; di < dlmax+2; di++) uI[j].push_back(0.0);
		rhoE.push_back(0.0);
		IEa.push_back(0.0); IEb.push_back(0.0);
		IIa.push_back(0.0); IIb.push_back(0.0);
	}
	for(int k = 0; k < N; k++){
		uL.push_back(ddeque);
		for(int di = 0; di < dlmax+2; di++) uL[k].push_back(0.0);
		rhoI.push_back(0.0);
		ILa.push_back(0.0); ILb.push_back(0.0);
	}

	deque< vector<int> > iveque;
	vector< deque<double> > ddec;

	vector< vector<double> > yd; //LTD
	vector< vector< deque<double> > > uyd; //LTD-post
	vector< deque< vector<int> > > syd; //LTD-pre
	vector< vector<double> > yp; //LTP
	vector< vector< deque<double> > > uyp; //LTP-pre
	vector< deque< vector<int> > > syp; //LTP-post

	for(int j = 0; j < M; j++){
		yd.push_back( dvec );
		uyd.push_back( ddec );
		for(int i = 0; i < L; i++){
			yd[j].push_back(0.0);
			uyd[j].push_back(ddeque);
			for(int di = 0; di < dEmax+2; di++) uyd[j][i].push_back(0.0);
		}
		syd.push_back( iveque );
		for(int di = 0; di < dEmax+2; di++) syd[j].push_back(ivec);
	}
	for(int i = 0; i < L; i++){
		yp.push_back( dvec );
		uyp.push_back( ddec );
		for(int j = 0; j < M; j++){
			yp[i].push_back(0.0);
			uyp[i].push_back(ddeque);
			for(int di = 0; di < dEmax+2; di++) uyp[i][j].push_back(0.0);
		}
		syp.push_back( iveque );
		for(int di = 0; di < dEmax+2; di++) syp[i].push_back(ivec);
	}

	vector<int> so;
	for(int q = 0; q < p; q++) so.push_back(0);
	vector<double> rhos;
	for(int q = 0; q < p; q++) rhos.push_back(0.0);

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
			int i = spt[iidx];
			for(int j = 0; j < M; j++){
				uE[j][dEtots[j][i]] += wEs[j][i];
				uyp[i][j][dEas[j][i]] += eta*(1.0 + ssig*ngn());
				syd[j][dEas[j][i]].push_back(i);
			}
		}
		spt.clear();
		for(int q = 0; q < p; q++){
			Igs[q].pop_front(); Igs[q].push_back(0.0);
		}

		for(int j = 0; j < M; j++){ //excitatory update
			IEa[j] += uE[j][0]; IEb[j] += uE[j][0];
			IIa[j] += uI[j][0]; IIb[j] += uI[j][0];

			IEa[j] = rk(IEa[j],tauEa); IEb[j] = rk(IEb[j],tauEb);
			IIa[j] = rk(IIa[j],tauIa); IIb[j] = rk(IIb[j],tauIb);

			rhoE[j] = (IEb[j]-IEa[j])/(tauEb - tauEa) - (IIb[j]-IIa[j])/(tauIb - tauIa);
			uE[j].pop_front(); uE[j].push_back(0.0);
			uI[j].pop_front(); uI[j].push_back(0.0);
			if( poisson(rhoE[j]) == 1 ){
				if(rtof) ofsr << t << " " << j << endl; 
				for(int kidx = 0; kidx < wIEoutidx[j].size(); kidx++){
					int k = wIEoutidx[j][kidx]; uL[k][dLs[k][j]] += wLs[k][j];
				}
				for(int i = 0; i < L; i++){
					uyd[j][i][dEds[j][i]] += eta*(1.0 + ssig*ngn());
					syp[i][dEds[j][i]].push_back(j);
				}
			}
		}

		for(int k = 0; k < N; k++){ //inhibitory update
			ILa[k] += uL[k][0]; ILb[k] += uL[k][0];
			ILa[k] = rk(ILa[k],tauLa);
			ILb[k] = rk(ILb[k],tauLb);
			rhoI[k] = (ILb[k]-ILa[k])/(tauLb - tauLa) + nuIo;
			uL[k].pop_front(); uL[k].push_back(0.0);
			if( poisson(rhoI[k]) == 1 ){
				if(rtof) ofsr << t << " " << M+k << endl; 
				for(int jidx = 0; jidx < wEIoutidx[k].size(); jidx++){
					int j = wEIoutidx[k][jidx];
					uI[j][dIs[j][k]] += wIs[j][k];
				}
			}
		}
		//weight update
		//E-LTP
		for(int i = 0; i < L; i++){
			for(int j = 0; j < M; j++){ 			
				yp[i][j] += uyp[i][j][0]; yp[i][j] = rk(yp[i][j],tpp);
				uyp[i][j].pop_front(); uyp[i][j].push_back(0.0);
			}
			for(int jidx = 0; jidx < syp[i][0].size(); jidx++){
				int j = syp[i][0][jidx]; 
				wEs[j][i] += yp[i][j]*Wp(wEs[j][i]);
				if( wEs[j][i] < wEmin ) wEs[j][i] = wEmin;
				if( wEs[j][i] > wEmax ) wEs[j][i] = wEmax;
			}
			syp[i].pop_front(); syp[i].push_back(ivec);
		}
		//E-LTD
		for(int j = 0; j < M; j++){ 
			for(int i = 0; i < L; i++){		
				yd[j][i] += uyd[j][i][0]; yd[j][i] = rk(yd[j][i],tpd);
				uyd[j][i].pop_front(); uyd[j][i].push_back(0.0);
			}
			for(int iidx = 0; iidx < syd[j][0].size(); iidx++){
				int i = syd[j][0][iidx]; 
				wEs[j][i] += -yd[j][i]*Wd(wEs[j][i]);
				if( wEs[j][i] < wEmin ) wEs[j][i] = wEmin;
				if( wEs[j][i] > wEmax ) wEs[j][i] = wEmax;
			}
			syd[j].pop_front(); syd[j].push_back(ivec);
		}
		
		if( ((int)floor(t/h))%100 == 0 ){//measurement
			if( ((int)floor(t/h))%(10*1000*100) == 0 ){
				for(int j = 0; j < M; j++){
					for(int i = 0; i < L; i++) ofsw << wEs[j][i] << " ";
					for(int k = 0; k < N; k++) ofsw << wLs[k][j] << " ";
					for(int k = 0; k < N; k++) ofsw << wIs[j][k] << " ";
					ofsw << endl;
				}
				if( t < 9*1000.0 || t > 2989*1000.0 ){
					rtof= true;
				}else{
					rtof = false;
				}
				cout << t/1000.0 << endl;
			}
		}
	}
}

void simul(double qA, double qB, double wIo, int kg, double ttheta, int ik){
	calc(qA, qB, wIo, kg, ttheta, ik);
}

int main(int argc, char **argv){
	double qA = 0.0; double qB = 0.0; double wIo = 0.0; 
	int kg = 0; double ttheta = 0.0; int ik = 0;
	if(argc > 1) qA = atof(argv[1]);
	if(argc > 2) qB = atof(argv[2]);
	if(argc > 3) wIo = atof(argv[3]);
	if(argc > 4) kg = atoi(argv[4]);
	if(argc > 5) ttheta = atof(argv[5]);
	if(argc > 6) ik = atoi(argv[6]);
	srand((unsigned int)time(NULL));
	simul(qA,qB,wIo,kg,ttheta,ik);
	return 0;
}
