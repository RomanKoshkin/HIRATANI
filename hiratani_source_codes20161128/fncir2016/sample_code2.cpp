// Model of learning with synaptic weight and wiring plasticity
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

int const T =  5020000; 
int const T1 = 5000000;
int const Tk = 100000;
int const To = 1000;

int const p = 10; 
int const M = 200;

double pso = 1.0;
double muo = 1.0;

double lSsigma = 0.0;
double lXSsigma = 1.0;

double vmin = -60.0;
double vminmin = -100.0;
double dvm = 10.0;

double rXo = 1.0; //Hz
double rYo = 1.0;
double sigmainitw = 0.1;

double wXmin = 0.0001;
double rhomin = 0.0001;
double rhomax = 1.0;

int tcdfm = 1000;
double drXo = 0.01;

double vsigma = 0.0;

//double etar = 0.001; //0.0001;
double etaX = 0.01;
//double bh = 0.0;//100.0;

vector<double> dvec;
vector<int> ivec;
vector< vector<double> > ddvec;

double dice(){
	return rand()/(RAND_MAX + 1.0);
}

double ngn(){
	double u = dice(); double v = dice();
	return sqrt(-2.0*log(u))*cos(2.0*pi*v);
}

double thetah(double x){
	if( x > 0 ){
		return x;
	}else{
		return 0.0;
	}
}

int Sk_get(vector<double> Scdf){
	double rtmp = dice();
	int qidx = 0;
	for(int q = 0; q < p; q++){
		if( Scdf[q] <= rtmp && rtmp < Scdf[q+1] ) qidx = q;
	}
	return qidx;
}

vector<double> calc_pS(){
	vector<double> pS;
	double Zqtmp = 0.0; 
	for(int q = 0; q < p; q++){
		pS.push_back( pso + lSsigma*ngn() );
		Zqtmp += pS[q];
	}
	for(int q = 0; q < p; q++) pS[q] = pS[q]/Zqtmp;
	return pS;
}

vector<double> calc_Scdf(vector<double> pS){
	vector<double> Scdf; Scdf.push_back(0.0);
	for(int q = 0; q < p; q++) Scdf.push_back( Scdf[q] + pS[q] );
	for(int q = 0; q < p+1; q++) Scdf[q] = Scdf[q]/Scdf[p];
	return Scdf;
}

vector< vector<double> > calc_muss(double lXSsigma, double sigmaw){
	vector< vector<double> > muss;
	for(int j = 0; j < M; j++){
		muss.push_back( dvec );
		for(int q = 0; q < p; q++) muss[j].push_back(0.0);
	}
	double Zjtmp = 0.0;
	for(int q = 0; q < p; q++){
		Zjtmp = 0.0;
		for(int j = 0; j < M; j++){
			muss[j][q] = muo + lXSsigma*ngn();
			while( muss[j][q] < 0.0 ) muss[j][q] = muo + lXSsigma*ngn();
			Zjtmp += muss[j][q]*muss[j][q];
		}
		for(int j = 0; j < M; j++) muss[j][q] = sqrt(M/Zjtmp)*rXo*muss[j][q];
	}

	return muss;
}

vector< vector<double> > calc_rho(vector< vector<double> > muss, double gm,  double sigmaw, int Ly){
	int Ny = p*Ly;
	double mumean = 0.0;
	for(int q = 0; q < p; q++){
		for(int j = 0; j < M; j++) mumean += muss[j][q]/((double)(p*M));
	}

	vector< vector<double> > Qss;
	for(int q = 0; q < p; q++){
		Qss.push_back(dvec);
		for(int j = 0; j < M; j++){
			//Qss[q].push_back( muss[j][q]*lambda + mumean*(1.0-lambda) );
			Qss[q].push_back( mumean );
		}
	}
	
	vector< vector<double> > rho; double rhomean = 0.0;
	for(int i = 0; i < Ny; i++){
		rho.push_back(dvec);
		for(int j = 0; j < M; j++){
			rho[i].push_back( min( 1.0, gm*Qss[i/Ly][j]/(sigmaw*sigmaw) ) );
			rhomean += rho[i][j]/((double)(M*Ny));
		}
	}
	cout << "rhomean : " << rhomean << endl;
	return rho;		
}

vector< vector<double> > calc_Cnc(vector< vector<double> > rho, int Ly){
	vector< vector<double> > Cnc;
	int cnccnt = 0;
	for(int i = 0; i < p*Ly; i++){
		Cnc.push_back(dvec);
		for(int j = 0; j < M; j++){
			Cnc[i].push_back(0.0);
			if( dice() < rho[i][j] ){
				Cnc[i][j] = 1.0; cnccnt += 1;
			}
		}
	}
	cout << "cnccnt : " << cnccnt << endl;
	return Cnc;
}

vector< vector<double> > calc_wX(vector< vector<double> > muss, vector< vector<double> > rho, double gm, double wXmax, double Ly, double sigmaw){
	vector< vector<double> > wX;
	for(int i = 0; i < p*Ly; i++){
		wX.push_back(dvec);
		for(int j = 0; j < M; j++){
			//wX[i].push_back( (muss[j][i/Ly]/(sigmaw*sigmaw))/(rho[i][j]) );
			wX[i].push_back( (1.0 + sigmainitw*ngn())/gm );
			if( wX[i][j] < wXmin ) wX[i][j] = wXmin;
			if( wX[i][j] > wXmax ) wX[i][j] = wXmax;
		}
	}
	
	return wX;
}

vector<int> calc_prefq(vector< vector<double> > prefs, vector<double> prefcnts, int Ny){
	vector<int> prefq;
	for(int i = 0; i < Ny; i++){
		double prefmaxtmp = -To; //0.0;
		int prefidx = 0;
		for(int q = 0; q < p; q++){
			if( prefs[i][q]/prefcnts[q] > prefmaxtmp ){
				prefmaxtmp = prefs[i][q]/prefcnts[q]; prefidx = q;
			}
		}
		prefq.push_back(prefidx);
	}
	return prefq;
}

int estimate_Sk2(vector<double> rYs, vector<int> prefq, int Ly){
	int Ny = p*Ly;
	vector<double> qlens;
	for(int q = 0; q < p; q++) qlens.push_back(0.0);
	for(int i = 0; i < Ny; i++) qlens[ prefq[i] ] += 1.0;

	vector<double> rYmeans;
	for(int q = 0; q < p; q++) rYmeans.push_back(0.0);
	for(int i = 0; i < Ny; i++){
		if( qlens[ prefq[i] ] != 0) rYmeans[ prefq[i] ] += rYs[i]/qlens[ prefq[i] ];
	}
	
	double rYmaxtmp = 0.0; int Sktmp = 0;
	for(int q = 0; q < p; q++){
		if( rYmeans[q] > rYmaxtmp ){
			rYmaxtmp = rYmeans[q]; Sktmp = q;
		}
	}
	return Sktmp;
}

double get_rX(double mutmp, double sigmaw){
	return mutmp + sigmaw*ngn();
}

int get_maxq(vector<double> perfstmp){
	int qtmp = 0; double qtmpmax = 0;
	for(int q = 0; q < p; q++){
		if( qtmpmax < perfstmp[q] ){
			qtmp = q; qtmpmax = perfstmp[q];
		}
	}
	return qtmp;
}

double calc_mumean(double sigmaw, double kappa){
	double mumean = 0.0; int mumeancnt = 100;
	vector< vector<double> > muss;
	vector< vector<double> > muAsstmp;
	vector< vector<double> > muBsstmp;
	for(int j = 0; j < M; j++){
		muss.push_back(dvec);
		for(int q = 0; q < p; q++) muss[j].push_back(0.0);
	}
	for(int i = 0; i < mumeancnt; i++){
		muAsstmp = calc_muss(lXSsigma, sigmaw);
		muBsstmp = calc_muss(lXSsigma, sigmaw);
		double Zjtmp = 0.0;
		for(int q = 0; q < p; q++){
			Zjtmp = 0.0;
			for(int j = 0; j < M; j++){
				muss[j][q] = (kappa*(muAsstmp[j][q]-rXo) + (1.0-kappa)*(muBsstmp[j][q]-rXo)) + rXo;
				Zjtmp += muss[j][q]*muss[j][q];
			}
			for(int j = 0; j < M; j++){
				muss[j][q] = sqrt(M/Zjtmp)*rXo*muss[j][q];
				mumean += muss[j][q]/((double)(p*M*mumeancnt));
			}
		}
	}
	cout << "mumean : " << mumean << endl;
	return mumean;
}

void calc(int Ly, double gm, double sigmaw, double bh, double etar, double treco, double kappa, int ik){
	int igm = (int)floor( gm*1000.0 + 0.01 );
	int isigmaw = (int)floor( sigmaw*100.0 + 0.01 );
	int ibh = (int)floor( bh*1000.0 + 0.01 );
	int ietar = (int)floor( etar*10000.0 + 0.01 );
	int itreco = (int)floor( treco + 0.01 );
	int ikappa = (int)floor( kappa*100.0 + 0.01 );

	ostringstream ossp; 
	ossp << "sample_code2p_y" << Ly << "_g" << igm << "_s" << isigmaw << "_h" << ibh << "_e" << ietar << "_t" << itreco << "_kp" << ikappa << "_k" << ik << ".txt"; 
	string fstrp = ossp.str(); ofstream ofsp; ofsp.open( fstrp.c_str() );
	ostringstream ossq; 
	ossq << "sample_code2q_y" << Ly  << "_g" << igm << "_s" << isigmaw << "_h" << ibh << "_e" << ietar << "_t" << itreco << "_kp" << ikappa << "_k" << ik << ".txt"; 
	string fstrq = ossq.str(); ofstream ofsq; ofsq.open( fstrq.c_str() );
	
	ostringstream ossrs; 
	ossrs << "sample_code2s_y" << Ly  << "_g" << igm << "_s" << isigmaw << "_h" << ibh << "_e" << ietar << "_t" << itreco << "_kp" << ikappa << "_k" << ik << ".txt"; 
	string fstrrs = ossrs.str(); ofstream ofsrs; ofsrs.open( fstrrs.c_str() );
	ostringstream ossrx; 
	ossrx << "sample_code2rx_y" << Ly  << "_g" << igm << "_s" << isigmaw << "_h" << ibh << "_e" << ietar << "_t" << itreco << "_kp" << ikappa << "_k" << ik << ".txt"; 
	string fstrrx = ossrx.str(); ofstream ofsrx; ofsrx.open( fstrrx.c_str() );
	ostringstream ossry; 
	ossry << "sample_code2ry_y" << Ly  << "_g" << igm << "_s" << isigmaw << "_h" << ibh << "_e" << ietar << "_t" << itreco << "_kp" << ikappa << "_k" << ik << ".txt"; 
	string fstrry = ossry.str(); ofstream ofsry; ofsry.open( fstrry.c_str() );
	
	double trec = treco*1000.0;
	int Ny = Ly*p; double wXmax = 100.0/gm;

	vector<double> pS = calc_pS();
	vector<double> Scdf = calc_Scdf(pS);
	
	vector< vector<double> > muAss = calc_muss(lXSsigma, sigmaw);

	for(int q = 0; q < p; q++) ofsq << pS[q] << " ";
	ofsq << endl;
	for(int q = 0; q < p; q++){
		for(int j = 0; j < M; j++) ofsq << muAss[j][q] << " ";
		ofsq << endl;
	}
	double mumean = calc_mumean(sigmaw,kappa);

	vector< vector<double> > muss;
	for(int j = 0; j < M; j++){
		muss.push_back(dvec);
		for(int q = 0; q < p; q++) muss[j].push_back(0.0);
	}
	vector< vector<double> > muBss;
	
	vector< vector<double> > rho = calc_rho(muAss,gm,sigmaw,Ly);
	vector< vector<double> > Cnc = calc_Cnc(rho,Ly);
	
	vector< vector<int> > Cncidxs;
	for(int i = 0; i < Ny; i++){
		Cncidxs.push_back( ivec );
		for(int j = 0; j < M; j++){
			if( Cnc[i][j] > 0.5 ) Cncidxs[i].push_back(j);
		}
	}

	double rhomean = 0.0;
	for(int i = 0; i < Ny; i++){
		for(int j = 0; j < M; j++) rhomean += rho[i][j]/((double)(M*Ny));
	}
	vector< vector<double> > wX = calc_wX(muAss, rho, gm, wXmax, Ly, sigmaw);
	
	vector< vector<double> > prefs;
	for(int i = 0; i < Ny; i++){
		prefs.push_back(dvec);
		for(int q = 0; q < p; q++) prefs[i].push_back(0.0);
	}
	vector<double> prefcnts;
	for(int q = 0; q < p; q++) prefcnts.push_back(0.0);
	vector<int> prefq;
	for(int i = 0; i < Ny; i++) prefq.push_back( (int)floor(p*dice()) );
	double perf2 = 0.0;

	vector<double> rXs;
	for(int j = 0; j < M; j++) rXs.push_back(0.0); 
	vector<double> rYs,vs;
	for(int i = 0; i < Ny; i++) rYs.push_back(0.0); 
	for(int i = 0; i < Ny; i++) vs.push_back(0.0);

	double ZrI = 0.0; double vmaxtmp = 0.0;
	double It = 0.0; 
	int Sk; double xsum = 0.0;
	for(int t = 0; t < T; t++){
		if(t%Tk == 0){
			muBss = calc_muss(lXSsigma, sigmaw);
			for(int q = 0; q < p; q++){
				for(int j = 0; j < M; j++) ofsq << muBss[j][q] << " ";
				ofsq << endl;
			}
			double Zjtmp = 0.0;
			for(int q = 0; q < p; q++){
				Zjtmp = 0.0;
				for(int j = 0; j < M; j++){
					muss[j][q] = (kappa*(muAss[j][q]-rXo) + (1.0-kappa)*(muBss[j][q]-rXo)) + rXo;
					Zjtmp += muss[j][q]*muss[j][q];
				}
				for(int j = 0; j < M; j++) muss[j][q] = sqrt(M/Zjtmp)*rXo*muss[j][q];
			}
		}
		Sk = Sk_get(Scdf);
		for(int i = 0; i < Ny; i++) vs[i] = 0.0;

		for(int j = 0; j < M; j++){
			rXs[j] = get_rX( muss[j][Sk], sigmaw );
		}

		ZrI = 0.0; vmaxtmp = vmin;
		for(int i = 0; i < Ny; i++){
			vs[i] = 0.0;
			for(int j = 0; j < M; j++){
				vs[i] += Cnc[i][j]*(wX[i][j]*rXs[j] - rXo/gm); 
			}
			if( vs[i] > vmaxtmp ) vmaxtmp = vs[i];
		}
		for(int i = 0; i < Ny; i++){
			if( vs[i] - vmaxtmp < vmin ) vs[i] = vmin + vmaxtmp;
		}

		for(int i = 0; i < Ny; i++) ZrI += exp(vs[i]-vmaxtmp+dvm);
		if( Ly*exp(vmin) < ZrI && ZrI < Ly*exp(dvm) ){
			It = log(ZrI) + vmaxtmp - dvm;
			for(int i = 0; i < Ny; i++){
				vs[i] = vs[i]-It; rYs[i] = rYo*exp(vs[i]);
				if( rYs[i] > rYo ) rYs[i] = rYo;
				if( rYs[i] < exp(vmin) ) rYs[i] = exp(vmin);
			}
		}else{ //numerical error catch
			for(int j = 0; j < M; j++) rXs[j] = 0.0;
			for(int i = 0; i < Ny; i++) rYs[i] = 1.0/((double)Ny);
		}

		for(int i = 0; i < Ny; i++) prefs[i][Sk] += rYs[i];
		prefcnts[Sk] += 1.0;
		if( Sk == estimate_Sk2(rYs,prefq,Ly) ) perf2 += 1.0/To;
		
		if( t < T1 ){
			for(int i = 0; i < Ny; i++){
				for(int j = 0; j < M; j++){
					if( Cnc[i][j] > 0.5 ){
						wX[i][j] += (etaX/gm)*( rYs[i]*(rXs[j] - sigmaw*sigmaw*rhomean*wX[i][j]) + bh*(rYo/((double)Ny) - rYs[i]));
						if(wX[i][j] > wXmax) wX[i][j] = wXmax;
						if(wX[i][j] < wXmin) wX[i][j] = wXmin;
					}
					
					rho[i][j] += etar*rYs[i]*(rXs[j] - sigmaw*sigmaw*rho[i][j]/gm);
					if(rho[i][j] < rhomin) rho[i][j] = rhomin;
					if(rho[i][j] > rhomax) rho[i][j] = rhomax;
					if( Cnc[i][j] < 0.5 ){
						if( dice() < rho[i][j]/trec ){
							Cnc[i][j] = 1.0; 
						}
					}else{
						if( dice() < (1.0-rho[i][j])/trec ){
							Cnc[i][j] = 0.0; wX[i][j] = (1.0 + sigmainitw*ngn())/gm;
							if(wX[i][j] > wXmax) wX[i][j] = wXmax;
							if(wX[i][j] < wXmin) wX[i][j] = wXmin;
						}
					}
					
				}
			}
		}
		if(t > T-To && ik == 0){
			ofsrs << t << " " << Sk << endl;
			for(int j = 0; j < M; j++) ofsrx << rXs[j] << " ";
			ofsrx << endl;
			for(int i = 0; i < Ny; i++){
				ofsry << rYs[i] << " ";
			}
			ofsry << endl;
		}

		if(t%To == 0){
			ofsp << t << " " << perf2 << endl;
			perf2 = 0.0;
			prefq = calc_prefq(prefs,prefcnts,Ny);
			for(int i = 0; i < Ny; i++){
				for(int q = 0; q < p; q++) prefs[i][q] = 0.0;
			}
			for(int q = 0; q < p; q++) prefcnts[q] = 0.0;
		}
	}
}

void simul(int Ly, double gm, double sigmaw, double bh, double etar, double treco, double kappa, int ik){
	calc(Ly, gm, sigmaw, bh, etar, treco, kappa, ik);
}

int main(int argc, char **argv){
	int Ly = 0; double gm = 0.0; double sigmaw = 0.0; double bh = 0.0; 
	double etar = 0.0; double treco = 0.0; double kappa = 0.0; int ik = 0;
	if(argc > 1) Ly = atoi(argv[1]);
	if(argc > 2) gm = atof(argv[2]);
	if(argc > 3) sigmaw = atof(argv[3]);
	if(argc > 4) bh = atof(argv[4]);
	if(argc > 5) etar = atof(argv[5]);
	if(argc > 6) treco = atof(argv[6]);
	if(argc > 7) kappa = atof(argv[7]);
	if(argc > 8) ik = atoi(argv[8]);
	srand((unsigned int)(time(NULL)+ik));
	simul(Ly,gm,sigmaw,bh,etar,treco,kappa,ik);
	return 0;
}
