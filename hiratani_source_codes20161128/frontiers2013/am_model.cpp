// Associative memory model with long-tail-distributed Hebbian synaptic connections
//
// Created by Naoki Hiratani (N.Hiratani@gmail.com)
//

#include <iostream>
#include <vector>
#include <string>
#include <deque>
#include <map>
#include <algorithm>
#include <cmath>
#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <sstream>

using namespace std;
double const pi = 3.14159265;

//////////////
//parameters//
//////////////
int const NE = 10000;
int const NI = 2000;
int N = NE + NI;

//all time consts have [msec] unit
double const T = 500.0;
double const h = 0.01; // time step
double const tmE = 20.0; // leak const of v
double const tmI = 10.0;
double const ts = 2.0; // leak const of g
double const tdoE = 2.0; // time delay
double const tdoI = 1.0;
double const tref = 1.0; // refractory period

double const tdEmax = 3.0; //maxium delay of Excitatory input
double const tdImax = 2.0; //maxium delay of Inhibitary input
int const dlylen = (int)floor(tdEmax/h + 0.5);

//[mV]
double const Vth = -50.0; // threshold
double const VL = -70.0; // reversal potential of leak current
double const VE = -0.0; // reversal potential of excitatory postsynaptic current
double const VI = -80.0;
double const Va = 0.1; // failing potential

double const GEI = 0.017; //Excitatory to Inhibitory (0.018)
double const GIE = 0.0018;//(0.002)
double const GII = 0.0025;//(0.002)(0.0025)

double cE = 0.1; // sparseness of conectivity
double cI = 0.5;

double const sigma = 1.0;
double const mu = log(0.2) + sigma*sigma;

//init
double const tinit = 100.0; //[ms]
double const frinit = 10.0; //[Hz]
double const rinit = frinit*0.001*h;
double const Vinit = 10.0; //[mV]

//GtoEPSP, EPSPtoG
double Vmax = 20.0;
double dv = 0.001;

double dpks = 0.001;
double Vul = 3.0; //upper limit
double Vll = 1.0; //lower limit

//pattern retrieval
double const tprs = 200.0; // time when pattern retrieval starts
double const tprf = 210.0;
double const frpr = 100.0; // firing rate of pattern retrieval input
double const rrpr = frpr*0.001*h;

///////////////////
//basic functions//
///////////////////

double dice(){
	return rand()/(RAND_MAX + 1.0);
}

double ngn(){ //normal gaussian noise
	double u = dice();
	double v = dice();
	return sqrt(-2.0*log(u))*cos(2.0*pi*v);
}

double pngn(double tmpx, double tmpsigma){
	return 0.5*( 1.0 + erf( tmpx/(sqrt(2.0)*tmpsigma) ) );
}

double ngnl(){ //normal gaussian noise
	double ngnltmp = ngn();
	while( ngnltmp < -6.0 || 6.0 < ngnltmp ){
		ngnltmp = ngn();
	}
	return ngnltmp;
}

double VlogN(){ // log-normal variable with above limit Vmax
	double Vtmp = exp(mu+sigma*ngn());
	while( Vtmp > Vmax ){
		Vtmp = exp(mu+sigma*ngn());
	}
	return Vtmp;
}

double erf_func(double vtmp){
	return erf( (log(vtmp) - mu)/( sqrt(2.0)*sigma ) );
}

vector<int> rnd_sample(int Nar, int Nr){ // choose Nr*ar components from an array (0,1,...,Nr-1)
	vector<int> x;
	vector<int> y;
	for(int i = 0; i < Nr; i++) y.push_back(0.0);
	int tmp = 0;
	int z = 0;
	while(tmp < Nar){
		z = int(floor(dice()*Nr));
		if(y[z] == 0){
			x.push_back(z); y[z] = 1; tmp++;
		}
	}
	return x;
}

double poisson( double x ){
	if( dice() < x ){
		return 1.0;
	}else{
		return 0.0;
	}
}

template<class T> struct index_cmp { //index sort
	index_cmp(const T arr) : arr(arr) {}
	bool operator()(const size_t a, const size_t b) const{ 
		return arr[a] < arr[b]; 
	}
	const T arr;
};

vector<int> isort(vector<double> a){
	vector<size_t> b;
	for (unsigned i = 0; i < a.size(); ++i) b.push_back(i);
	sort(b.begin(), b.end(), index_cmp<vector<double>&>(a));
	vector<int> bidx;
	for (int i = 0; i < a.size(); ++i) bidx.push_back(b[i]);
	return bidx;
}

double erf_inv(double y){
	double xmax = 100.0;
	double xmin = -100.0;
	double xmid = 0.5*(xmax + xmin);
	while( abs(y - erf(xmid)) > dpks*0.01 ){
		if( y > erf(xmid) ){
			xmin = xmid;
		}else{
			xmax = xmid;
		}
		xmid = 0.5*(xmax + xmin);
	}
	return xmid;
}

double nCka(int n, int k, double a){
	if( k == 0){
		return 1.0;
	}else if( k > n/2 ){
		return pow(a,2*k-n)*nCka(n,n-k,a);
	}else{
		double s = 1.0;
		for(int l = 0; l < k; l++) s *= (n-l)*a/( (double)(k-l) );
		return s;
	}
}

//////////////////////
//specific functions//
//////////////////////

double tdrnd(double tdo){ //calc delay
	double tmp = dice()*2.0-1.0;
	return tdo + tmp;
}

double GtoE( double g ){ //synapitic weight to EPSP
	double v = VL;
	double gE = g;
	double dVmax = 0.0;
	for(double t = 0.0; t < 10.0; t += h){
		gE -= h*gE/ts;
		v -= h*( (v-VL)/tmE + gE*(v-VE) );
		if( v-VL > dVmax) dVmax = v-VL;
	}
	return dVmax;
}

double calc_gmax(){
	double g = 1.0;
	while( GtoE(g) < Vmax ){
		g *= 2.0;
	}
	return g;
}

double EtoG( double epsp , double gmax){ // EPSP to synaptic weight
	double gtmax = gmax;	
	double gtmin = 0.0;
	double gtmp = 0.0;
	while( GtoE(gtmax) - GtoE(gtmin) > dv ){
		gtmp = (gtmin + gtmax)*0.5;
		if( epsp > GtoE(gtmp) ){
			gtmin = gtmp;
		}else{
			gtmax = gtmp;
		}
	}
	return (gtmin + gtmax)*0.5;
}

vector<int> calc_pi(vector< vector<int> > ptn){
	vector<int> pi;
	for(int i = 0; i < NE; i++) pi.push_back(0);
	for(int q = 0; q < ptn.size(); q++){
		for(int qi = 0; qi < ptn[q].size(); qi++) pi[ ptn[q][qi] ] += 1;
	}
	return pi;
}

vector<double> calc_Ztj( vector<int> pi, double pa ){
	vector<double> Ztj; double Ztjsum = 0.0;
	for(int j = 0; j < NE; j++){
		Ztj.push_back( exp( pi[j]/pa - 1.0 ) ); Ztjsum += Ztj[j];
	}
	Ztjsum = Ztjsum/((double)NE);
	for(int j = 0; j < NE; j++){
		Ztj[j] = Ztj[j]/Ztjsum;
	}

	return Ztj;
}

int select_k( vector<int> giwksyn, int i ){
	int gilen = giwksyn.size();
	int ktmp = giwksyn[ (int)floor( gilen*dice() ) ];
	return ktmp;
}		

bool burst_detector( deque< vector<int> > spkout, vector< vector<int> > ptn, double t ,int NEa, int rtp){
	int p = ptn.size();
	bool tof =false;
	vector<int> ptnde_inv;
	double sqcnt = 0.0;
	for(int ttmp = 0; ttmp < spkout.size(); ttmp++){
		for(int iidx = 0; iidx < spkout[ttmp].size(); iidx++){
			if( spkout[ttmp][iidx] < NE ) sqcnt += 1.0/(2.0*NEa);
		}
	}

	if( ( (int)floor(t*100.0) )%1000==0 && sqcnt > 0.01*NE){
		for(int q = 0; q < p; q++){
			ptnde_inv.clear();
			sqcnt = 0.0;
			for(int i = 0; i < N; i++) ptnde_inv.push_back(-1);
			for(int i = 0; i < ptn[q].size(); i++) ptnde_inv[ ptn[q][i] ] = 1;
			for(int ttmp = 0; ttmp < spkout.size(); ttmp++){
				for(int iidx = 0; iidx < spkout[ttmp].size(); iidx++){
					if( ptnde_inv[ spkout[ttmp][iidx] ] > 0 ) sqcnt += 1.0/(2.0*NEa);
				}
			}
			if( (q != rtp && sqcnt > 0.1*NEa) || (q == rtp && sqcnt > 0.2*NEa && t > tprs + 100.0) 
				|| (q == rtp && sqcnt > 0.1*NEa && t < tprs) ){
				tof = true;
				cout << "SB : " << q << endl;
			}
		}
	}
	
	return tof;
}


void calc(vector< vector<double> > G, vector< vector<double> > d, vector< vector<int> > ptn, int NEa, int p, int k, int rtp){
	vector<double> st; //spike train
	vector<double> v; //membrene potential
	vector<double> gE;
	vector<double> gI;
	for(int i = 0; i < N; i++){//initialization
		st.push_back(-T);
		v.push_back(VL);
		gE.push_back(0.0); gI.push_back(0.0);
	}

	vector<int> ptnin_inv;
	for(int i = 0; i < N; i++) ptnin_inv.push_back(-1);
	for(int i = 0; i < ptn[rtp].size(); i++) ptnin_inv[ ptn[rtp][i] ] = 1;

	ostringstream ossr; ossr << "am_model_r_NEa" << NEa << "_p" << p << "_k" << k << "_r" << rtp << ".txt";
	string fstrr = ossr.str(); ofstream ofsr; ofsr.open( fstrr.c_str() );
	ostringstream ossv; ossv << "am_model_v_NEa" << NEa << "_p" << p << "_k" << k << "_r" << rtp << ".txt";
	string fstrv = ossv.str(); ofstream ofsv; ofsv.open( fstrv.c_str() );
	
	double gmax = calc_gmax();
	double ginit = EtoG(Vinit, gmax);
	double Ga = EtoG(Va, gmax);

	vector<int> ivec;
	deque< vector<int> > spkin,spkout;
	vector<int> spktmp;
	for(int ttmp = 0; ttmp < dlylen; ttmp++){
		spkin.push_back(ivec); spkout.push_back(ivec);
	}

	vector<double> inputE; vector<double> inputI;

	for(double t = 0; t < T; t += h){
		for(int i = 0; i < N; i++){		
			inputE.push_back(0.0); inputI.push_back(0.0);
		}
		for(int iidx = 0; iidx < spkin[0].size(); iidx++){
			if( spkout[0][iidx] < NE ){
				inputE[ spkin[0][iidx] ] += G[ spkin[0][iidx] ][ spkout[0][iidx] ];
			}else{
				inputI[ spkin[0][iidx] ] += G[ spkin[0][iidx] ][ spkout[0][iidx] ];
			}
		}
		for(int i = 0; i < N; i++){
			if( t < tinit && dice() < rinit ) inputE[i] += ginit;
			if( dice() < rrpr && ptnin_inv[i] > 0 && tprs < t && t < tprf ) inputE[i] += ginit;
		}

		for(int i = 0; i < N; i++){
			gE[i] += h*(-gE[i]/ts) + inputE[i];
			gI[i] += h*(-gI[i]/ts) + inputI[i];
		}
	
		for(int i = 0; i < NE; i++){
			if(t-st[i] < tref){
				v[i] = VL;
			}else{
				v[i] -= h*( (v[i]-VL)/tmE + gE[i]*(v[i]-VE) + gI[i]*(v[i]-VI) );
			}
		}
		for(int i = NE; i < N; i++)	{
			if(t-st[i] < tref){
				v[i] = VL;
			}else{
				v[i] -= h*( (v[i]-VL)/tmI + gE[i]*(v[i]-VE) + gI[i]*(v[i]-VI) );
			}
		}

		for(int i = 0; i < N; i++){
			if(v[i] > Vth){
				spktmp.push_back(i); st[i] = t; v[i] = VL; ofsr << t << " " << i << endl;
			}
		}

		int idt;
		spkin.pop_front(); spkin.push_back(ivec);
		spkout.pop_front(); spkout.push_back(ivec);
		for(int jidx = 0; jidx < spktmp.size(); jidx++){
			int j = spktmp[jidx];
			for(int i = 0; i < N; i++){
				if( (i < NE && j < NE && dice() > Ga/(Ga+G[i][j]) ) || (i >=NE || j >= NE) ){
					idt = (int)floor(d[i][j]/h + 0.5);
					if(idt > dlylen -1) idt = dlylen - 1;
					if(idt < 1) idt = 1;
					spkin[idt].push_back(i); spkout[idt].push_back(j);
				}
			}
		}
		
		if( ((int)floor(t*100.0))%1000==0 ){
			cout << t << endl;
			ofsv << t;
			for(int i = 0; i < NE; i++ ) ofsv << " " << v[i];
			ofsv << endl;
			if( burst_detector(spkout, ptn, t, NEa, rtp) ) break;
		}
		
		inputE.clear(); inputI.clear(); spktmp.clear();
	}

}

//Rewire excitatory connections in a way that associative memories are represented by weights
vector< vector<double> > recompose(vector< vector<double> > G, vector< vector<int> > ptn, int NEa, int p, int k){
	double a = NEa/( (double)NE );

	vector<double> dvec;
	vector< vector<double> > T; //Synapse weight mat from patterns
	for(int i = 0; i < NE; i++){
		T.push_back(dvec);
		for(int j = 0; j < NE; j++) T[i].push_back( 0.0 );
	}

	for(int ptmp = 0; ptmp < p; ptmp++){		
		for(int pi = 0; pi < ptn[ptmp].size(); pi++){
			for(int pj = 0; pj < ptn[ptmp].size(); pj++){
				if(pi != pj) T[ ptn[ptmp][pi] ][ ptn[ptmp][pj] ] += 1.0;
			}
		}
	}
	double Zv = 0.5*(1.0 + erf_func(Vmax));
	cout << Zv << endl;

	vector<int> pi = calc_pi(ptn);
	int pimax = 0;
	for(int i = 0; i < NE; i++){
		if( pi[i] > pimax ) pimax = pi[i];
	}
	vector<double> Ztj = calc_Ztj(pi,p*a);
	cout << pimax << endl;

	double dhtmp = 0.0001;
	vector< vector<double> > Hqs; vector< vector<double> > dHqs; 
	for(int q = 0; q <= pimax; q++){
		dHqs.push_back(dvec); 
		Hqs.push_back(dvec); Hqs[q].push_back(0.0);
		for(int qi = 0; qi <= q; qi++){
			dHqs[q].push_back( nCka(q,qi,a)*pow(1.0-a,q-qi) ); Hqs[q].push_back(Hqs[q][qi] + dHqs[q][qi]);
			cout << q << " " << qi << " " << dHqs[q][qi] << " " << Hqs[q][qi+1] << endl;
			if( Hqs[q][qi+1] > 1.0 - dhtmp ) break;
		}
	}

	vector<double> HVarray; 
	for(double htmp = dhtmp; htmp < 1.0000; htmp += dhtmp){
		HVarray.push_back( exp( mu + sqrt(2.0)*sigma*erf_inv(2.0*htmp - 1.0) ) );
	}

	double gmax = calc_gmax(); vector<double> EGarray;
	for(double v = 0.0; v < Vmax; v += dv) EGarray.push_back( EtoG(v,gmax) );

	int excnt = 0; int dhcnt = 0;
	double vtmp,htmp; int qx;
	for(int i = 0; i < NE; i++){
		for(int j = 0; j < NE; j++){
			if( dice() < cE && i != j){
				qx = T[i][j]; T[i][j] += dice();  
				htmp = Hqs[ pi[i] ][qx] + dHqs[ pi[i] ][qx]*(T[i][j] - qx);
				//if( dice() < 0.001 ) cout << i << " " << j << " " << T[i][j] << " " << htmp << endl;
				if(htmp > 1.0 - dhtmp){
					htmp = 1.0 - dhtmp; dhcnt++;
				}
				vtmp = HVarray[ (int)floor(Zv*htmp/dhtmp) ]/Ztj[j];
				if( vtmp < Vmax ){
					G[i][j] = EGarray[(int)floor(vtmp/dv)];
				}else{
					G[i][j] = EGarray[(int)floor(VlogN()/dv)]; excnt++;
				}
			}
		}
	}
	cout << excnt << " " << dhcnt << endl;
	
	vector<int> ivec;
	vector< vector<int> > gistsyn; vector< vector<int> > giwksyn;
	double Gul = EGarray[(int)floor(Vul/dv)]; double Gll = EGarray[(int)floor(Vll/dv)];

	for(int i = 0; i < NE; i++){
		gistsyn.push_back( ivec ); giwksyn.push_back( ivec );
		for(int j = 0; j < NE; j++){
			if(G[i][j] > Gul) gistsyn[i].push_back( j );
			if(0.0 < G[i][j] && G[i][j] < Gll) giwksyn[i].push_back( j );
		}
	}

    //rewire strong reciprocal connections
	double gtmp; int ktmp;
	for(int i = 0; i < NE; i++){
		for(int jidx = 0; jidx < gistsyn[i].size(); jidx++){
			int j = gistsyn[i][jidx];
			if( i < j ){
				for(int kidx = 0; kidx < gistsyn[j].size(); kidx++){
					if( i == gistsyn[j][kidx] ){
						if(dice() < 0.5 ){
							ktmp = select_k(giwksyn[i], i);
							gtmp = G[i][j]; G[i][j] = G[i][ktmp]; G[i][ktmp] = gtmp;
						}else{
							ktmp = select_k(giwksyn[j], j);
							gtmp = G[j][i]; G[j][i] = G[j][ktmp]; G[j][ktmp] = gtmp;
						}
					}
				}
			}
		}
	}

	gistsyn.clear(); giwksyn.clear();
	for(int i = 0; i < NE; i++){
		gistsyn.push_back( ivec ); giwksyn.push_back( ivec );
		for(int j = 0; j < NE; j++){
			if(G[i][j] > Gul) gistsyn[i].push_back( j );
			if(0.0 < G[i][j] && G[i][j] < Gll) giwksyn[i].push_back( j );
		}
	}

    //rewire strong cyclic connections
	for(int i = 0; i < NE; i++){
		for(int jidx = 0; jidx < gistsyn[i].size(); jidx++){
			int j = gistsyn[i][jidx];
			if( i < j ){
				for(int kidx = 0; kidx < gistsyn[j].size(); kidx++ ){
					int k = gistsyn[j][kidx];
					if( j < k ){
						for(int iidx = 0; iidx < gistsyn[k].size(); iidx++){
							if( i == gistsyn[k][iidx] ){
								if( dice() < 0.33 ){
									ktmp = select_k(giwksyn[i], i);
									gtmp = G[i][j]; G[i][j] = G[i][ktmp]; G[i][ktmp] = gtmp;
								}else if( dice() < 0.33 ){
									ktmp = select_k(giwksyn[j], j);
									gtmp = G[j][k]; G[j][k] = G[j][ktmp]; G[j][ktmp] = gtmp;
								}else{
									ktmp = select_k(giwksyn[k], k);
									gtmp = G[k][i]; G[k][i] = G[k][ktmp]; G[k][ktmp] = gtmp;
								}
							}
						}
					}
				}
			}
		}
	}
	
	if(k == 0){
		ostringstream ossw; ossw << "lnammw_NEa" << NEa << "_p" << p << "_k" << k << ".txt"; 
		string fstrw = ossw.str(); ofstream ofsw; ofsw.open( fstrw.c_str() );
		for(int i = 0; i < NE; i++){
			for(int j = 0; j < NE; j++){
				ofsw << G[i][j] << " ";
			}
			ofsw << endl;
		}
	}

	return G;
}

vector< vector<int> > create_ptn(int NEa, int p, int k){
	ostringstream ossp; ossp << "lnammp_NEa" << NEa << "_p" << p  << "_k" << k << ".txt"; 
	string fstrp = ossp.str(); ofstream ofsp; ofsp.open( fstrp.c_str() );

	double a = NEa/( (double)NE );
	int qtmp;
	vector< vector<int> > ptn; vector<int> ptnidx;
	vector<int> ivec;
	for(int q = 0; q < p; q++){
		ptn.push_back( ivec );
		ptnidx = rnd_sample(NEa,NE);
		for(int i = 0; i < NEa; i++){
			ptn[q].push_back( ptnidx[i] );
		}
	}

	for(int q = 0; q < p; q++){
		for(int i = 0; i < ptn[q].size(); i++) ofsp << ptn[q][i] << " ";
		ofsp << endl;
	}
	
	return ptn;
}

void simul(int NEa, int p){ //model setting
	double gmax = calc_gmax();
	vector<double> EGarray;
	for(double v = 0.0; v < Vmax; v += dv){
		EGarray.push_back( EtoG(v,gmax) );
	}

	vector< vector<double> > G; //synaptic weight matrix
	vector<double> vec;
	for(int i = 0; i < NE; i++){
		G.push_back( vec );
		for(int j = 0; j < NE; j++){
			G[i].push_back(0.0);
		}
		for(int j = NE; j < N; j++){
			G[i].push_back(0.0);
			if(dice() < cI) G[i][j] = GIE;
		}
	}
	for(int i = NE; i < N; i++){
		G.push_back( vec );
		for(int j = 0; j < NE; j++){
			G[i].push_back(0.0);
			if(dice() < cE) G[i][j] = GEI;
		}
		for(int j = NE; j < N; j++){
			G[i].push_back(0.0);
			if(dice() < cI) G[i][j] = GII;
		}
	}
	
	vector< vector<double> > d; //delay matrix
	for(int i = 0; i < NE; i++){
		d.push_back( vec );
		for(int j = 0; j < NE; j++) d[i].push_back( tdrnd(tdoE) );
		for(int j = NE; j < N; j++) d[i].push_back( tdrnd(tdoI) );
	}
	for(int i = NE; i < N; i++){
		d.push_back( vec );
		for(int j = 0; j < N; j++) d[i].push_back( tdrnd(tdoI) );
	}
	
	vector< vector<double> > Gp;
	for(int i = 0; i < N; i++) Gp.push_back( vec );
	vector< vector<int> > ptn;

	for(int k = 0; k < 1; k++){
		ptn = create_ptn(NEa,p,k);
		Gp = recompose(G,ptn,NEa,p,k);
		for(int rtp = 0; rtp < 5; rtp++){
			calc(Gp,d,ptn,NEa,p,k,rtp);
		}
	}
}

int main(int argc, char **argv){
	int NEa = 0; int p = 0;
	if(argc > 1) NEa = atoi(argv[1]);
	if(argc > 2) p = atoi(argv[2]);

	srand((unsigned int)time(NULL));
	simul(NEa, p);

	return 0;
}
