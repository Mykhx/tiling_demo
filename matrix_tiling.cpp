#include <iostream>
#include <stdlib.h>
#include <random>
#include <chrono>
#include <iomanip>
#include <string.h>
#include <omp.h>

using namespace std;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

// bounds for matrix values
constexpr int MAX =  100 ;
constexpr int MIN = -100 ;

void extrem(double d, double& rmin, double& rmax, double& rabsmin){

	if (d < rmin) rmin = d;
	if (abs(d) < rabsmin) rabsmin = abs(d);
	if (d > rmax) rmax = d;

}

void __attribute__((optimize("O0"))) mulabcO0(double* a, double* b, double* c ,int m, int n){

	auto ts1 = high_resolution_clock::now();
	int i,j,k,x,y,z;
	for(i=0; i<m; i++ ){
		for(j = 0; j<m; j++ ){
			for(k=0; k<n; k++ ){
				c[ i * m + j ] += a[ i * m + k ] * b[ k * m + j ];
			}
		}
	}

	auto ts2 = high_resolution_clock::now();
	duration<double, std::milli> ms_double = ts2 - ts1 ;
	cout << "Total execution time is " << ms_double.count() << " ms " << endl;
}

// take better advantage of locality
void __attribute__((optimize("O0"))) mulabcO0ikj(double* a, double* b, double* c ,int m, int n){

	auto ts1 = high_resolution_clock::now();

	int i,j,k,x,y,z;
	for(i=0; i<m; i++ ){
		for(k=0; k<n; k++ ){
			for(j = 0; j<m; j++ ){
				c[ i * m + j ] += a[ i * m + k ] * b[ k * m + j ];
			}
		}
	}

	auto ts2 = high_resolution_clock::now();
	duration<double, std::milli> ms_double = ts2 - ts1 ;
	cout << "Total execution time is " << ms_double.count() << " ms " << endl;
}

void mulabc(double* a, double* b, double* c ,int m, int n){

	auto ts1 = high_resolution_clock::now();

	int i,j,k,x,y,z;
	for(i=0; i<m; i++ ){
		for(j = 0; j<m; j++ ){
			for(k=0; k<n; k++ ){
				c[ i * m + j ] += a[ i * m + k ] * b[ k * m + j ];
			}
		}
	}

	auto ts2 = high_resolution_clock::now();
	duration<double, std::milli> ms_double = ts2 - ts1 ;
	cout << "Total execution time is " << ms_double.count() << " ms " << endl;
}

void tilingomp1(double* a, double* b, double* c ,int m, int n, int itile, int jtile, int ktile){

	int kmin = 0;
	int jmin = 0;
	int imin = 0;

	auto ts1 = high_resolution_clock::now();

	int i,j,k,x,y,z;
#pragma parallel omp for
	{
	for(i=0; i<m; i += itile ){
		imin = std::min(i+itile,n);
		for(j = 0; j<m; j+= jtile ){
			jmin = std::min(j+jtile,m);
			for(k=0; k<n; k+= ktile ){
				kmin = std::min(k+ktile,n);
				for(x = i; x < imin; x++){
					for(y = j; y < jmin; y++){
						for(z = k; z < kmin ; z++){
							c[ x * m + y ] += a[ x * m + z ] * b[ z * m + y ];
							// this will not work due to the race condition
						}
					}
				}
			}
		}
	}
	}
	auto ts2 = high_resolution_clock::now();
	duration<double, std::milli> ms_double = ts2 - ts1 ;
	cout << "Total execution time is " << ms_double.count() << " ms " << endl;
}

void tiling(double* a, double* b, double* c ,int m, int n, int itile, int jtile, int ktile){

	int kmin = 0;
	int jmin = 0;
	int imin = 0;

	auto ts1 = high_resolution_clock::now();

	int i,j,k,x,y,z;
	for(i=0; i<m; i += itile ){
		imin = std::min(i+itile,n);
		for(j = 0; j<m; j+= jtile ){
			jmin = std::min(j+jtile,m);
			for(k=0; k<n; k+= ktile ){
				kmin = std::min(k+ktile,n);
				for(x = i; x < imin; x++){
					for(y = j; y < jmin; y++){
						for(z = k; z < kmin ; z++){
							c[ x * m + y ] += a[ x * m + z ] * b[ z * m + y ];
						}
					}
				}
			}
		}
	}

	auto ts2 = high_resolution_clock::now();
	duration<double, std::milli> ms_double = ts2 - ts1 ;
	cout << "Total execution time is " << ms_double.count() << " ms " << endl;
}

void tiling0(double* a, double* b, double* c ,int m, int n, int itile, int jtile, int ktile){

	int kmin = 0;
	int jmin = 0;
	int imin = 0;

	auto ts1 = high_resolution_clock::now();

	int i,j,k,x,y,z;
	for(i=0; i<m; i += itile ){
		//imin = std::min(i+itile,n);
		for(j = 0; j<m; j+= jtile ){
			//jmin = std::min(j+jtile,m);
			for(k=0; k<n; k+= ktile ){
				//kmin = std::min(k+ktile,n);
				for(x = i; x < std::min(i+itile,n); x++){
					for(y = j; y < std::min(j+jtile,m); y++){
						for(z = k; z < std::min(k+ktile,n) ; z++){
							c[ x * m + y ] += a[ x * m + z ] * b[ z * m + y ];
						}
					}
				}
			}
		}
	}

	auto ts2 = high_resolution_clock::now();
	duration<double, std::milli> ms_double = ts2 - ts1 ;
	cout << "Total execution time is " << ms_double.count() << " ms " << endl;
}



int main()
{

	cout << "Number of available threads: " << omp_get_num_threads() << endl;
#pragma omp parallel
	{
		usleep(5000 *  omp_get_thread_num()) ; // avoid race condition
		cout << "Number of available threads: " << omp_get_num_threads() << endl;
	}
	
	
	auto t1 = high_resolution_clock::now();
	cout << "Hello" << endl;

	int n = 1000 ;
	int m = 1000 ;
	cout << "n "<< n << endl;
	cout << "m "<< m << endl;


	srand(time(nullptr));

	double *a = new double[m*n] ;
	double rmin = (double) MAX ;
	double rmax = (double) MIN ;
	double rabsmin = (double) MAX ;

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			
	                a[i*n+j] =  ( MIN + static_cast <double> (rand()) / ( static_cast <float> (RAND_MAX/(MAX-MIN))) );
			extrem(a[i*m,j],rmin,rmax,rabsmin);
		}
		
	}


	cout << "\nMatrix A" << endl;
	cout << "Max " <<  rmax << endl;
	cout << "Min " <<  rmin << endl;
	cout << "Min(abs) " <<  rabsmin << endl;

	double *b = new double[n*m] ;
	rmin = (double) MAX ;
	rmax = (double) MIN ;
	rabsmin = (double) MAX ;

	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			
	                b[i*n+j] =  ( MIN + static_cast <double> (rand()) / ( static_cast <float> (RAND_MAX/(MAX-MIN))) );
			extrem(b[i*n+j],rmin,rmax,rabsmin);
		}
		
	}

	cout << "\nMatrix B" << endl;
	cout << "Max " <<  rmax << endl;
	cout << "Min " <<  rmin << endl;
	cout << "Min(abs) " <<  rabsmin << endl;


	double *c = new double[m*m]() ;
	cout << "\nMatrix C" << endl;

	/////////////////////////////////////////////////////
	//
	// Set up matrixes ...
	//
	// Let's do some math ...
	//
	/////////////////////////////////////////////////////

	/////////////////////////////////////////////////////
	// Primitiv 1
	/////////////////////////////////////////////////////
	auto ts1 = high_resolution_clock::now();
	for(int i = 0; i<m; i++){
		for(int j=0; j<m; j++){
			for(int k=0; k<n; k++){
				c[i,j] += a[i,k] * b[k,j];
			}
		}
	}

	auto ts2 = high_resolution_clock::now();
	duration<double, std::milli> ms_double = ts2 - ts1 ;

	memset(c, 0, m*sizeof(c));

	memset(c, 0, m*sizeof(c));
	cout << "Ref(O0)001 001 001" ;
        mulabcO0(a,b,c,m,n);

	memset(c, 0, m*sizeof(c));
	cout << "ikj(O0)001 001 001" ;
        mulabcO0ikj(a,b,c,m,n);

	memset(c, 0, m*sizeof(c));
	cout << "Naive  001 001 001" ;
        mulabc(a,b,c,m,n);

	memset(c, 0, m*sizeof(c));
	cout << "Tiling 001 001 001" ;
        tiling(a,b,c,m,n,1,1,1);

	memset(c, 0, m*sizeof(c));
	cout << "Tiling 002 001 001" ;
        tiling(a,b,c,m,n,2,1,1);

	memset(c, 0, m*sizeof(c));
	cout << "Tiling 002 002 001" ;
        tiling(a,b,c,m,n,2,2,1);

	memset(c, 0, m*sizeof(c));
	cout << "Tiling 001 002 002" ;
        tiling(a,b,c,m,n,1,2,2);

	memset(c, 0, m*sizeof(c));
	cout << "Tiling 002 002 002" ;
        tiling(a,b,c,m,n,2,2,2);

	memset(c, 0, m*sizeof(c));
	cout << "Tiling 064 001 001" ;
        tiling(a,b,c,m,n,64,1,1);

	memset(c, 0, m*sizeof(c));
	cout << "Tiling 064 064 001" ;
        tiling(a,b,c,m,n,64,64,1);

	memset(c, 0, m*sizeof(c));
	cout << "Tiling 001 064 064" ;
        tiling(a,b,c,m,n,1,64,64);

	memset(c, 0, m*sizeof(c));
	cout << "Tiling 064 064 064" ;
        tiling(a,b,c,m,n,64,64,64);

	memset(c, 0, m*sizeof(c));
	cout << "Tiling 128 064 064" ;
        tiling(a,b,c,m,n,128,64,64);

	memset(c, 0, m*sizeof(c));
	cout << "Tiling 128 128 064" ;
        tiling(a,b,c,m,n,128,128,64);


	memset(c, 0, m*sizeof(c));
	cout << "Tiling 064 064 128" ;
        tiling(a,b,c,m,n,64,64,128);

	memset(c, 0, m*sizeof(c));
	cout << "Tiling 128 128 128" ;
        tiling(a,b,c,m,n,128,128,128);

	cout << "unopt min" << endl;

	memset(c, 0, m*sizeof(c));
	cout << "Tiling 064 064 128" ;
        tiling0(a,b,c,m,n,64,64,128);

	memset(c, 0, m*sizeof(c));
	cout << "Tiling 128 128 128" ;
        tiling0(a,b,c,m,n,128,128,128);

	/////////////////////////////////////////////////////
	cout << "Ending bye" << endl ;

	auto t2 = high_resolution_clock::now();

	ms_double = t2 - t1 ;

	cout << "Total execution time is " << ms_double.count() << " ms " << endl;

	delete [] a ;
	delete [] b ;
	delete [] c ;
}

