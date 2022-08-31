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
//using std::setprecision;

// generate random matrix
constexpr int MAX =  100 ;
constexpr int MIN = -100 ;

void extrem(double d, double& rmin, double& rmax, double& rabsmin){

	if (d < rmin) rmin = d;
	if (abs(d) < rabsmin) rabsmin = abs(d);
	if (d > rmax) rmax = d;

}

void __attribute__((optimize("O0"))) mulabcO0(double* a, double* b, double* c ,int m, int n){

	// profiling
	auto ts1 = high_resolution_clock::now();

	// tiled loop
	int i,j,k,x,y,z;
	for(i=0; i<m; i++ ){
		for(j = 0; j<m; j++ ){
			for(k=0; k<n; k++ ){
				c[ i * m + j ] += a[ i * m + k ] * b[ k * m + j ];
			}
		}
	}

	// profiling
	auto ts2 = high_resolution_clock::now();
	duration<double, std::milli> ms_double = ts2 - ts1 ;
	cout << "Total execution time is " << ms_double.count() << " ms " << endl;
}

// take better advantage of locality
void __attribute__((optimize("O0"))) mulabcO0ikj(double* a, double* b, double* c ,int m, int n){

	// profiling
	auto ts1 = high_resolution_clock::now();

	// tiled loop
	int i,j,k,x,y,z;
	for(i=0; i<m; i++ ){
		for(k=0; k<n; k++ ){
			for(j = 0; j<m; j++ ){
				c[ i * m + j ] += a[ i * m + k ] * b[ k * m + j ];
			}
		}
	}

	// profiling
	auto ts2 = high_resolution_clock::now();
	duration<double, std::milli> ms_double = ts2 - ts1 ;
	cout << "Total execution time is " << ms_double.count() << " ms " << endl;
}

void mulabc(double* a, double* b, double* c ,int m, int n){

	// profiling
	auto ts1 = high_resolution_clock::now();

	// tiled loop
	int i,j,k,x,y,z;
	for(i=0; i<m; i++ ){
		for(j = 0; j<m; j++ ){
			for(k=0; k<n; k++ ){
				c[ i * m + j ] += a[ i * m + k ] * b[ k * m + j ];
			}
		}
	}

	// profiling
	auto ts2 = high_resolution_clock::now();
	duration<double, std::milli> ms_double = ts2 - ts1 ;
	cout << "Total execution time is " << ms_double.count() << " ms " << endl;
}

void tilingomp1(double* a, double* b, double* c ,int m, int n, int itile, int jtile, int ktile){

	// init
	int kmin = 0;
	int jmin = 0;
	int imin = 0;

	// profiling
	auto ts1 = high_resolution_clock::now();

	// tiled loop
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

						/*	cout << "accessing" << endl;
							cout << x*m + y << " of " << m*m << endl;
							cout << x*m + z << " of " << m*n << endl;
							cout << z*m + y << " of " << m*n << endl;

							cout << x*m + y << " c " <<  c[ x * m + y ] << endl;
							cout << x*m + z << " a " <<  a[ x * m + z ] << endl;
							cout << z*m + y << " b " <<  b[ z * m + y ] << endl;
							*/
							c[ x * m + y ] += a[ x * m + z ] * b[ z * m + y ];
							// this will not work due to the race condition
						}
					}
				}
			}
		}
	}
	}
	// profiling
	auto ts2 = high_resolution_clock::now();
	duration<double, std::milli> ms_double = ts2 - ts1 ;
	cout << "Total execution time is " << ms_double.count() << " ms " << endl;
}

void tiling(double* a, double* b, double* c ,int m, int n, int itile, int jtile, int ktile){

	// init
	int kmin = 0;
	int jmin = 0;
	int imin = 0;

	// profiling
	auto ts1 = high_resolution_clock::now();

	// tiled loop
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

						/*	cout << "accessing" << endl;
							cout << x*m + y << " of " << m*m << endl;
							cout << x*m + z << " of " << m*n << endl;
							cout << z*m + y << " of " << m*n << endl;

							cout << x*m + y << " c " <<  c[ x * m + y ] << endl;
							cout << x*m + z << " a " <<  a[ x * m + z ] << endl;
							cout << z*m + y << " b " <<  b[ z * m + y ] << endl;
							*/
							c[ x * m + y ] += a[ x * m + z ] * b[ z * m + y ];
						}
					}
				}
			}
		}
	}

	// profiling
	auto ts2 = high_resolution_clock::now();
	duration<double, std::milli> ms_double = ts2 - ts1 ;
	cout << "Total execution time is " << ms_double.count() << " ms " << endl;
}

void tiling0(double* a, double* b, double* c ,int m, int n, int itile, int jtile, int ktile){

	// init
	int kmin = 0;
	int jmin = 0;
	int imin = 0;

	// profiling
	auto ts1 = high_resolution_clock::now();

	// tiled loop
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

						/*	cout << "accessing" << endl;
							cout << x*m + y << " of " << m*m << endl;
							cout << x*m + z << " of " << m*n << endl;
							cout << z*m + y << " of " << m*n << endl;

							cout << x*m + y << " c " <<  c[ x * m + y ] << endl;
							cout << x*m + z << " a " <<  a[ x * m + z ] << endl;
							cout << z*m + y << " b " <<  b[ z * m + y ] << endl;
							*/
							c[ x * m + y ] += a[ x * m + z ] * b[ z * m + y ];
						}
					}
				}
			}
		}
	}

	// profiling
	auto ts2 = high_resolution_clock::now();
	duration<double, std::milli> ms_double = ts2 - ts1 ;
	cout << "Total execution time is " << ms_double.count() << " ms " << endl;
}



int main()
{

	// OMP info
	cout << "Number of available threads: " << omp_get_num_threads() << endl;
#pragma omp parallel
	{
		usleep(5000 *  omp_get_thread_num()) ; // avoid race condition
		cout << "Number of available threads: " << omp_get_num_threads() << endl;
	}
	
	
	// start timer
	auto t1 = high_resolution_clock::now();
	cout << "Hello" << endl;

	// size
	int n = 1000 ;
	int m = 1000 ;
	cout << "n "<< n << endl;
	cout << "m "<< m << endl;


	srand(time(nullptr));

	// setup matrix a
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

	// setup matrix b
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


	// setup matrix c
	double *c = new double[m*m]() ;
	cout << "\nMatrix C" << endl;
//	for (int i = 0; i < m; i++) {
//		for (int j = 0; j < m; j++) {
//			
//			cout << "Element " << "("<< i << ";" << j << ") " << setprecision(20) << c[i,j] << endl;
//		}
//		
//	}

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
//				cout << "c " << i << " "<< j << " " << c[i,j] << endl;
//				cout << "a " << i << " "<< k << " " << a[i,k] << endl;
//				cout << "b " << k << " "<< j << " " << b[k,j] << endl;

			}
		}
	}

	auto ts2 = high_resolution_clock::now();
	duration<double, std::milli> ms_double = ts2 - ts1 ;

	// zero matrix
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



//	for (int i = 0; i < m; i++) {
//		for (int j = 0; j < m; j++) {
//			
//			cout << "Element " << "("<< i << ";" << j << ") " << setprecision(20) << c[i,j] << endl;
//		}
//		
//	}




	/////////////////////////////////////////////////////
	cout << "Ending bye" << endl ;

	// total time
	auto t2 = high_resolution_clock::now();

	ms_double = t2 - t1 ;

	cout << "Total execution time is " << ms_double.count() << " ms " << endl;


	// free memory
	delete [] a ;
	delete [] b ;
	delete [] c ;

}

