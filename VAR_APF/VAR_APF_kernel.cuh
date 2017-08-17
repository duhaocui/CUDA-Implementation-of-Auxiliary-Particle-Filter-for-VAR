#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <math.h>

#include <curand.h>
#include <curand_kernel.h>

#define BLOCK_SIZE (128)

#define DEVICE_CAPABILITY_MAJOR (1)
#define DEVICE_CAPABILITY_MINOR (3)

#define SIZEOFX (9)
#define SIZEOFY (7)
#define SIZEOFU (2)
//>>>>>>>>>>>>>Dimension of control particle>>>>>>>>>>
#define SIZEOFC (13)
//<<<<<<<<<<<<<Dimension of control particle<<<<<<<<<<

#define pi 3.1415926535
#define in2cm 2.54

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
	printf("Error at %s:%d\n",__FILE__,__LINE__);\
	return EXIT_FAILURE;}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
	printf("Error at %s:%d\n",__FILE__,__LINE__);\
	return EXIT_FAILURE;}} while(0)

class cv_obs
{
public:
  double Time;
  double G, Xram, I, Me;
  double Scl, Smr, phe;
};

class cv_inputs
{
public:
  double Ic;
  double Vramc;
};

struct SimParams {
	//                   //
	//Furnace parameters //
	//                   //
	double De; //Electrode diameter (in)
	double Di; //Ingot diameter (in)
	double I0; //Nominal current (A)
	double G0; //Nominal gap (in)
	double phe0; //Nominal helium pressure (Torr)
	double mu0; //Nominal melting efficiency
	//                      //
	//Electrical parameters //
	//                      //
	double Vc; //Cathode voltage fall (V)
	double Ri; //Gap-independent electric resistance (Ohm)
	double Rg; //Gap-dependent electric resistance (Ohm/cm)
	//                 //
	//Ingot parameters //
	//                 //
	double Scl0; //Nominal centerline pool depth (cm)
	double Smr0; //Nominal mid-radius pool depth (cm)
	double Acl; //A matrix (1)
	double Amr; 
	double Bdeltacl; //Bdelta matrix (1)
	double Bdeltamr; 
	double Bicl; //Bi matrix (cm/A)
	double Bimr;  
	double Bmucl; //Bmu matrix (cm)
	double Bmumr; 
	double Bhecl; //Bhe matrix (cm/Torr)
	double Bhemr; 
	//                  //
	//Input noise terms //
	//                  //
	double sigmaI; //Current standard deviation (A)
	double sigmaVram; //Ram speed standard deviation (cm/s)
	//                             //
	//Process modeling noise terms //
	//                             //
	double sigmamur; //Melting efficiency standard deviation (1)
	double sigmaa; //Fill ratio standard deviation (1)
	double sigmaVb; //Voltage bias standard deviation
	double sigmaIb; //Current bias standard deviation
	double sigmaVramb; //Ram speed bias standard deviation
	double sigmahe; //Helium pressure standard deviation
	//                        //
	//Measurement noise terms //
	//                        //
	double sigmaG; //Measured electrode gap standard deviation (cm)
	double sigmaPos; //Measured ram position standard deviation (cm)
	double sigmaImeas; //Measured current standard deviation (A)
	double sigmaLC; //Measured load cell standard deviation (g)
	double sigmaVmeas; //Measured voltage standard deviation (V)
	double sigmaCL; //Measured centerline pool depth standard deviation (cm)
	double sigmaMR; //Measured mid-radius pool depth standard deviation (cm)
	double sigmahemeas; //Measured helium pressure standard deviation (Torr)
	//                      //
	//Simulation parameters //
	//                      //
	double dt; //Time step (s)
	//                        //
	// Other global variables //
	//                        //
	double alphar;
	double alpham;
	double hr;
	double rhor;
	double hm;
	double hs;
	double a0;
	double betam;
	double Cdd;
	double Cdp;
	double Csd;
	double Csp;
	double delta0;
	double Vram0;

	double epsilon;
	double var_delta0;
	double var_G0;
	double var_Xram0;
	double var_Me0;
	double sigma_pooldepth;
	double G11;
	double G21;
	double G41;
};

__constant__ SimParams deviceParams[1];

bool InitCUDA();

void ReadConfig(SimParams *hostParams);

int Rng_Initialise(SimParams *hostParams, curandGenerator_t gen, double *device_rand_init , unsigned int NP , unsigned long long seed);

int Rng_LogLikelihood(SimParams *hostParams, curandGenerator_t gen, double *device_rand_Log , unsigned int NP , unsigned long long seed);

int Rng_Move(SimParams *hostParams, curandGenerator_t gen, double *device_rand_move , unsigned int NP , unsigned long long seed);

long load_data(char const * szName, cv_obs** yp, cv_inputs** up);

__device__ double LinearSovler(double * Sigma, double *Error, double *Error_copy, int N)
{
	double x[4] = {0,0,0,0};
	int i,j,k;
	for(k=0; k<N; k++){
       for(i=k+1; i<N; i++){
			double M = Sigma[i*N+k] / Sigma[k*N+k];
			for(j=k; j<N; j++)
				Sigma[i*N+j] -= M * Sigma[k*N+j];
			Error[i] -= M*Error[k];
			}
   }
   for(i=N-1; i>=0; i--){
       float s = 0;
       for(j = i; j<N; j++){
	   s = s+Sigma[i*N+j]*x[j];
       }
       x[i] = (Error[i] - s) / Sigma[i*N+i];
   }
   double result = 0.0;
   for(i = 0 ; i < N ; i ++)
	   result += Error_copy[i] * x[i];
   return result;
}