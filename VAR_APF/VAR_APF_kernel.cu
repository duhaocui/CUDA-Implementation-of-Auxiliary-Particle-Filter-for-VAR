#include "VAR_APF_kernel.cuh"


//======================================================================//
// Initialise the particles and compute their weights                   //
//======================================================================//
__global__ void Initialise( /* Input and output*/
							double *device_X_particles,	
							/* Inputs*/
							double *device_y,			double *device_u,			/* Observations and Inputs*/
							double delta_i,				double *device_rand_init,	/* Random numbers*/	
							unsigned int NP,			/* Number of particles*/
							/* Output*/
							double *device_W_particles)
{
	
	// Load global mvariables to shared memories

	__shared__ double u_0, y_0, y_1, y_2, y_3, y_4, y_5, y_6; 
	if (0 == threadIdx.x)
	{
		u_0 = device_u[0];
		y_0 = device_y[0];
		y_1 = device_y[1];
		y_2 = device_y[2];
		y_3 = device_y[3];
		y_4 = device_y[4];
		y_5 = device_y[5];
		y_6 = device_y[6];
	}
	__syncthreads();

	// fInitialise 

	unsigned int particleID = threadIdx.x + blockIdx.x * blockDim.x;
	double x0,x1,x2,x3,x4,x5,x6,x7,x8;		//9 states of one particle
	double y0,y1,y2,y3,y4,y5,y6;			//7 observations of one particle
	x0 = delta_i + sqrt(deviceParams->var_delta0)*device_rand_init[particleID + NP * 0];
	x1 = y_0 + deviceParams->sigmaG*device_rand_init[particleID + NP * 1];
	x2 = y_1 + deviceParams->sigmaPos*device_rand_init[particleID + NP * 2];
	x3 = y_3 + deviceParams->sigmaLC*device_rand_init[particleID + NP * 3];
	x4 = deviceParams->mu0 + sqrt(deviceParams->dt)*deviceParams->sigmamur*
			device_rand_init[particleID + NP * 4];
	x5 = y_6 + sqrt(deviceParams->dt)*deviceParams->sigmahe*
			device_rand_init[particleID + NP * 5];
	x6 = y_4 + deviceParams->sigmaCL*device_rand_init[particleID + NP * 6];
	x7 = y_5 + deviceParams->sigmaMR*device_rand_init[particleID + NP * 7];
	x8 = u_0 + deviceParams->sigmaImeas*device_rand_init[particleID + NP * 8];

	device_X_particles[particleID + NP * 0] = x0;
	device_X_particles[particleID + NP * 1] = x1;
	device_X_particles[particleID + NP * 2] = x2;
	device_X_particles[particleID + NP * 3] = x3;
	device_X_particles[particleID + NP * 4] = x4;
	device_X_particles[particleID + NP * 5] = x5;
	device_X_particles[particleID + NP * 6] = x6;
	device_X_particles[particleID + NP * 7] = x7;
	device_X_particles[particleID + NP * 8] = x8;

	// logLikelihood

	y0 = x1;	// G: Electrode gap	
	y1 = x2;	// Xram: Ram position	
	y2 = x8;	// I: Measured current	
	y3 = x3;	// Me: Electrode mass	
	y4 = x6;	// Scl: Centerline pool depth	
	y5 = x7;	// Smr: Mid-radius pool depth	
	y6 = x5;	// phe: Helium pressure

	device_W_particles[particleID] = exp(- 0.5 * ( pow(y_0 - y0, 2)/deviceParams->sigmaG/deviceParams->sigmaG +  
		   pow(y_1 - y1, 2)/deviceParams->sigmaPos/deviceParams->sigmaPos +
		   pow(y_2 - y2, 2)/deviceParams->sigmaImeas/deviceParams->sigmaImeas + 
		   pow(y_3 - y3, 2)/deviceParams->sigmaLC/deviceParams->sigmaLC + 
		   pow(y_4 - y4, 2)/deviceParams->sigmaCL/deviceParams->sigmaCL +
		   pow(y_5 - y5, 2)/deviceParams->sigmaMR/deviceParams->sigmaMR + 
		   pow(y_6 - y6, 2)/deviceParams->sigmahemeas/deviceParams->sigmahemeas ));
}

__global__ void Initialise_controller(double *device_C_particles, double *device_X_particles, 
									  double *device_u, double *device_rand_control, unsigned int NP){
	// Load global mvariables to shared memories
	__shared__ double u_0, u_1; 
	if (0 == threadIdx.x)
	{
		u_0 = device_u[0]; //Ic
		u_1 = device_u[1]; //Vramc
	}
	__syncthreads();

	unsigned int particleID = threadIdx.x + blockIdx.x * blockDim.x;
	// Initialise the state
	device_C_particles[particleID + NP * 0] = device_X_particles[particleID + NP * 0];
	device_C_particles[particleID + NP * 1] = device_X_particles[particleID + NP * 1];
	device_C_particles[particleID + NP * 2] = device_X_particles[particleID + NP * 2];
	device_C_particles[particleID + NP * 3] = device_X_particles[particleID + NP * 3];
	device_C_particles[particleID + NP * 4] = device_X_particles[particleID + NP * 4];
	device_C_particles[particleID + NP * 5] = device_X_particles[particleID + NP * 5];
	device_C_particles[particleID + NP * 6] = device_X_particles[particleID + NP * 6];
	device_C_particles[particleID + NP * 7] = device_X_particles[particleID + NP * 7];
	device_C_particles[particleID + NP * 8] = device_X_particles[particleID + NP * 8];
	// Initialise u_bar
	//xMc(j).u_bar.Ic = u(i-1).Ic+randn(1)/uwt(1);
	device_C_particles[particleID + NP * 9] = u_0 + device_rand_control[particleID + NP * 0]/(1e+2/deviceParams->I0);
	//xMc(j).u_bar.Vramc = u(i-1).Vramc+randn(1)/uwt(2);
	device_C_particles[particleID + NP * 10] = u_1 + device_rand_control[particleID + NP * 1]/(1e+2/deviceParams->Vram0);
	// Initialise u_tilde
	device_C_particles[particleID + NP * 11] = device_C_particles[particleID + NP * 9];
	device_C_particles[particleID + NP * 12] = device_C_particles[particleID + NP * 10];
}



//======================================================================//
// Lookahead								                            //
//======================================================================//
__global__ void Lookahead_init(/* Inputs*/
					double *device_X_particles,	
					double *device_y,			double *device_u,			/* Observations and Inputs*/	
					unsigned int NP,			/*unsigned int iterate,*/		/* Number of particles*/
					/* Output*/
					double *device_W_particles)
{
	__shared__ double u_0, u_1, y_0, y_1, y_2, y_3, y_4, y_5, y_6; 
	if (0 == threadIdx.x)
	{
		u_0 = device_u[0];
		u_1 = device_u[1];
		y_0 = device_y[0];
		y_1 = device_y[1];
		y_2 = device_y[2];
		y_3 = device_y[3];
		y_4 = device_y[4];
		y_5 = device_y[5];
		y_6 = device_y[6];
	}
	__syncthreads();

	unsigned int particleID = threadIdx.x + blockIdx.x * blockDim.x;
	double x0,x1,x2,x3,x4,x5,x6,x7,x8;	//9 states of one particle
	double new_x0,new_x1,new_x2,new_x3,new_x4,new_x5,new_x6,new_x7,new_x8;	//9 states of one new particle
	double y0,y1,y2,y3,y4,y5,y6;			//7 observations of one particle
	x0 = device_X_particles[particleID + NP * 0];
	x1 = device_X_particles[particleID + NP * 1];
	x2 = device_X_particles[particleID + NP * 2];
	x3 = device_X_particles[particleID + NP * 3];
	x4 = device_X_particles[particleID + NP * 4];
	x5 = device_X_particles[particleID + NP * 5];
	x6 = device_X_particles[particleID + NP * 6];
	x7 = device_X_particles[particleID + NP * 7];
	x8 = device_X_particles[particleID + NP * 8];
	// Reconstruct inputs
	double  Vram, P;
	Vram = u_1;
	P = (deviceParams->Vc + deviceParams->Ri * x8) * x8;
	// Differential equations
	double deltadot, Gdot, Xramdot, Medot;
	double Scldot, Smrdot;
	deltadot = deviceParams->alphar*deviceParams->Cdd/x0 - 4.0*deviceParams->Cdp*x4*
				P/pi/deviceParams->De/deviceParams->De/deviceParams->hm;
	Gdot = deviceParams->a0 * (-deviceParams->alphar*deviceParams->Csd/x0 + 4.0*deviceParams->Csp*x4*
				P/deviceParams->hm/pi/deviceParams->De/deviceParams->De)-Vram;
	Medot = pi*deviceParams->rhor*deviceParams->De*deviceParams->De*deviceParams->alphar*
				deviceParams->Csd/4.0/x0 - deviceParams->rhor*deviceParams->Csp*x4*P/deviceParams->hm;
	Scldot = -deviceParams->Acl*(x6-deviceParams->Scl0)+deviceParams->Bdeltacl*(x0-deviceParams->delta0)
				+deviceParams->Bicl*(x8-deviceParams->I0)+deviceParams->Bmucl*(x4-deviceParams->mu0)+
				deviceParams->Bhecl*(x5-deviceParams->phe0);
	Smrdot = -deviceParams->Amr*(x7-deviceParams->Smr0)+deviceParams->Bdeltamr*(x0-deviceParams->delta0)
				+deviceParams->Bimr*(x8-deviceParams->I0)+deviceParams->Bmumr*(x4-deviceParams->mu0)+
				deviceParams->Bhemr*(x5-deviceParams->phe0);
	// Discrete propagation equations
	new_x0 = x0 + deltadot*deviceParams->dt;
	new_x1 = x1 + Gdot*deviceParams->dt;
	new_x2 = x2 + Vram*deviceParams->dt;
	new_x3 = x3 + Medot*deviceParams->dt;
	new_x4 = x4;
	new_x5 = x5;
	new_x6 = x6 + Scldot*deviceParams->dt;
	new_x7 = x7 + Smrdot*deviceParams->dt;
	new_x8 = u_0 + (u_0 - x8)*exp(-deviceParams->dt/1.0);

	// Calculate the weight
	y0 = new_x1 ;//+ device_rand_log[particleID + NP * 0];	// G: Electrode gap	
	y1 = new_x2 ;//+ device_rand_log[particleID + NP * 1];	// Xram: Ram position	
	y2 = new_x8 ;//+ device_rand_log[particleID + NP * 2];	// I: Measured current	
	y3 = new_x3 ;//+ device_rand_log[particleID + NP * 3];	// Me: Electrode mass	
	y4 = new_x6 ;//+ device_rand_log[particleID + NP * 5];	// Scl: Centerline pool depth	
	y5 = new_x7 ;//+ device_rand_log[particleID + NP * 6];	// Smr: Mid-radius pool depth	
	y6 = new_x5 ;//+ device_rand_log[particleID + NP * 7];	// phe: Helium pressure

	//inv(Sigma11) =| 24.9959	6.6166	|
	//    	        | 6.6166	29414	|
	//inv(Sigma12) =|6.8388e-05		5.3928e-07		-6.4856e-06		-3.1163e-06		|
	//				|5.3928e-07		2.5e-05			7.9925e-10		3.8403e-10		|
	//				|-6.4856e-06	7.9925e-10		1				-4.6186e-09		|
	//				|-3.1163e-06	3.8403e-10		-4.6186e-09		1				|
	//inv(Sigma13) = 6.4935e3;
	
	double Sigma11[4] = {0.04,-9e-6,-9e-6,3.4e-5};
	double Sigma12[16] = {14625,	-315.479287411471,	0.0948528,	0.045576,
							-315.479287411471,	40006.9116097768,	-0.00207806206617936,	-0.000998491944657307,
							0.0948528,	-0.00207806206617936,	1.00000062479539,	3.00209112e-07,
							0.045576,	-0.000998491944657307,	3.00209112000000e-07,	1.00000014424804};
	double Error11[2];
	double Error12[4];
	Error11[0] = y_0-y0;
	Error11[1] = y_1-y1;
	Error12[0] = y_2-y2;
	Error12[1] = y_3-y3;
	Error12[2] = y_4-y4;
	Error12[3] = y_5-y5;
	double *Error11_copy = Error11;
	double *Error12_copy = Error12;
	
	device_W_particles[particleID] = exp(log(device_W_particles[particleID]) - 0.5 *
										(LinearSovler(Sigma11,Error11,Error11_copy,2)+
										LinearSovler(Sigma12,Error12,Error12_copy,4)+
										(y_6-y6)*(y_6-y6)*6.4935e3));

	//device_W_particles[particleID] = exp(log(device_W_particles[particleID]) - 0.5 * 
	//		//Error11*(Sigma11\Error11')
	//	 ( (y_0-y0)*(y_0-y0)*24.9959+(y_1-y1)*(y_1-y1)*29414+ 2*(y_0-y0)*(y_1-y1)*6.6166+
	//		//Error12*(Sigma12\Error12')
	//	   (y_2-y2)*(y_2-y2)*6.8388e-05+(y_3-y3)*(y_3-y3)*2.5e-05+
	//	   (y_4-y4)*(y_4-y4)*1+(y_5-y5)*(y_5-y5)*1+
	//	   2*(y_2-y2)*(y_3-y3)*5.3928e-07 + 2*(y_2-y2)*(y_4-y4)*(-6.4856e-06)+2*(y_2-y2)*(y_5-y5)*(-3.1163e-06)+
	//	   2*(y_3-y3)*(y_4-y4)*7.9925e-10+2*(y_3-y3)*(y_5-y5)*3.8403e-10+
	//	   2*(y_4-y4)*(y_5-y5)*(-4.6186e-09)+
	//		//Error13*(Sigma13\Error13') 
	//	   (y_6-y6)*(y_6-y6)*6.4935e3));
}


__global__ void Lookahead(/* Inputs*/
					double *device_X_particles,	
					double *device_y,			double *device_u,			/* Observations and Inputs*/	
					unsigned int NP,			/*unsigned int iterate,*/		/* Number of particles*/
					/* Output*/
					double *device_W_particles)
{
	__shared__ double u_0, u_1, y_0, y_1, y_2, y_3, y_4, y_5, y_6; 
	if (0 == threadIdx.x)
	{
		u_0 = device_u[0];
		u_1 = device_u[1];
		y_0 = device_y[0];
		y_1 = device_y[1];
		y_2 = device_y[2];
		y_3 = device_y[3];
		y_4 = device_y[4];
		y_5 = device_y[5];
		y_6 = device_y[6];
	}
	__syncthreads();

	unsigned int particleID = threadIdx.x + blockIdx.x * blockDim.x;
	double x0,x1,x2,x3,x4,x5,x6,x7,x8;	//9 states of one particle
	double new_x0,new_x1,new_x2,new_x3,new_x4,new_x5,new_x6,new_x7,new_x8;	//9 states of one new particle
	double y0,y1,y2,y3,y4,y5,y6;			//7 observations of one particle
	x0 = device_X_particles[particleID + NP * 0];
	x1 = device_X_particles[particleID + NP * 1];
	x2 = device_X_particles[particleID + NP * 2];
	x3 = device_X_particles[particleID + NP * 3];
	x4 = device_X_particles[particleID + NP * 4];
	x5 = device_X_particles[particleID + NP * 5];
	x6 = device_X_particles[particleID + NP * 6];
	x7 = device_X_particles[particleID + NP * 7];
	x8 = device_X_particles[particleID + NP * 8];
	// Reconstruct inputs
	double  Vram, P;
	Vram = u_1;
	P = (deviceParams->Vc + deviceParams->Ri * x8) * x8;
	// Differential equations
	double deltadot, Gdot, Xramdot, Medot;
	double Scldot, Smrdot;
	deltadot = deviceParams->alphar*deviceParams->Cdd/x0 - 4.0*deviceParams->Cdp*x4*
				P/pi/deviceParams->De/deviceParams->De/deviceParams->hm;
	Gdot = deviceParams->a0 * (-deviceParams->alphar*deviceParams->Csd/x0 + 4.0*deviceParams->Csp*x4*
				P/deviceParams->hm/pi/deviceParams->De/deviceParams->De)-Vram;
	Medot = pi*deviceParams->rhor*deviceParams->De*deviceParams->De*deviceParams->alphar*
				deviceParams->Csd/4.0/x0 - deviceParams->rhor*deviceParams->Csp*x4*P/deviceParams->hm;
	Scldot = -deviceParams->Acl*(x6-deviceParams->Scl0)+deviceParams->Bdeltacl*(x0-deviceParams->delta0)
				+deviceParams->Bicl*(x8-deviceParams->I0)+deviceParams->Bmucl*(x4-deviceParams->mu0)+
				deviceParams->Bhecl*(x5-deviceParams->phe0);
	Smrdot = -deviceParams->Amr*(x7-deviceParams->Smr0)+deviceParams->Bdeltamr*(x0-deviceParams->delta0)
				+deviceParams->Bimr*(x8-deviceParams->I0)+deviceParams->Bmumr*(x4-deviceParams->mu0)+
				deviceParams->Bhemr*(x5-deviceParams->phe0);
	// Discrete propagation equations
	new_x0 = x0 + deltadot*deviceParams->dt;
	new_x1 = x1 + Gdot*deviceParams->dt;
	new_x2 = x2 + Vram*deviceParams->dt;
	new_x3 = x3 + Medot*deviceParams->dt;
	new_x4 = x4;
	new_x5 = x5;
	new_x6 = x6 + Scldot*deviceParams->dt;
	new_x7 = x7 + Smrdot*deviceParams->dt;
	new_x8 = u_0 + (u_0 - x8)*exp(-deviceParams->dt/1.0);

	// Calculate the weight
	y0 = new_x1 ;//+ device_rand_log[particleID + NP * 0];	// G: Electrode gap	
	y1 = new_x2 ;//+ device_rand_log[particleID + NP * 1];	// Xram: Ram position	
	y2 = new_x8 ;//+ device_rand_log[particleID + NP * 2];	// I: Measured current	
	y3 = new_x3 ;//+ device_rand_log[particleID + NP * 3];	// Me: Electrode mass	
	y4 = new_x6 ;//+ device_rand_log[particleID + NP * 5];	// Scl: Centerline pool depth	
	y5 = new_x7 ;//+ device_rand_log[particleID + NP * 6];	// Smr: Mid-radius pool depth	
	y6 = new_x5 ;//+ device_rand_log[particleID + NP * 7];	// phe: Helium pressure

	//inv(Sigma11) =| 24.9959	6.6166	|
	//    	        | 6.6166	29414	|
	//inv(Sigma12) =|6.8388e-05		5.3928e-07		-6.4856e-06		-3.1163e-06		|
	//				|5.3928e-07		2.5e-05			7.9925e-10		3.8403e-10		|
	//				|-6.4856e-06	7.9925e-10		1				-4.6186e-09		|
	//				|-3.1163e-06	3.8403e-10		-4.6186e-09		1				|
	//inv(Sigma13) = 6.4935e3;
	
	double Sigma11[4] = {0.04,-9e-6,-9e-6,3.4e-5};
	double Sigma12[16] = {14625,	-315.479287411471,	0.0948528,	0.045576,
							-315.479287411471,	40006.9116097768,	-0.00207806206617936,	-0.000998491944657307,
							0.0948528,	-0.00207806206617936,	1.00000062479539,	3.00209112e-07,
							0.045576,	-0.000998491944657307,	3.00209112000000e-07,	1.00000014424804};
	double Error11[2];
	double Error12[4];
	Error11[0] = y_0-y0;
	Error11[1] = y_1-y1;
	Error12[0] = y_2-y2;
	Error12[1] = y_3-y3;
	Error12[2] = y_4-y4;
	Error12[3] = y_5-y5;
	double *Error11_copy = Error11;
	double *Error12_copy = Error12;
	
	device_W_particles[particleID] = exp( - 0.5 * // This is the only difference from lookahead_init
										(LinearSovler(Sigma11,Error11,Error11_copy,2)+
										LinearSovler(Sigma12,Error12,Error12_copy,4)+
										(y_6-y6)*(y_6-y6)*6.4935e3));
	//device_W_particles[particleID] = exp( - 0.5 *  // This is the only difference from lookahead_init
	//		//Error11*(Sigma11\Error11')
	//	 ( (y_0-y0)*(y_0-y0)*24.9959+(y_1-y1)*(y_1-y1)*29414+ 2*(y_0-y0)*(y_1-y1)*6.6166+
	//		//Error12*(Sigma12\Error12')
	//	   (y_2-y2)*(y_2-y2)*6.8388e-05+(y_3-y3)*(y_3-y3)*2.5e-05+
	//	   (y_4-y4)*(y_4-y4)*1+(y_5-y5)*(y_5-y5)*1+
	//	   2*(y_2-y2)*(y_3-y3)*5.3928e-07 + 2*(y_2-y2)*(y_4-y4)*(-6.4856e-06)+2*(y_2-y2)*(y_5-y5)*(-3.1163e-06)+
	//	   2*(y_3-y3)*(y_4-y4)*7.9925e-10+2*(y_3-y3)*(y_5-y5)*3.8403e-10+
	//	   2*(y_4-y4)*(y_5-y5)*(-4.6186e-09)+
	//		//Error13*(Sigma13\Error13') 
	//	   (y_6-y6)*(y_6-y6)*6.4935e3));
}

//======================================================================//
// Propagation								                            //
//======================================================================//
__global__ void propagation(/* Inputs*/
					double *device_X_particles,	
					double *device_u,			/* Control*/
					double *device_rand_move,	/* Random numbers*/	
					unsigned int NP,		/* Number of particles*/
					/* Output*/
					double *device_X_particles_new)
{
	__shared__ double u_0, u_1; 
	if (0 == threadIdx.x)
	{
		u_0 = device_u[0];
		u_1 = device_u[1];
	}
	__syncthreads();

	unsigned int particleID = threadIdx.x + blockIdx.x * blockDim.x;
	double x0,x1,x2,x3,x4,x5,x6,x7,x8;	//9 states of one particle
	double new_x0,new_x1,new_x2,new_x3,new_x4,new_x5,new_x6,new_x7,new_x8;	//9 states of one new particle
	x0 = device_X_particles[particleID + NP * 0];
	x1 = device_X_particles[particleID + NP * 1];
	x2 = device_X_particles[particleID + NP * 2];
	x3 = device_X_particles[particleID + NP * 3];
	x4 = device_X_particles[particleID + NP * 4];
	x5 = device_X_particles[particleID + NP * 5];
	x6 = device_X_particles[particleID + NP * 6];
	x7 = device_X_particles[particleID + NP * 7];
	x8 = device_X_particles[particleID + NP * 8];
	// Reconstruct inputs
	double  Vram, P;
	Vram = u_1;
	P = (deviceParams->Vc + deviceParams->Ri * x8) * x8;
	// Sample process uncertainties
	double dI = deviceParams->dt*deviceParams->sigmaI*device_rand_move[particleID + NP * 0];
	double dVram = deviceParams->dt*deviceParams->sigmaVram*device_rand_move[particleID + NP * 1];
	double dmu = sqrt(deviceParams->dt)*deviceParams->sigmamur*device_rand_move[particleID + NP * 2];
	double dhe = sqrt(deviceParams->dt)*deviceParams->sigmahe*device_rand_move[particleID + NP * 3];
	// Differential equations
	double deltadot, Gdot, Xramdot, Medot;
	double Scldot, Smrdot;
	deltadot = deviceParams->alphar*deviceParams->Cdd/x0 - 4.0*deviceParams->Cdp*x4*
				P/pi/deviceParams->De/deviceParams->De/deviceParams->hm;
	Gdot = deviceParams->a0 * (-deviceParams->alphar*deviceParams->Csd/x0 + 
			4.0*deviceParams->Csp*x4*P/deviceParams->hm/pi/deviceParams->De/deviceParams->De)-Vram;
	Medot = pi*deviceParams->rhor*deviceParams->De*deviceParams->De*deviceParams->alphar*
			deviceParams->Csd/4.0/x0 - deviceParams->rhor*deviceParams->Csp*x4*P/deviceParams->hm;
	if (Medot >= deviceParams->epsilon)
		Medot = deviceParams->epsilon;
	Scldot = -deviceParams->Acl*(x6-deviceParams->Scl0)+deviceParams->Bdeltacl*(x0-deviceParams->delta0)
				+deviceParams->Bicl*(x8-deviceParams->I0)+deviceParams->Bmucl*(x4-deviceParams->mu0)+
				deviceParams->Bhecl*(x5-deviceParams->phe0);
	Smrdot = -deviceParams->Amr*(x7-deviceParams->Smr0)+deviceParams->Bdeltamr*(x0-deviceParams->delta0)
				+deviceParams->Bimr*(x8-deviceParams->I0)+deviceParams->Bmumr*(x4-deviceParams->mu0)+
				deviceParams->Bhemr*(x5-deviceParams->phe0);
	// Discrete propagation equations
	// Delta: electrode thermal boundary layer
	new_x0 = x0 + deltadot*deviceParams->dt + deviceParams->G11*dI;
	if (new_x0 < deviceParams->epsilon)
		new_x0 = deviceParams->epsilon;
	// G: electrode gap
	new_x1 = x1 + Gdot*deviceParams->dt+ deviceParams->G21*dI - dVram;
	if (new_x1 < deviceParams->epsilon)
		new_x1 = deviceParams->epsilon;
	// Xram: ram position
	new_x2 = x2 + deviceParams->dt*Vram + dVram;
	// Me: electrode mass
	new_x3 = x3 + Medot*deviceParams->dt + deviceParams->G41*dI;
	if (new_x3 < deviceParams->epsilon)
		new_x3 = deviceParams->epsilon;
	// mu: melting efficiency	
	new_x4 = x4 + dmu;
	if (new_x4 < deviceParams->epsilon)
		new_x4 = deviceParams->epsilon;

	// phe: helium pressure
	new_x5 = x5 + dhe;
	if (new_x5 < deviceParams->epsilon)
		new_x5 = deviceParams->epsilon;

	// Scl: centerline pool depth and Smr: mid-radius pool depth	
	new_x6 = x6 + Scldot*deviceParams->dt + deviceParams->Bicl*dI;
	if (new_x6 < deviceParams->epsilon)
		new_x6 = deviceParams->epsilon;

	new_x7 = x7 + Smrdot*deviceParams->dt + deviceParams->Bimr*dI;
	if (new_x7 < deviceParams->epsilon)
		new_x7 = deviceParams->epsilon;

	double mI = u_0 + (u_0 - x8)*exp(-deviceParams->dt/1.0);
	new_x8 = mI + dI;

	// Copy new particle to global memory
	device_X_particles_new[particleID + NP * 0] = new_x0; 
	device_X_particles_new[particleID + NP * 1] = new_x1;
	device_X_particles_new[particleID + NP * 2] = new_x2;
	device_X_particles_new[particleID + NP * 3] = new_x3;
	device_X_particles_new[particleID + NP * 4] = new_x4;
	device_X_particles_new[particleID + NP * 5] = new_x5;
	device_X_particles_new[particleID + NP * 6] = new_x6;
	device_X_particles_new[particleID + NP * 7] = new_x7;
	device_X_particles_new[particleID + NP * 8] = new_x8;

}

__global__ void Propagation_controller(/* Inputs*/
					double *device_C_particles,	
					double *device_y,
					double *device_rand_control,	/* Random numbers*/	
					unsigned int NP,		/* Number of particles*/
					/* Output*/
					double *device_C_particles_new,
					double *device_WC_particles)
{
	__shared__ double y_0, y_1, y_2, y_3, y_4, y_5, y_6; 
	if (0 == threadIdx.x)
	{
		y_0 = device_y[0];
		y_1 = device_y[1];
		y_2 = device_y[2];
		y_3 = device_y[3];
		y_4 = device_y[4];
		y_5 = device_y[5];
		y_6 = device_y[6];
	}
	__syncthreads();

	unsigned int particleID = threadIdx.x + blockIdx.x * blockDim.x;
	double u0,u1;
	double x0,x1,x2,x3,x4,x5,x6,x7,x8;	//9 states of one particle
	double new_x0,new_x1,new_x2,new_x3,new_x4,new_x5,new_x6,new_x7,new_x8;	//9 states of one new particle
	double y0,y1,y2,y3,y4,y5,y6; // 7 observations
	x0 = device_C_particles[particleID + NP * 0];
	x1 = device_C_particles[particleID + NP * 1];
	x2 = device_C_particles[particleID + NP * 2];
	x3 = device_C_particles[particleID + NP * 3];
	x4 = device_C_particles[particleID + NP * 4];
	x5 = device_C_particles[particleID + NP * 5];
	x6 = device_C_particles[particleID + NP * 6];
	x7 = device_C_particles[particleID + NP * 7];
	x8 = device_C_particles[particleID + NP * 8];

	u0 = device_C_particles[particleID + NP * 9];
	u1 = device_C_particles[particleID + NP * 10];

	// Reconstruct inputs
	double  Vram, P;
	Vram = u1;
	P = (deviceParams->Vc + deviceParams->Ri * x8) * x8;
	// Differential equations
	double deltadot, Gdot, Xramdot, Medot;
	double Scldot, Smrdot;
	deltadot = deviceParams->alphar*deviceParams->Cdd/x0 - 4.0*deviceParams->Cdp*x4*
				P/pi/deviceParams->De/deviceParams->De/deviceParams->hm;
	Gdot = deviceParams->a0 * (-deviceParams->alphar*deviceParams->Csd/x0 + 
			4.0*deviceParams->Csp*x4*P/deviceParams->hm/pi/deviceParams->De/deviceParams->De)-Vram;
	Medot = pi*deviceParams->rhor*deviceParams->De*deviceParams->De*deviceParams->alphar*
			deviceParams->Csd/4.0/x0 - deviceParams->rhor*deviceParams->Csp*x4*P/deviceParams->hm;
	if (Medot >= deviceParams->epsilon)
		Medot = deviceParams->epsilon;
	Scldot = -deviceParams->Acl*(x6-deviceParams->Scl0)+deviceParams->Bdeltacl*(x0-deviceParams->delta0)
				+deviceParams->Bicl*(x8-deviceParams->I0)+deviceParams->Bmucl*(x4-deviceParams->mu0)+
				deviceParams->Bhecl*(x5-deviceParams->phe0);
	Smrdot = -deviceParams->Amr*(x7-deviceParams->Smr0)+deviceParams->Bdeltamr*(x0-deviceParams->delta0)
				+deviceParams->Bimr*(x8-deviceParams->I0)+deviceParams->Bmumr*(x4-deviceParams->mu0)+
				deviceParams->Bhemr*(x5-deviceParams->phe0);
	// Discrete propagation equations
	// Delta: electrode thermal boundary layer
	new_x0 = x0 + deltadot*deviceParams->dt;
	if (new_x0 < deviceParams->epsilon)
		new_x0 = deviceParams->epsilon;
	// G: electrode gap
	new_x1 = x1 + Gdot*deviceParams->dt;
	if (new_x1 < deviceParams->epsilon)
		new_x1 = deviceParams->epsilon;
	// Xram: ram position
	new_x2 = x2 + deviceParams->dt*Vram;
	// Me: electrode mass
	new_x3 = x3 + Medot*deviceParams->dt;
	if (new_x3 < deviceParams->epsilon)
		new_x3 = deviceParams->epsilon;
	// mu: melting efficiency	
	new_x4 = x4;
	if (new_x4 < deviceParams->epsilon)
		new_x4 = deviceParams->epsilon;

	// phe: helium pressure
	new_x5 = x5;
	if (new_x5 < deviceParams->epsilon)
		new_x5 = deviceParams->epsilon;

	// Scl: centerline pool depth and Smr: mid-radius pool depth	
	new_x6 = x6 + Scldot*deviceParams->dt;
	if (new_x6 < deviceParams->epsilon)
		new_x6 = deviceParams->epsilon;

	new_x7 = x7 + Smrdot*deviceParams->dt;
	if (new_x7 < deviceParams->epsilon)
		new_x7 = deviceParams->epsilon;

	double mI = u0 + (u0 - x8)*exp(-deviceParams->dt/1.0);
	new_x8 = mI;

	// Copy new particle to global memory
	device_C_particles_new[particleID + NP * 0] = new_x0; 
	device_C_particles_new[particleID + NP * 1] = new_x1;
	device_C_particles_new[particleID + NP * 2] = new_x2;
	device_C_particles_new[particleID + NP * 3] = new_x3;
	device_C_particles_new[particleID + NP * 4] = new_x4;
	device_C_particles_new[particleID + NP * 5] = new_x5;
	device_C_particles_new[particleID + NP * 6] = new_x6;
	device_C_particles_new[particleID + NP * 7] = new_x7;
	device_C_particles_new[particleID + NP * 8] = new_x8;

	device_C_particles_new[particleID + NP * 9] = device_C_particles[particleID + NP * 9] 
												+ device_rand_control[particleID + NP * 0]/(1e+2/deviceParams->I0);
	device_C_particles_new[particleID + NP * 10] = device_C_particles[particleID + NP * 10] 
												+ device_rand_control[particleID + NP * 1]/(1e+2/deviceParams->Vram0);
	device_C_particles_new[particleID + NP * 11] = device_C_particles[particleID + NP * 11];
	device_C_particles_new[particleID + NP * 12] = device_C_particles[particleID + NP * 12];

	// Calculate the weight
	y0 = new_x1 ;//+ device_rand_log[particleID + NP * 0];	// G: Electrode gap	
	y1 = new_x2 ;//+ device_rand_log[particleID + NP * 1];	// Xram: Ram position	
	y2 = new_x8 ;//+ device_rand_log[particleID + NP * 2];	// I: Measured current	
	y3 = new_x3 ;//+ device_rand_log[particleID + NP * 3];	// Me: Electrode mass	
	y4 = new_x6 ;//+ device_rand_log[particleID + NP * 5];	// Scl: Centerline pool depth	
	y5 = new_x7 ;//+ device_rand_log[particleID + NP * 6];	// Smr: Mid-radius pool depth	
	y6 = new_x5 ;//+ device_rand_log[particleID + NP * 7];	// phe: Helium pressure
	

	//ywt = [5e+1/G0 0  0 0 5e+1/Scl0 0e+0/Smr0 0];
	device_WC_particles[particleID] = 1e-250 + exp(-0.5 * 
									(((y_0-y0)*5e+1/deviceParams->G0)*((y_0-y0)*5e+1/deviceParams->G0) +
									 ((y_4-y4)*5e+1/deviceParams->Scl0)*((y_4-y4)*5e+1/deviceParams->Scl0) +
									 ((y_5-y5)*0e+0/deviceParams->Smr0)*((y_5-y5)*0e+0/deviceParams->Smr0)));

}




__global__ void normalize(double *inArray, int numElements, int *maxj)
{
	__shared__ double sum;
	unsigned int thid = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;
	if(threadIdx.x == 0) sum = inArray[numElements-1];
	__syncthreads();
	int index = thid;
	double temp;
	while(index < numElements){
		maxj[index] = (int)(inArray[index] / (sum) * numElements);
		index += stride;
	}
}

__global__ void resample_init(int numElements, double *particleArray, double *particleArray_new, 
						 int *maxj, double *device_W_particles, double *device_W_particles_new)
{
	unsigned int thid = blockDim.x * blockIdx.x + threadIdx.x;
	int particleID = thid;
	//int low_maxj, high_maxj;
	int minidx, maxidx, mididx;
	int stride = gridDim.x * blockDim.x;
	
	while(particleID < numElements){
		minidx = 0;
		maxidx = numElements - 1;
		while (minidx <= maxidx) {
			mididx = (minidx + maxidx) / 2;
			if (particleID > maxj[mididx + 1]) {
				minidx = mididx + 1;
			} else {
				if (particleID > maxj[mididx]) {
					break;
				}
				maxidx = mididx - 1;
			}
		}
		for (int k = 0; k < SIZEOFX; k++)
			particleArray_new[particleID + k * numElements] = particleArray[mididx + k * numElements];
		device_W_particles_new[particleID] = device_W_particles[mididx];
		particleID += stride;
	}
}

__global__ void resample(int numElements, double *particleArray, double *particleArray_new, int *maxj)
{
	unsigned int thid = blockDim.x * blockIdx.x + threadIdx.x;
	int particleID = thid;
	//int low_maxj, high_maxj;
	int minidx, maxidx, mididx;
	int stride = gridDim.x * blockDim.x;
	
	while(particleID < numElements){
		minidx = 0;
		maxidx = numElements - 1;
		while (minidx <= maxidx) {
			mididx = (minidx + maxidx) / 2;
			if (particleID > maxj[mididx + 1]) {
				minidx = mididx + 1;
			} else {
				if (particleID > maxj[mididx]) {
					break;
				}
				maxidx = mididx - 1;
			}
		}
		for (int k = 0; k < SIZEOFX; k++)
			particleArray_new[particleID + k * numElements] = particleArray[mididx + k * numElements];
//		device_W_particles[particleID] = 1;
		particleID += stride;
	}
}


__global__ void resample_control(int numElements, double *particleArray, double *particleArray_new, int *maxj)
{
	unsigned int thid = blockDim.x * blockIdx.x + threadIdx.x;
	int particleID = thid;
	//int low_maxj, high_maxj;
	int minidx, maxidx, mididx;
	int stride = gridDim.x * blockDim.x;
	
	while(particleID < numElements){
		minidx = 0;
		maxidx = numElements - 1;
		while (minidx <= maxidx) {
			mididx = (minidx + maxidx) / 2;
			if (particleID > maxj[mididx + 1]) {
				minidx = mididx + 1;
			} else {
				if (particleID > maxj[mididx]) {
					break;
				}
				maxidx = mididx - 1;
			}
		}
		for (int k = 0; k < SIZEOFC; k++)
			particleArray_new[particleID + k * numElements] = particleArray[mididx + k * numElements];
//		device_W_particles[particleID] = 1;
		particleID += stride;
	}
}


__global__ void calculateXe(int reduceBlocks, double *statesBlockSumArray, double *Xe) {
	if (threadIdx.x < SIZEOFX) {
		Xe[threadIdx.x] = statesBlockSumArray[reduceBlocks * threadIdx.x];
	}
}

__global__ void calculateCe(int reduceBlocks, double *statesBlockSumArray, double *Ce) {
	if (threadIdx.x < 2) {
		Ce[threadIdx.x] = statesBlockSumArray[reduceBlocks * threadIdx.x];
	}
}

__global__ void extract_control(double *device_C_estimate, double *device_C_particles, int NP){
	unsigned int thid = blockDim.x * blockIdx.x + threadIdx.x;
	device_C_estimate[thid] = device_C_particles[thid + NP*11];
	device_C_estimate[thid+NP] = device_C_particles[thid + NP*12];
}