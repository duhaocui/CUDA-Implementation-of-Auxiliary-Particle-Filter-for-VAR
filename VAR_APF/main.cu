#include "VAR_APF_kernel.cu" 
#include "scan_kernel.cu"
#include <random>
#include <time.h>
#include <algorithm>

std::random_device rd;
std::default_random_engine generator( rd() );
std::normal_distribution<double> distribution(0, 1.0);

using namespace std;

///The observations
cv_obs * y;
///The inputs
cv_inputs * u;

int main (void)
{
	unsigned int iteration = 1201; // Number of iterations
	FILE *fid_time = fopen("time.txt", "w");
	//======================================================================//
	// Initialise CUDA Device                                               //
	//======================================================================//
	if(!InitCUDA()) return 0;
	//======================================================================//
	// Read Configurations                                                  //
	//======================================================================//
	SimParams *hostParams;
	hostParams = (SimParams *)malloc(sizeof(SimParams));
	ReadConfig(hostParams);


	double Gdot = hostParams->delta0 - hostParams->Csp*hostParams->hm;










	CUDA_CALL(cudaMemcpyToSymbol(deviceParams, hostParams, sizeof(SimParams), 0, 
				cudaMemcpyHostToDevice));	
	//======================================================================//
	// Load y and u from data.csv                                           //
	//======================================================================//
	load_data("data.csv", &y, &u);	
	//======================================================================//
	// Initialise y and u on Device Memory                                  //
	//======================================================================//
	double *device_y, *host_y;
	CUDA_CALL(cudaMalloc((void **) &device_y, sizeof(double) * SIZEOFY));
	host_y = (double*)malloc(sizeof(double) * SIZEOFY);
	double *device_u, *host_u;
	CUDA_CALL(cudaMalloc((void **) &device_u, sizeof(double) * SIZEOFU));
	host_u = (double*)malloc(sizeof(double) * SIZEOFU);
	host_y[0] = y[0].G;
	host_y[1] = y[0].Xram;     
	host_y[2] = y[0].I;
	host_y[3] = y[0].Me;
	host_y[4] = y[0].Scl;
	host_y[5] = y[0].Smr;
	host_y[6] = y[0].phe;
	host_u[0] = u[0].Ic;
	host_u[1] = u[0].Vramc;
	CUDA_CALL(cudaMemcpy(device_y , host_y , sizeof(double) * SIZEOFY , cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(device_u , host_u , sizeof(double) * SIZEOFU , cudaMemcpyHostToDevice));
	//======================================================================//
	// Allocate memory for XE                   	                        //
	//======================================================================//
	double *device_Xe, *host_Xe;
	CUDA_CALL(cudaMalloc((void **) &device_Xe, sizeof(double) * SIZEOFX));
	host_Xe = (double *)malloc(sizeof(double) * SIZEOFX);	

	//>>>>>>>>>>>>>>>>>>>>>>>Control Estimation
	double *device_Ce, *host_Ce;
	CUDA_CALL(cudaMalloc((void **) &device_Ce, sizeof(double) * 2));
	host_Ce = (double *)malloc(sizeof(double) * 2);

//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>//
//	The code below is relevant to NP
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>//
	
	//int experiment = 1;
	int power = 17;
	unsigned int NP = 1<<power;
			//printf("NP = %d\tExperiment %d\n",NP,experiment);
			//======================================================================//
			// Open Files to Record Data                   	                        //
			//======================================================================//
			char Estimation_filename[50];
			char Expe[2];
			sprintf(Estimation_filename, "%d", power);
			strcat (Estimation_filename, "_APF_estimation_");
			//sprintf(Expe, "%d", experiment);
			//strcat (Estimation_filename, Expe);
			strcat (Estimation_filename, ".txt");
			FILE *fid_Estimation = fopen(Estimation_filename, "w");
			FILE *fid_control = fopen("Estimate_control.txt","w");
			//======================================================================//
			// Allocate Device Memory                                               //
			//======================================================================//
			// Allocate X(state) particles on Device Memory
			unsigned int size_X_particles = sizeof(double) * NP * SIZEOFX;
			double *device_X_particles;
			double *device_X_particles_copy;
			CUDA_CALL(cudaMalloc((void **) &device_X_particles , size_X_particles));
			CUDA_CALL(cudaMalloc((void **) &device_X_particles_copy , size_X_particles));
			// Allocate W(weight) particles on Device Memory
			unsigned int size_W_particles = sizeof(double) * NP;
			double *device_W_particles, *host_W_particles;
			double *device_W_particles_scanned, *device_W_particles_new;
			host_W_particles = (double *)malloc(size_W_particles);
			CUDA_CALL(cudaMalloc((void **) &device_W_particles , size_W_particles));
			CUDA_CALL(cudaMalloc((void **) &device_W_particles_scanned , size_W_particles));
			CUDA_CALL(cudaMalloc((void **) &device_W_particles_new , size_W_particles));

			//>>>>>>>>>>>>>>>>>>>>>Control particles>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
			// Allocate C(control) particles on Device Memory
			unsigned int size_C_particles = sizeof(double) * NP * SIZEOFC;
			double *device_C_particles;
			double *device_C_particles_copy;
			double *device_C_estimate, *host_C_estimate;
			CUDA_CALL(cudaMalloc((void **)&device_C_estimate, sizeof(double)*NP*2));
			host_C_estimate = (double *)malloc(sizeof(double)*NP*2);
			CUDA_CALL(cudaMalloc((void **) &device_C_particles , size_C_particles));
			CUDA_CALL(cudaMalloc((void **) &device_C_particles_copy , size_C_particles));
			// Allocate WC(weight of control) particles on Device Memory
			unsigned int size_WC_particles = sizeof(double) * NP;
			double *device_WC_particles, *host_WC_particles;
			double *device_WC_particles_scanned, *device_WC_particles_new;
			host_WC_particles = (double *)malloc(size_WC_particles);
			CUDA_CALL(cudaMalloc((void **) &device_WC_particles , size_WC_particles));
			CUDA_CALL(cudaMalloc((void **) &device_WC_particles_scanned , size_WC_particles));
			CUDA_CALL(cudaMalloc((void **) &device_WC_particles_new , size_WC_particles));
			//<<<<<<<<<<<<<<<<<<<<<<Control particles<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

			//======================================================================//
			// Allocate Device Memory for SCANFAN	     	                        //
			//======================================================================//
			double** g_scanBlockSums;
			int level = 0;
			int numEle = NP;
			while(numEle > BLOCK_SIZE){
				level ++;
				numEle = numEle / BLOCK_SIZE;
			}
			g_scanBlockSums = (double**) malloc(level * sizeof(double*));
			numEle = NP;
			level = 0;
			while(numEle > BLOCK_SIZE){
				numEle = numEle / BLOCK_SIZE;
				CUDA_CALL(cudaMalloc((void**) &g_scanBlockSums[level],  numEle * sizeof(double)));
				level ++;
			}
			//======================================================================//
			// Assign memory space to Index Array                                   //
			//======================================================================//
			unsigned int sizeMaxj = sizeof(int) * (NP + 1);
			int *device_maxj, *host_maxj;
			CUDA_CALL(cudaMalloc((void**)&device_maxj, sizeMaxj));
			CUDA_CALL(cudaMallocHost((void **)&host_maxj, sizeof(int) * (NP + 1)));
			host_maxj[NP] = NP - 1;
			CUDA_CALL(cudaMemcpy(device_maxj , host_maxj , sizeof(int) * (NP + 1) , 
						cudaMemcpyHostToDevice));
			//======================================================================//
			// Allocate memory for REDUCE               	                        //
			//======================================================================//   
			unsigned int reduceThreads, reduceBlocks;
			reduceThreads = (NP < BLOCK_SIZE*2) ? nextPow2((NP + 1)/ 2) : BLOCK_SIZE;
			reduceBlocks = (NP + (reduceThreads * 2 - 1)) / (reduceThreads * 2);
			if (reduceBlocks >= 8)
				reduceBlocks = reduceBlocks/8;
			if (reduceBlocks >= 4)
				reduceBlocks = reduceBlocks/4;
			if (reduceBlocks >= 2)
				reduceBlocks = reduceBlocks/2;
			unsigned int sizeStatesBlockSumArray = sizeof(double) * reduceBlocks * SIZEOFX;
			double *device_statesBlockSumArray;
			CUDA_CALL(cudaMalloc((void**)&device_statesBlockSumArray, sizeStatesBlockSumArray));
			//======================================================================//
			// Allocate memory for REDUCE (Control)        	                        //
			//======================================================================//   
			unsigned int sizeControlsBlockSumArray = sizeof(double) * reduceBlocks * 2;
			double *device_controlsBlockSumArray;
			CUDA_CALL(cudaMalloc((void**)&device_controlsBlockSumArray, sizeControlsBlockSumArray));
			//======================================================================//
			// Initialisation                                                       //
			//======================================================================//
			// Allocate space for rand numbers for Initialise
			double *host_rand_init = (double*) malloc(sizeof(double)*NP*SIZEOFX);
			double *host_rand_move = (double*) malloc(sizeof(double)*NP*4);
			double *device_rand_init, *device_rand_move;
			CUDA_CALL(cudaMalloc((void **) &device_rand_init, sizeof(double) * NP * SIZEOFX));
			CUDA_CALL(cudaMalloc((void **) &device_rand_move, sizeof(double) * NP * 4));
			for (int i = 0; i < SIZEOFX * NP; i++)
					host_rand_init[i] = distribution(generator);
			CUDA_CALL(cudaMemcpy(device_rand_init, host_rand_init,sizeof(double) * NP * SIZEOFX,
						cudaMemcpyHostToDevice));
			//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Control random numbers>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
			double *host_rand_control = (double*) malloc(sizeof(double)*NP*2);
			double *device_rand_control_init, *device_rand_control;
			CUDA_CALL(cudaMalloc((void **) &device_rand_control_init, sizeof(double) * NP * 2));
			CUDA_CALL(cudaMalloc((void **) &device_rand_control, sizeof(double) * NP * 2));
			//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<Control random numbers<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
			// Manually-produced data
			double Sdot_i = 4.0*hostParams->mu0*(hostParams->Vc+(hostParams->Ri+hostParams->Rg*y[0].G)
							*u[0].Ic)*u[0].Ic/hostParams->hs/pi/hostParams->De/hostParams->De;
			double delta_i = 7.0*hostParams->alphar*(0.5+hostParams->betam/3.0)/Sdot_i;
			// Initialise the particles and compute their weights
			Initialise<<< NP/BLOCK_SIZE, BLOCK_SIZE >>>
							(/* Input and also output */
							device_X_particles, 
							/* Inputs*/
							device_y,			device_u,			/* Observations and Inputs*/
							delta_i,			device_rand_init,	/* Random numbers*/
							NP,		/* Number of particles*/
							/* Output */
							device_W_particles);	
			//======================================================================//
			// Time-stepping routine						                        //
			//======================================================================//
			unsigned int iter = 0;
			StopWatchInterface *timer = NULL;
			cudaEvent_t start, stop;
			float GPUtime = 0.0;
			float time_part = 0.0f;

			float time_look = 0.0f;
			float time_scan = 0.0f;
			float time_normalise = 0.0f;
			float time_resampling = 0.0f;
			float time_propagation = 0.0f;
			float time_reduce = 0.0f;
			float time_MPC = 0.0f;

			float time_look_total = 0.0f;
			float time_scan_total = 0.0f;
			float time_normalise_total = 0.0f;
			float time_resampling_total = 0.0f;
			float time_propagation_total = 0.0f;
			float time_reduce_total = 0.0f;
			float time_MPC_total = 0.0f;

			float time_total = 0.0f;

			for (iter = 1; iter < 1187; iter ++)
			{
				printf("Iteration %d ......\n",iter);
				// Prepare device_y and device_u
				host_y[0] = y[iter].G;
				host_y[1] = y[iter].Xram;
				host_y[2] = y[iter].I;
				host_y[3] = y[iter].Me;
				host_y[4] = y[iter].Scl;
				host_y[5] = y[iter].Smr;
				host_y[6] = y[iter].phe;
				host_u[0] = u[iter-1].Ic;
				host_u[1] = u[iter-1].Vramc;
				CUDA_CALL(cudaMemcpy(device_y , host_y , sizeof(double) * SIZEOFY , cudaMemcpyHostToDevice));
				CUDA_CALL(cudaMemcpy(device_u , host_u , sizeof(double) * SIZEOFU , cudaMemcpyHostToDevice));
				// Prepare random numbers for particle propagation
				for (int i = 0; i < 4 * NP; i++)
					host_rand_move[i] = distribution(generator);
				CUDA_CALL(cudaMemcpy(device_rand_move, host_rand_move,sizeof(double) * NP * 4,cudaMemcpyHostToDevice));
				//======================================================================//
				// Start recording time							                        //
				//======================================================================//
				//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<//
				// Auxilliary Particle Filter             //                                       
				//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<//
				sdkCreateTimer(&timer);
				checkCudaErrors(cudaEventCreate(&start));
				checkCudaErrors(cudaEventCreate(&stop));
				sdkStartTimer(&timer);
				checkCudaErrors(cudaEventRecord(start, 0));
				// Lookahead resampling
				if(iter == 1)
					Lookahead_init<<< NP / BLOCK_SIZE , BLOCK_SIZE>>>(	/* Inputs*/
																device_X_particles,	
																device_y,			device_u,	/* Observations and Inputs*/
																NP,				/* Number of particles and iterations*/
																/* Input and Output*/
																device_W_particles);
				else 
					Lookahead<<< NP / BLOCK_SIZE , BLOCK_SIZE>>>(	/* Inputs*/
																device_X_particles,	
																device_y,			device_u,	/* Observations and Inputs*/
																NP,				/* Number of particles and iterations*/
																/* Input and Output*/
																device_W_particles);
				
				checkCudaErrors(cudaEventRecord(stop, 0));
				checkCudaErrors(cudaDeviceSynchronize());
				sdkStopTimer(&timer);
				checkCudaErrors(cudaEventElapsedTime(&time_look, start, stop));

				sdkCreateTimer(&timer);
				checkCudaErrors(cudaEventCreate(&start));
				checkCudaErrors(cudaEventCreate(&stop));
				sdkStartTimer(&timer);
				checkCudaErrors(cudaEventRecord(start, 0));
				
				// Do the RESAMPLING 
				scanFan<double>(device_W_particles_scanned , device_W_particles , NP , BLOCK_SIZE , 
									0 , g_scanBlockSums);
				checkCudaErrors(cudaEventRecord(stop, 0));
				checkCudaErrors(cudaDeviceSynchronize());
				sdkStopTimer(&timer);
				checkCudaErrors(cudaEventElapsedTime(&time_scan, start, stop));

				sdkCreateTimer(&timer);
				checkCudaErrors(cudaEventCreate(&start));
				checkCudaErrors(cudaEventCreate(&stop));
				sdkStartTimer(&timer);
				checkCudaErrors(cudaEventRecord(start, 0));
				
				normalize<<< NP / BLOCK_SIZE , BLOCK_SIZE>>>(device_W_particles_scanned, NP, device_maxj);	
				checkCudaErrors(cudaEventRecord(stop, 0));
				checkCudaErrors(cudaDeviceSynchronize());
				sdkStopTimer(&timer);
				checkCudaErrors(cudaEventElapsedTime(&time_normalise, start, stop));

				sdkCreateTimer(&timer);
				checkCudaErrors(cudaEventCreate(&start));
				checkCudaErrors(cudaEventCreate(&stop));
				sdkStartTimer(&timer);
				checkCudaErrors(cudaEventRecord(start, 0));
				
				resample<<< NP / BLOCK_SIZE , BLOCK_SIZE>>>(NP, device_X_particles, device_X_particles_copy, device_maxj);
				checkCudaErrors(cudaEventRecord(stop, 0));
				checkCudaErrors(cudaDeviceSynchronize());
				sdkStopTimer(&timer);
				checkCudaErrors(cudaEventElapsedTime(&time_resampling, start, stop));

				sdkCreateTimer(&timer);
				checkCudaErrors(cudaEventCreate(&start));
				checkCudaErrors(cudaEventCreate(&stop));
				sdkStartTimer(&timer);
				checkCudaErrors(cudaEventRecord(start, 0));
				
				//Propagate the particles
				propagation<<<NP / BLOCK_SIZE , BLOCK_SIZE>>>(/* Inputs*/
																device_X_particles_copy,	
																device_u,			/* Control*/
																device_rand_move,	/* Random numbers*/	
																NP,					/* Number of particles*/
																/* Output*/
																device_X_particles);
				checkCudaErrors(cudaEventRecord(stop, 0));
				checkCudaErrors(cudaDeviceSynchronize());
				sdkStopTimer(&timer);
				checkCudaErrors(cudaEventElapsedTime(&time_propagation, start, stop));

				sdkCreateTimer(&timer);
				checkCudaErrors(cudaEventCreate(&start));
				checkCudaErrors(cudaEventCreate(&stop));
				sdkStartTimer(&timer);
				checkCudaErrors(cudaEventRecord(start, 0));
				
				// Output maximum likelihood estimate
				reduce<double>(NP, reduceThreads, reduceBlocks, 6, device_X_particles, device_statesBlockSumArray , NP, reduceBlocks);
				reduce<double>(reduceBlocks, reduceThreads, 1, 6, device_statesBlockSumArray, device_statesBlockSumArray , reduceBlocks, reduceBlocks);
				calculateXe<<<1,64>>>(reduceBlocks, device_statesBlockSumArray, device_Xe);
				CUDA_CALL(cudaMemcpy(host_Xe , device_Xe , sizeof(double) * SIZEOFX , cudaMemcpyDeviceToHost));

				checkCudaErrors(cudaEventRecord(stop, 0));
				checkCudaErrors(cudaDeviceSynchronize());
				sdkStopTimer(&timer);
				checkCudaErrors(cudaEventElapsedTime(&time_reduce, start, stop));	
				//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<//
				// Model Predictive Controller            //                                       
				//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<//
				// Prepare random numbers for MPC
				for (int i = 0; i < 2 * NP; i++)
					host_rand_control[i] = distribution(generator);
				CUDA_CALL(cudaMemcpy(device_rand_control_init, host_rand_control, sizeof(double) * NP * 2, cudaMemcpyHostToDevice));
				for (int i = 0; i < 2 * NP; i++)
					host_rand_control[i] = distribution(generator);
				CUDA_CALL(cudaMemcpy(device_rand_control, host_rand_control, sizeof(double) * NP * 2, cudaMemcpyHostToDevice));
				

				checkCudaErrors(cudaEventCreate(&start));
				checkCudaErrors(cudaEventCreate(&stop));
				sdkStartTimer(&timer);
				checkCudaErrors(cudaEventRecord(start, 0));

				Initialise_controller<<<NP / BLOCK_SIZE , BLOCK_SIZE>>>(device_C_particles, 
																		device_X_particles, 
																		device_u, 
																		device_rand_control_init, 
																		NP);
				for (int i = 0; i < 15; i++){
					if (i > 0){
						//TODO: resampling
						// Do the RESAMPLING 
						scanFan<double>(device_WC_particles_scanned , device_WC_particles , NP , BLOCK_SIZE , 
									0 , g_scanBlockSums);
						normalize<<< NP / BLOCK_SIZE , BLOCK_SIZE>>>(device_WC_particles_scanned, NP, device_maxj);	
						resample_control<<< NP / BLOCK_SIZE , BLOCK_SIZE>>>(NP, device_C_particles_copy, device_C_particles, device_maxj);
					}
					// Prepare device_y and device_u
					host_y[0] = y[iter+i+1].G;
					host_y[1] = y[iter+i+1].Xram;
					host_y[2] = y[iter+i+1].I;
					host_y[3] = y[iter+i+1].Me;
					host_y[4] = y[iter+i+1].Scl;
					host_y[5] = y[iter+i+1].Smr;
					host_y[6] = y[iter+i+1].phe;
					CUDA_CALL(cudaMemcpy(device_y , host_y , sizeof(double) * SIZEOFY , cudaMemcpyHostToDevice));
					Propagation_controller<<<NP / BLOCK_SIZE , BLOCK_SIZE>>>(device_C_particles, 
																			device_y,
																			device_rand_control,	
																			NP, 
																			device_C_particles_copy,
																			device_WC_particles);
				}
				//TODO: reduce to compute the average
				// Output maximum likelihood estimate
				//reduce<double>(NP, reduceThreads, reduceBlocks, 6, device_C_particles_copy, device_controlsBlockSumArray , NP, reduceBlocks);
				//reduce<double>(reduceBlocks, reduceThreads, 1, 6, device_statesBlockSumArray, device_statesBlockSumArray , reduceBlocks, reduceBlocks);
				//calculateCe<<<1,64>>>(reduceBlocks, device_controlsBlockSumArray, device_Ce);

				CUDA_CALL(cudaMemcpy(host_WC_particles,device_WC_particles, sizeof(double)*NP, cudaMemcpyDeviceToHost));
				extract_control<<<NP/BLOCK_SIZE, BLOCK_SIZE>>>(device_C_estimate, device_C_particles, NP);
				CUDA_CALL(cudaMemcpy(host_C_estimate,device_C_estimate, sizeof(double)*NP*2, cudaMemcpyDeviceToHost));
				double sum = 0;
				double I_estimate = 0;
				double V_estimate = 0;
				for (int i = 0; i < NP; i++)
					sum += host_WC_particles[i];
				for (int i = 0; i < NP; i++){
					host_WC_particles[i] = host_WC_particles[i]/sum;
					I_estimate +=host_WC_particles[i]*host_C_estimate[i];
					V_estimate +=host_WC_particles[i]*host_C_estimate[i+NP];
				}

				checkCudaErrors(cudaEventRecord(stop, 0));
				checkCudaErrors(cudaDeviceSynchronize());
				sdkStopTimer(&timer);
				checkCudaErrors(cudaEventElapsedTime(&time_MPC, start, stop));
				//======================================================================//
				// End recording time							                        //
				//======================================================================//
				//GPUtime += time_part;

				time_look_total += time_look;
				time_scan_total += time_scan;
				time_normalise_total += time_normalise;
				time_resampling_total += time_resampling;
				time_propagation_total += time_propagation;
				time_reduce_total += time_reduce;
				time_MPC_total += time_MPC;

				//Record Estimation
				for (int i = 0; i < SIZEOFX; i++)
					fprintf(fid_Estimation, "%12.6f\t",host_Xe[i]/NP);
				fprintf(fid_Estimation, "\n");

				fprintf(fid_control,"%lf\t%lf\n",I_estimate, V_estimate);
			} // end for (iter = 1; iter < iteration; iter ++)
			time_total = time_look_total + time_scan_total + time_normalise_total + time_resampling_total
						+ time_propagation_total + time_reduce_total + time_MPC_total;
			//fprintf(fid_time, "%lf\t",GPUtime/(iter - 1.0));
			//printf("Time : %lf\n",GPUtime/(iter - 1.0));
			//printf("\nNumber of particles: %d\n",NP);
			//printf("Look Time: \t%f\n", time_look_total/(iter - 1.0));
			//printf("Scan Time: \t%f\n", time_scan_total/(iter - 1.0));
			//printf("Normalise Time: \t%f\n", time_normalise_total/(iter - 1.0));
			//printf("Resampling Time: \t%f\n", time_resampling_total/(iter - 1.0));
			//printf("resample time: \t%f\n", time_scan_total/(iter - 1.0)+time_normalise_total/(iter - 1.0)+time_resampling_total/(iter - 1.0));
			//printf("propagation Time: \t%f\n", time_propagation_total/(iter - 1.0));
			//printf("Reduce Time: \t%f\n", time_reduce_total/(iter - 1.0));
			printf("Total time per iteration: \t%f\n",time_total/(iter-1.0));

			//======================================================================//
			// Free Memory space on both Host and Device                            //
			//======================================================================//
			free(host_W_particles);
			CUDA_CALL(cudaFree(device_X_particles));
			CUDA_CALL(cudaFree(device_X_particles_copy));
			//CUDA_CALL(cudaFree(device_Y_particles));
			CUDA_CALL(cudaFree(device_W_particles));
			CUDA_CALL(cudaFree(device_W_particles_scanned));
			CUDA_CALL(cudaFree(device_W_particles_new));
			CUDA_CALL(cudaFree(device_rand_init));
			for (int i = 0; i < level; i++) CUDA_CALL(cudaFree(g_scanBlockSums[i]));
			free((void**)g_scanBlockSums);
			CUDA_CALL(cudaFree(device_maxj));
			CUDA_CALL(cudaFreeHost(host_maxj));
			CUDA_CALL(cudaFree(device_statesBlockSumArray));	
			free(host_rand_init);
			free(host_rand_move);
			CUDA_CALL(cudaFree(device_rand_move));
			fclose(fid_Estimation);
			CUDA_CALL(cudaFree(device_C_particles));
			CUDA_CALL(cudaFree(device_C_particles_copy));
			CUDA_CALL(cudaFree(device_WC_particles));
			CUDA_CALL(cudaFree(device_WC_particles_scanned));
			CUDA_CALL(cudaFree(device_WC_particles_new));
			free(host_WC_particles);
			CUDA_CALL(cudaFree(device_controlsBlockSumArray));
			free(host_rand_control);
			CUDA_CALL(cudaFree(device_rand_control));
			CUDA_CALL(cudaFree(device_rand_control_init));
			CUDA_CALL(cudaFree(device_C_estimate));
			free(host_C_estimate);

//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>//
//	The code above is relevant to NP
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>//
	free(host_Xe);
	CUDA_CALL(cudaFree(device_Xe));
	CUDA_CALL(cudaFree(device_y));
	CUDA_CALL(cudaFree(device_u));
	free(host_y);
	free(host_u);
	free(hostParams);
	fclose(fid_control);


	fclose(fid_time);
	return 0;
}



bool InitCUDA()
{
	int count = 0;
	int i = 0;
	cudaGetDeviceCount(&count);
	if(count == 0) {
		fprintf(stderr, "There is no device.\n");
		return false;
	}
	cudaDeviceProp prop;
	for(i = 0; i < count; i++) {	
		if(cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
			if(prop.major > DEVICE_CAPABILITY_MAJOR) break;
			else if(prop.major == DEVICE_CAPABILITY_MAJOR && prop.minor >= DEVICE_CAPABILITY_MINOR) break;
		}
	}
	if(i == count) {
		fprintf(stderr, "There is no device supporting CUDA.\n");
		return false;
	}
	cudaSetDevice(i);
//	printf("CUDA Device initialized.\n");
//	printf ("Running on device %s with compute capability %d.%d\n", prop.name,prop.major,prop.minor);
	return true;
}

void ReadConfig(SimParams *hostParams)
{
	hostParams->De = 17.0*2.54; //Electrode diameter (in)
	hostParams->Di = 20.0*2.54; //Ingot diameter (in)
	hostParams->I0 = 6000; //Nominal current (A)
	hostParams->G0 = 0.36*2.54; //Nominal gap (in)
	hostParams->phe0 = 3.0; //Nominal helium pressure (Torr)
	hostParams->mu0 = 0.440087; //Nominal melting efficiency
	hostParams->Vc = 21.18; //Cathode voltage fall (V)
	hostParams->Ri = 4.37e-4; //Gap-independent electric resistance (Ohm)
	hostParams->Rg = 0.0; //Gap-dependent electric resistance (Ohm/cm)
	hostParams->Scl0 = 15.702565; //Nominal centerline pool depth (cm)
	hostParams->Smr0 = 13.234694; //Nominal mid-radius pool depth (cm)
	hostParams->Acl = 1.909e-3; //A matrix (1)
	hostParams->Amr = 1.443e-3; //
	hostParams->Bdeltacl = 2.60e-5; //Bdelta matrix (1)
	hostParams->Bdeltamr = -1.29e-4; //
	hostParams->Bicl = 6.587e-6; //Bi matrix (cm/A)
	hostParams->Bimr = 3.165e-6; //
	hostParams->Bmucl = 6.899e-2; //Bmu matrix (cm)
	hostParams->Bmumr = 2.686e-2; //
	hostParams->Bhecl = -8.091e-4; //Bhe matrix (cm/Torr)
	hostParams->Bhemr = -6.541e-4; //
	hostParams->sigmaI = 20.0; //Current standard deviation (A)
	hostParams->sigmaVram = 5.0e-4; //Ram speed standard deviation (cm/s)
	hostParams->sigmamur = 1.0e-2; //Melting efficiency standard deviation (1)
	hostParams->sigmaa = 1.0e-3; //Fill ratio standard deviation (1)
	hostParams->sigmaVb = 1.0e-3; //Voltage bias standard deviation
	hostParams->sigmaIb = 1.0e-3; //Current bias standard deviation
	hostParams->sigmaVramb = 1.0e-3; //Ram speed bias standard deviation
	hostParams->sigmahe = 1.0e-3; //Helium pressure standard deviation
	hostParams->sigmaG = 0.2; //Measured electrode gap standard deviation (cm)
	hostParams->sigmaPos = 0.005; //Measured ram position standard deviation (cm)
	hostParams->sigmaImeas = 15.0; //Measured current standard deviation (A)
	hostParams->sigmaLC = 200.0; //Measured load cell standard deviation (g)
	hostParams->sigmaVmeas = 0.1; //Measured voltage standard deviation (V)
	hostParams->sigmaCL = 1.0; //Measured centerline pool depth standard deviation (cm)
	hostParams->sigmaMR = 1.0; //Measured mid-radius pool depth standard deviation (cm)
	hostParams->sigmahemeas = 1.0e-2; //Measured helium pressure standard deviation (Torr)
	hostParams->dt = 6; //Time step (s)
	// Other global variables
	hostParams->alphar = 0.023821;
	hostParams->alpham = 0.059553;
	hostParams->hr = 0.000000;
	hostParams->rhor = 7.750000;
	hostParams->hm = 698.750000 * hostParams->rhor;
	hostParams->hs = 1038.750000 * hostParams->rhor;
	double Lambda = (hostParams->hs-hostParams->hm)/hostParams->hm;
	hostParams->a0 = 1-(hostParams->De/hostParams->Di)*(hostParams->De/hostParams->Di);
	double V0 = hostParams->Vc + (hostParams->Ri+hostParams->Rg*hostParams->G0)*hostParams->I0;
	double Pm0 = hostParams->mu0*V0*hostParams->I0;
	double den = 11.0*Lambda+3.0;
	hostParams->betam = (hostParams->alpham-hostParams->alphar)/hostParams->alphar;
	hostParams->Cdd = 224.0*(Lambda+1.0)*(0.5+hostParams->betam/3.0)/den;
	hostParams->Cdp = 32.0/den;
	hostParams->Csd = 56.0*(0.5+hostParams->betam/3.0)/den;
	hostParams->Csp = 11.0/den;
	double mdot0, Sdot0;
	mdot0 = Pm0*hostParams->rhor/hostParams->hs;
	Sdot0 = 4*mdot0/hostParams->rhor/pi/hostParams->De/hostParams->De;
	hostParams->delta0 = 7.0*hostParams->alphar*(0.5+hostParams->betam/3.0)/Sdot0;
	hostParams->Vram0 = 4*hostParams->a0*mdot0/hostParams->rhor/pi/hostParams->De/hostParams->De;
	hostParams->sigmamur *= hostParams->mu0;
	hostParams->sigmaa *= hostParams->a0;
	hostParams->sigmaVb *= V0;
	hostParams->sigmaIb *= hostParams->I0;
	hostParams->sigmaVramb *= hostParams->Vram0;
	hostParams->sigmahe *= hostParams->phe0;

	hostParams->epsilon = 1.0e-10;
	hostParams->var_delta0 = 5.0;
//	hostParams->var_G0 = 1.0;
//	hostParams->var_Xram0 = 0.1;
//	hostParams->var_Me0 = 1.0;
//	hostParams->sigma_pooldepth = 0.01;
	hostParams->G11 = -4.0*hostParams->Cdp*hostParams->Vc*hostParams->mu0/pi/hostParams->De/hostParams->De/hostParams->hm - 8.0*hostParams->Cdp*hostParams->Ri*hostParams->mu0*hostParams->I0/pi/hostParams->De/hostParams->De/hostParams->hm;
	hostParams->G21 = 4.0*hostParams->Csp*hostParams->a0*hostParams->mu0*hostParams->Vc/pi/hostParams->De/hostParams->De/hostParams->hm + 8.0*hostParams->Csp*hostParams->a0*hostParams->mu0*hostParams->Ri*hostParams->I0/pi/hostParams->De/hostParams->De/hostParams->hm;
	hostParams->G41 = -hostParams->rhor*hostParams->Csp*hostParams->Vc*hostParams->mu0/hostParams->hm - 2.0*hostParams->rhor*hostParams->Csp*hostParams->Ri*hostParams->mu0*hostParams->I0/hostParams->hm;
}

long load_data(char const * szName, cv_obs** yp, cv_inputs** up)
{
  FILE * fObs = fopen(szName,"rt");
  if (fObs==NULL) {fputs ("File error: fObs",stderr); exit (1);}
  char* szBuffer = new char[1024];
  if ( fgets(szBuffer, 1024, fObs) == NULL) {
    perror("Need total number of observations");
    return 0;
  }
  long lIterates = strtol(szBuffer, NULL, 10);
  char * pch ;

  *yp = new cv_obs[lIterates];
  *up = new cv_inputs[lIterates];
  
  for(long i = 0; i < lIterates; ++i)
    {
      if ( fgets(szBuffer, 1024, fObs) == NULL ) {
	perror("Not enough lines");
	return 0;
      }
      pch = strtok(szBuffer, ",\r\n\t ");
      (*yp)[i].Time = strtod(pch, NULL);
      //pch = strtok(NULL, ",\r\n ");
      //(*yp)[i].V = strtod(pch, NULL);
      pch = strtok(NULL, ",\r\n\t ");
      (*yp)[i].I = strtod(pch, NULL);
      pch = strtok(NULL, ",\r\n\t ");
      (*yp)[i].Me = strtod(pch, NULL);
      pch = strtok(NULL, ",\r\n\t ");
      (*yp)[i].Xram = strtod(pch, NULL);
      pch = strtok(NULL, ",\r\n\t ");
      (*yp)[i].G = strtod(pch, NULL);
      pch = strtok(NULL, ",\r\n\t ");
      (*up)[i].Ic = strtod(pch, NULL);
      pch = strtok(NULL, ",\r\n\t ");
      (*up)[i].Vramc = strtod(pch, NULL);
      pch = strtok(NULL, ",\r\n\t ");
      (*yp)[i].Smr = strtod(pch, NULL);
      pch = strtok(NULL, ",\r\n\t ");
      (*yp)[i].Scl = strtod(pch, NULL);
      pch = strtok(NULL, ",\r\n\t ");
      (*yp)[i].phe = strtod(pch, NULL);
    }
  fclose(fObs);

  delete [] szBuffer;

  return lIterates;
}