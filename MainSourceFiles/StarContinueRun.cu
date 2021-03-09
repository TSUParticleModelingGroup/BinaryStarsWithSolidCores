/*
nvcc StarContinueRun.cu -o StarContinueRun.exe -lglut -lGL -lGLU -lm
nvcc StarContinueRun.cu -o StarContinueRun.exe -lglut -lGL -lGLU -lm --use_fast_math
*/

#include "../CommonCompileFiles/binaryStarCommonIncludes.h"
#include "../CommonCompileFiles/binaryStarCommonDefines.h"
#include "../CommonCompileFiles/binaryStarCommonGlobals.h"
#include "../CommonCompileFiles/binaryStarCommonFunctions.h"
#include "../CommonCompileFiles/binaryStarCommonRunGlobals.h"
#include "../CommonCompileFiles/binaryStarCommonRunFunctions.h"

//Time to add on to the run. Readin from the comand line.
float ContinueRunTime;

void openAndReadFiles()
{
	ifstream data;
	string name;
	
	//Opening the positions and velosity file to dump stuff to make movies out of. Need to move to the end of the file.
	PosAndVelFile = fopen("PosAndVel", "rb+");
	if(PosAndVelFile == NULL)
	{
		printf("\n\n The PosAndVel file does not exist\n\n");
		exit(0);
	}
	fseek(PosAndVelFile,0,SEEK_END);
	
	//Reading in the run parameters
	data.open("RunParameters");
	if(data.is_open() == 1)
	{
		getline(data,name,'=');
		data >> SystemLengthConverterToKilometers;
		
		getline(data,name,'=');
		data >> SystemMassConverterToKilograms;
		
		getline(data,name,'=');
		data >> SystemTimeConverterToSeconds;
		
		getline(data,name,'=');
		data >> NumberElementsStar1;
		
		getline(data,name,'=');
		data >> NumberElementsStar2;
		
		getline(data,name,'=');
		data >> CorePushBackReduction;
		
		getline(data,name,'=');
		data >> PlasmaPushBackReduction;
	}
	else
	{
		printf("\nTSU Error could not open RunParameters file\n");
		exit(0);
	}
	data.close();
	NumberElements = NumberElementsStar1 + NumberElementsStar2;
	ContinueRunTime *=((24.0*60.0*60.0)/SystemTimeConverterToSeconds);
	
	//Reading in the run parameters
	data.open("BranchRunParameters");
	if(data.is_open() == 1)
	{	
		getline(data,name,'=');
		data >> RecordRate;
		
		getline(data,name,'=');
		data >> DrawRate;
	}
	else
	{
		printf("\nTSU Error could not open BranchRunParameters file\n");
		exit(0);
	}
	data.close();
}

void readInTheInitialsStars()
{
	FILE *startFile = fopen("FinalPosVelForce","rb");
	if(startFile == NULL)
	{
		printf("\n\n The FinalPosVelForce file does not exist\n\n");
		exit(0);
	}
	fread(&StartTime, sizeof(float), 1, startFile);
	fread(PosCPU, sizeof(float4), NumberElements, startFile);
	fread(VelCPU, sizeof(float4), NumberElements, startFile);
	fread(ForceCPU, sizeof(float4), NumberElements, startFile);
	fclose(startFile);
}

__global__ void getForces(float4 *pos, float4 *vel, float4 *force, int numberElementsStar1, int numberOfElements, float corePushBackReduction, float plasmaPushBackReduction, int gPUNumber, int gPUsUsed)
{
	int id, ids, i, j, k;
	float4 posMe, velMe, forceMe;
	float4 partialForce;
	double forceSumX, forceSumY, forceSumZ;
	
	__shared__ float4 shPos[BLOCKSIZE];
	__shared__ float4 shVel[BLOCKSIZE];
	__shared__ float4 shForce[BLOCKSIZE];

	id = threadIdx.x + blockDim.x*blockIdx.x + blockDim.x*gridDim.x*gPUNumber;
	if(numberOfElements <= id)
	{
		printf("\n TSU error: id out of bounds in getForces. \n");
	}
		
	forceSumX = 0.0;
	forceSumY = 0.0;
	forceSumZ = 0.0;
		
	posMe.x = pos[id].x;
	posMe.y = pos[id].y;
	posMe.z = pos[id].z;
	posMe.w = pos[id].w;
	
	velMe.x = vel[id].x;
	velMe.y = vel[id].y;
	velMe.z = vel[id].z;
	velMe.w = vel[id].w;
	
	forceMe.x = force[id].x;
	forceMe.y = force[id].y;
	forceMe.z = force[id].z;
	forceMe.w = force[id].w;
	
	for(k =0; k < gPUsUsed; k++)
	{
		for(j = 0; j < gridDim.x; j++)
		{
			shPos[threadIdx.x]   = pos  [threadIdx.x + blockDim.x*j + blockDim.x*gridDim.x*k];
			shVel[threadIdx.x]   = vel  [threadIdx.x + blockDim.x*j + blockDim.x*gridDim.x*k];
			shForce[threadIdx.x] = force[threadIdx.x + blockDim.x*j + blockDim.x*gridDim.x*k];
			__syncthreads();
		   
			#pragma unroll 32
			for(i = 0; i < blockDim.x; i++)	
			{
				ids = i + blockDim.x*j + blockDim.x*gridDim.x*k;
				if(id != ids)
				{
					if(id == 0 && ids == numberElementsStar1)
					{
						partialForce = calculateCoreCoreForce(posMe, shPos[i], velMe, shVel[i], forceMe, shForce[i], corePushBackReduction);
					}
					else if(id == numberElementsStar1 && ids == 0)
					{
						partialForce = calculateCoreCoreForce(posMe, shPos[i], velMe, shVel[i], forceMe, shForce[i], corePushBackReduction);
					}
					else if(id == 0 || id == numberElementsStar1)
					{
						partialForce = calculateCorePlasmaForce(0, posMe, shPos[i], velMe, shVel[i], forceMe, shForce[i], corePushBackReduction);
					}
					else if(ids == 0 || ids == numberElementsStar1)
					{
						partialForce = calculateCorePlasmaForce(1, posMe, shPos[i], velMe, shVel[i], forceMe, shForce[i], corePushBackReduction);
					}
					else
					{
						partialForce = calculatePlasmaPlasmaForce(posMe, shPos[i], velMe, shVel[i], plasmaPushBackReduction);
					}
					forceSumX += partialForce.x;
					forceSumY += partialForce.y;
					forceSumZ += partialForce.z;
				}
			}
			__syncthreads();
		}
	}
	
	force[id].x = (float)forceSumX;
	force[id].y = (float)forceSumY;
	force[id].z = (float)forceSumZ;
}

__global__ void moveBodies(float4 *pos, float4 *vel, float4 *force, float dt, int gPUNumber)
{  
    	int id = threadIdx.x + blockDim.x*blockIdx.x + blockDim.x*gridDim.x*gPUNumber;

	vel[id].x += (force[id].x/pos[id].w)*dt;
	vel[id].y += (force[id].y/pos[id].w)*dt;
	vel[id].z += (force[id].z/pos[id].w)*dt;

	pos[id].x += vel[id].x*dt;
	pos[id].y += vel[id].y*dt;
	pos[id].z += vel[id].z*dt;
}

float starNbody(float time, float runTime, float dt, int gPUsUsed)
{ 
	int   	tDraw = 0;
	int   	tRecord = 0;
	int 	tBackup = 0;
	int 	backupRate = 1000;
	
	while(time < runTime)
	{	
		int offSet = NumberElements/gPUsUsed;
		
		//Getting forces
		for(int i = 0; i < gPUsUsed; i++)
		{
			cudaSetDevice(i);
			errorCheck("cudaSetDevice");
			getForces<<<GridConfig, BlockConfig>>>(PosGPU[i], VelGPU[i], ForceGPU[i], NumberElementsStar1, NumberElements, CorePushBackReduction, PlasmaPushBackReduction, i, gPUsUsed);
			errorCheck("getForces");
		}
		
		//Moving elements
		for(int i = 0; i < gPUsUsed; i++)
		{
			cudaSetDevice(i);
			errorCheck("cudaSetDevice");
			moveBodies<<<GridConfig, BlockConfig>>>(PosGPU[i], VelGPU[i], ForceGPU[i], dt, i);
			errorCheck("moveBodies");
		}
		cudaDeviceSynchronize();
		errorCheck("cudaDeviceSynchronize");
		
		//Sharing memory		
		for(int i = 0; i < gPUsUsed; i++)
		{
			cudaSetDevice(i);
			errorCheck("cudaSetDevice");
			for(int j = 0; j < gPUsUsed; j++)
			{
				if(i != j)
				{
					cudaMemcpyAsync(&PosGPU[j][i*offSet], &PosGPU[i][i*offSet], (NumberElements/gPUsUsed)*sizeof(float4), cudaMemcpyDeviceToDevice);
					errorCheck("cudaMemcpy Pos");
				
					cudaMemcpyAsync(&VelGPU[j][i*offSet], &VelGPU[i][i*offSet], (NumberElements/gPUsUsed)*sizeof(float4), cudaMemcpyDeviceToDevice);
					errorCheck("cudaMemcpy Vel");
				}
			}
		}
		cudaDeviceSynchronize();
		errorCheck("cudaDeviceSynchronize");
		
		if(tDraw == DrawRate) 
		{
			//Because it is shared above it will only need to be copied from one GPU.
			cudaSetDevice(0);
			errorCheck("cudaSetDevice");
			cudaMemcpy(PosCPU, PosGPU[0], (NumberElements)*sizeof(float4), cudaMemcpyDeviceToHost);
			errorCheck("cudaMemcpy Pos draw");
			drawPicture();
			tDraw = 0;
			printf("\n Time in days = %f", time*SystemTimeConverterToSeconds/(60.0*60.0*24.0)); 
		}
		if(tRecord == RecordRate) 
		{
			//Because it is shared above it will only need to be copied from one GPU.
			cudaSetDevice(0);
			errorCheck("cudaSetDevice");
			cudaMemcpy(PosCPU, PosGPU[0], (NumberElements)*sizeof(float4), cudaMemcpyDeviceToHost);
			errorCheck("cudaMemcpy Pos record");
			cudaMemcpy(VelCPU, VelGPU[0], (NumberElements)*sizeof(float4), cudaMemcpyDeviceToHost);
			errorCheck("cudaMemcpy Vel record");
			recordPosAndVel(time);
			tRecord = 0;
		}
		if(tBackup == backupRate) 
		{
			//Because it is shared above it will only need to be copied from one GPU.
			//Saving the the runs positions, velosities and forces incase the system crashes in the middle of a run	
			copyStarsDownFromGPU();
			recordFinalPosVelForceStars(time);
			tBackup = 0;
		}
		
		tDraw++;
		tRecord++;
		tBackup++;
		time += dt;
	}
	return(time - dt);
}

void control()
{	
	struct sigaction sa;
	float time = StartTime;
	clock_t startTimer, endTimer;
	int gPUsUsed;
	
	//Starting the timer.
	startTimer = clock();
	
	// Handling input from the screen.
	sa.sa_handler = signalHandler;
	sigemptyset(&sa.sa_mask);
	sa.sa_flags = SA_RESTART; // Restart functions if interrupted by handler
	if (sigaction(SIGINT, &sa, NULL) == -1)
	{
		printf("\nTSU Error: sigaction error\n");
	}
	
	// Reading in the build parameters.
	printf("\n Reading and setting the run parameters.\n");
	openAndReadFiles();
	
	// Allocating memory for CPU and GPU.
	printf("\n Allocating memory on the GPU and CPU and opening positions and velocities file.\n");
	allocateCPUMemory();
	
	// Reading in the raw stars generated by the build program.
	printf("\n Reading in the stars that were generated in the build program.\n");
	readInTheInitialsStars();
	
	// Draw the intial configuration.
	printf("\n Drawing initial picture.\n");
	drawPicture();
	
	// Seting up the GPU.
	printf("\n Setting up the GPU.\n");
	gPUsUsed = deviceSetup();
	
	// Running the simulation.
	printf("\n Running the simulation.\n");
	copyStarsUpToGPU(gPUsUsed);
	time = starNbody(StartTime, StartTime + ContinueRunTime, DT, gPUsUsed);
	
	// Saving the the runs final positions and velosities.	
	printf("\n Saving the the runs final positions and velosities.\n");
	copyStarsDownFromGPU();
	recordFinalPosVelForceStars(time);  
	
	// Saving any wanted stats about the run that you may want. I don't have anything to record as of yet.
	printf("\n Saving any wanted stats about the run that you may want.\n");
	//recordStarStats();	
	
	// Freeing memory. 	
	printf("\n Cleaning up the run.\n");
	cleanUp(gPUsUsed);
	fclose(PosAndVelFile);

	// Stopping timer and printing out run time.
	endTimer = clock();
	int seconds = (endTimer - startTimer)/CLOCKS_PER_SEC;
	int hours = seconds/3600;
	int minutes = (seconds - hours*3600)/60;
	seconds = seconds - hours*3600 - minutes*60;
   	printf("\n Total time taken for this run: %d hours %d minutes %d seconds\n", hours, minutes, seconds);

	printf("\n The run has finished successfully \n\n");
	exit(0);
}

int main(int argc, char** argv)
{ 
	if( argc < 2)
	{
		printf("\n You need to intire an amount of time to add to the run on the comand line\n");
		exit(0);
	}
	else
	{
		ContinueRunTime = atof(argv[1]); //Reading time in as days. Need to put in our units after paranter file is read in.
	}

	//Globals for setting up the viewing window 
	int xWindowSize = 2500;
	int yWindowSize = 2500; 
	
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB);
	glutInitWindowSize(xWindowSize,yWindowSize);
	glutInitWindowPosition(0,0);
	glutCreateWindow("Creating Stars");
	
	glutReshapeFunc(reshape);
	
	init();
	
	glShadeModel(GL_SMOOTH);
	glClearColor(0.0, 0.0, 0.0, 0.0);
	
	glutDisplayFunc(Display);
	glutReshapeFunc(reshape);
	glutIdleFunc(control);
	glutMainLoop();
	return 0;
}

