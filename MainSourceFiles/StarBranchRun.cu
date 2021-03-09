/*
nvcc StarBranchRun.cu -o StarBranchRun.exe -lglut -lGL -lGLU -lm
nvcc StarBranchRun.cu -o StarBranchRun.exe -lglut -lGL -lGLU -lm --use_fast_math
*/

#include "../CommonCompileFiles/binaryStarCommonIncludes.h"
#include "../CommonCompileFiles/binaryStarCommonDefines.h"
#include "../CommonCompileFiles/binaryStarCommonGlobals.h"
#include "../CommonCompileFiles/binaryStarCommonFunctions.h"
#include "../CommonCompileFiles/binaryStarCommonRunGlobals.h"
#include "../CommonCompileFiles/binaryStarCommonRunFunctions.h"

//FIle to hold the branch run parameters.
//FILE *StartPosVelForceFile;
FILE *BranchRunParameters;

//Globals read in from the BranchSetup file.
float BranchRunTime;
float GrowthStartTimeStar1, GrowthStopTimeStar1, PercentForceIncreaseStar1;
float GrowthStartTimeStar2, GrowthStopTimeStar2, PercentForceIncreaseStar2;
float4 InitailPosStar1, InitailVelStar1;
float4 InitailPosStar2, InitailVelStar2;

void createAndLoadFolderForNewBranchRun()
{   	
	//Create output folder to store the branch run
	time_t t = time(0); 
	struct tm * now = localtime( & t );
	int month = now->tm_mon + 1, day = now->tm_mday, curTimeHour = now->tm_hour, curTimeMin = now->tm_min;
	stringstream smonth, sday, stimeHour, stimeMin;
	smonth << month;
	sday << day;
	stimeHour << curTimeHour;
	stimeMin << curTimeMin;
	string monthday;
	if (curTimeMin <= 9)	monthday = smonth.str() + "-" + sday.str() + "-" + stimeHour.str() + ":0" + stimeMin.str();
	else			monthday = smonth.str() + "-" + sday.str() + "-" + stimeHour.str() + ":" + stimeMin.str();
	string foldernametemp = "BranchRun:" + monthday;
	
	const char *branchFolderName = foldernametemp.c_str();
	//char *branchFolderName = foldernametemp;
	mkdir(branchFolderName , S_IRWXU|S_IRWXG|S_IRWXO);
	
	//Copying files into the branch folder
	FILE *fileIn;
	FILE *fileOut;
	long sizeOfFile;
  	char *buffer;
	
	//Copying files from the main star folder into the branch folder
	chdir(branchFolderName);
		
	fileIn = fopen("../BranchSetup", "rb");
	if(fileIn == NULL)
	{
		printf("\n\n The BranchSetup file does not exist\n\n");
		exit(0);
	}
	fseek (fileIn , 0 , SEEK_END);
  	sizeOfFile = ftell(fileIn);
  	rewind (fileIn);
  	buffer = (char*)malloc(sizeof(char)*sizeOfFile);
  	fread (buffer, 1, sizeOfFile, fileIn);
	fileOut = fopen("BranchSetup", "wb");
	fwrite (buffer, 1, sizeOfFile, fileOut);
	fclose(fileIn);
	fclose(fileOut);
		
	fileIn = fopen("../ContinueFiles/ContinueRun", "rb");
	if(fileIn == NULL)
	{
		printf("\n\n The ContinueFiles/ContinueRun file does not exist\n\n");
		exit(0);
	}
	fseek (fileIn , 0 , SEEK_END);
  	sizeOfFile = ftell(fileIn);
  	rewind (fileIn);
  	buffer = (char*)malloc(sizeof(char)*sizeOfFile);
  	fread (buffer, 1, sizeOfFile, fileIn);
	fileOut = fopen("ContinueRun", "wb");
	fwrite (buffer, 1, sizeOfFile, fileOut);
	fclose(fileIn);
	fclose(fileOut);
	system("chmod 755 ./ContinueRun");
	
	fileIn = fopen("../FilesFromBuild/RunParameters", "rb");
	if(fileIn == NULL)
	{
		printf("\n\n The FilesFromBuild/RunParameters file does not exist\n\n");
		exit(0);
	}
	fseek (fileIn , 0 , SEEK_END);
  	sizeOfFile = ftell(fileIn);
  	rewind (fileIn);
  	buffer = (char*)malloc(sizeof(char)*sizeOfFile);
  	fread (buffer, 1, sizeOfFile, fileIn);
	fileOut = fopen("RunParameters", "wb");
	fwrite (buffer, 1, sizeOfFile, fileOut);
	fclose(fileIn);
	fclose(fileOut);
	
	//Creating the positions and velosity file to dump to stuff to make movies out of.
	PosAndVelFile = fopen("PosAndVel", "wb");
	
	//Creating file to hold the ending time Pos vel and forces to continue the run.
	//StartPosVelForceFile = fopen("StartPosVelForce", "wb");
	
	//Creating the BranchRunParameter file.
	BranchRunParameters = fopen("BranchRunParameters", "wb");
	
	free (buffer);
}

void readAndSetRunParameters()
{
	ifstream data;
	string name;
	   
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
}

void readAndSetBranchParameters()
{
	ifstream data;
	string name;
	
	data.open("BranchSetup");
	
	if(data.is_open() == 1)
	{
		getline(data,name,'=');
		data >> InitailPosStar1.x;
		getline(data,name,'=');
		data >> InitailPosStar1.y;
		getline(data,name,'=');
		data >> InitailPosStar1.z;
		
		getline(data,name,'=');
		data >> InitailPosStar2.x;
		getline(data,name,'=');
		data >> InitailPosStar2.y;
		getline(data,name,'=');
		data >> InitailPosStar2.z;
		
		getline(data,name,'=');
		data >> InitailVelStar1.x;
		getline(data,name,'=');
		data >> InitailVelStar1.y;
		getline(data,name,'=');
		data >> InitailVelStar1.z;
		
		getline(data,name,'=');
		data >> InitailVelStar2.x;
		getline(data,name,'=');
		data >> InitailVelStar2.y;
		getline(data,name,'=');
		data >> InitailVelStar2.z;
		
		getline(data,name,'=');
		data >> BranchRunTime;
		
		getline(data,name,'=');
		data >> GrowthStartTimeStar1;
		
		getline(data,name,'=');
		data >> GrowthStopTimeStar1;
		
		getline(data,name,'=');
		data >> PercentForceIncreaseStar1;
		
		getline(data,name,'=');
		data >> GrowthStartTimeStar2;
		
		getline(data,name,'=');
		data >> GrowthStopTimeStar2;
		
		getline(data,name,'=');
		data >> PercentForceIncreaseStar2;
		
		getline(data,name,'=');
		data >> RecordRate;
		
		getline(data,name,'=');
		data >> DrawRate;
	}
	else
	{
		printf("\nTSU Error could not open BranchSetup file\n");
		exit(0);
	}
	data.close();
	
	//Taking input positions into our units
	InitailPosStar1.x /= SystemLengthConverterToKilometers;
	InitailPosStar1.y /= SystemLengthConverterToKilometers;
	InitailPosStar1.z /= SystemLengthConverterToKilometers;

	InitailPosStar2.x /= SystemLengthConverterToKilometers;
	InitailPosStar2.y /= SystemLengthConverterToKilometers;
	InitailPosStar2.z /= SystemLengthConverterToKilometers;

	//Taking input velocities into our units
	InitailVelStar1.x /= (SystemLengthConverterToKilometers/SystemTimeConverterToSeconds);
	InitailVelStar1.y /= (SystemLengthConverterToKilometers/SystemTimeConverterToSeconds);
	InitailVelStar1.z /= (SystemLengthConverterToKilometers/SystemTimeConverterToSeconds);

	InitailVelStar2.x /= (SystemLengthConverterToKilometers/SystemTimeConverterToSeconds);
	InitailVelStar2.y /= (SystemLengthConverterToKilometers/SystemTimeConverterToSeconds);
	InitailVelStar2.z /= (SystemLengthConverterToKilometers/SystemTimeConverterToSeconds);

	//Taking the run times into our units
	BranchRunTime *= (60.0*60.0*24.0)/SystemTimeConverterToSeconds;
	GrowthStartTimeStar1 *= (60.0*60.0*24.0)/SystemTimeConverterToSeconds;
	GrowthStopTimeStar1 *= (60.0*60.0*24.0)/SystemTimeConverterToSeconds;
	GrowthStartTimeStar2 *= (60.0*60.0*24.0)/SystemTimeConverterToSeconds;
	GrowthStopTimeStar2 *= (60.0*60.0*24.0)/SystemTimeConverterToSeconds;
	
	if(BranchRunTime < GrowthStopTimeStar1)
	{
		printf("\nTSU Error: BranchRunTime is less than GrowthStopTimeStar1.\n");
		exit(0);
	}
	
	if(BranchRunTime < GrowthStopTimeStar2)
	{
		printf("\nTSU Error: BranchRunTime is less than GrowthStopTimeStar2.\n");
		exit(0);
	}
	
	//Recording info into the BranchRunParameters file
	fprintf(BranchRunParameters, "\n RecordRate = %d", RecordRate);
	fprintf(BranchRunParameters, "\n DrawRate = %d", DrawRate);
	fclose(BranchRunParameters);
}

void readInTheInitialsStars()
{  
	//chdir("../");
	//system("ls";
	FILE *startFile = fopen("../FilesFromBuild/StartPosVelForce","rb");
	if(startFile == NULL)
	{
		printf("\n\n The StartPosVelForce file does not exist\n\n");
		exit(0);
	}
	fread(&StartTime, sizeof(float), 1, startFile);
	fread(PosCPU, sizeof(float4), NumberElements, startFile);
	fread(VelCPU, sizeof(float4), NumberElements, startFile);
	fread(ForceCPU, sizeof(float4), NumberElements, startFile);
	fclose(startFile);
}

void setInitialConditions()
{
	for(int i = 0; i < NumberElementsStar1; i++)	
	{
		PosCPU[i].x += InitailPosStar1.x;
		PosCPU[i].y += InitailPosStar1.y;
		PosCPU[i].z += InitailPosStar1.z;
		
		VelCPU[i].x += InitailVelStar1.x;
		VelCPU[i].y += InitailVelStar1.y;
		VelCPU[i].z += InitailVelStar1.z;
	}
	
	for(int i = NumberElementsStar1; i < NumberElements; i++)	
	{
		PosCPU[i].x += InitailPosStar2.x;
		PosCPU[i].y += InitailPosStar2.y;
		PosCPU[i].z += InitailPosStar2.z;
		
		VelCPU[i].x += InitailVelStar2.x;
		VelCPU[i].y += InitailVelStar2.y;
		VelCPU[i].z += InitailVelStar2.z;
	}
	
	CenterOfView = getCenterOfMass();
}

__global__ void getForces(float4 *pos, float4 *vel, float4 *force, int numberElementsStar1, int numberOfElements, float pressureIncrease1, float pressureIncrease2, float corePushBackReduction, float plasmaPushBackReduction, int gPUNumber, int gPUsUsed)
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
	if(0 < id && id < numberElementsStar1)
	{
		vel[id].w += pressureIncrease1;
	}
	else if(numberElementsStar1 < id)
	{
		vel[id].w += pressureIncrease2;
	}
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
	int   tDraw = 0;
	int   tRecord = 0;
	float pressureIncrease1, pressureIncrease2;
	int offSet = NumberElements/gPUsUsed;
	
	pressureIncrease1 = 1.0f;
	pressureIncrease2 = 1.0f;
	
	while(time < runTime)
	{	
		//Getting forces
		for(int i = 0; i < gPUsUsed; i++)
		{
			cudaSetDevice(i);
			errorCheck("cudaSetDevice");
			getForces<<<GridConfig, BlockConfig>>>(PosGPU[i], VelGPU[i], ForceGPU[i], NumberElementsStar1, NumberElements, pressureIncrease1, pressureIncrease2, CorePushBackReduction, PlasmaPushBackReduction, i, gPUsUsed);
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
					//printf(" pos i = %d  j = %d\n", i, j);
					//cudaMemcpyAsync(PosGPU[j] + (i*offSet)*sizeof(float4), PosGPU[i] + (i*offSet)*sizeof(float4), (NumberElements/gPUsUsed)*sizeof(float4), cudaMemcpyDeviceToDevice);
					cudaMemcpyAsync(&PosGPU[j][i*offSet], &PosGPU[i][i*offSet], (NumberElements/gPUsUsed)*sizeof(float4), cudaMemcpyDeviceToDevice);
					errorCheck("cudaMemcpy Pos A");
					
					//printf(" vel i = %d  j = %d\n", i, j);
					//cudaMemcpyAsync(VelGPU[j] + (i*offSet)*sizeof(float4), VelGPU[i] + (i*offSet)*sizeof(float4), (NumberElements/gPUsUsed)*sizeof(float4), cudaMemcpyDeviceToDevice);
					cudaMemcpyAsync(&VelGPU[j][i*offSet], &VelGPU[i][i*offSet], (NumberElements/gPUsUsed)*sizeof(float4), cudaMemcpyDeviceToDevice);
					errorCheck("cudaMemcpy Vel");
				}
			}
		}
		cudaDeviceSynchronize();
		errorCheck("cudaDeviceSynchronize");
	
		//Increasing the plasma elements push back. I had to start a dt forward so I could get the blocks to sync.
		if((GrowthStartTimeStar1 - dt) < time && time < (GrowthStopTimeStar1 - dt)) 
		{
			pressureIncrease1 = PercentForceIncreaseStar1;
		}
		else
		{
			pressureIncrease1 = 0.0f;
		}
		
		if((GrowthStartTimeStar2 - dt) < time && time < (GrowthStopTimeStar2 - dt)) 
		{
			pressureIncrease2 = PercentForceIncreaseStar2;
		}
		else
		{
			pressureIncrease2 = 0.0f;
		}
		
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
		
		tDraw++;
		tRecord++;
		time += dt;
	}
	return(time - dt);
}

void control()
{	
	struct sigaction sa;
	float time = StartTime;
	int gPUsUsed;
	clock_t startTimer, endTimer;
	
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

	// Creating branch folder and copying in all the files that contributed to making the branch run.
	printf("\n Creating and loading folder for the branch run.\n");
	createAndLoadFolderForNewBranchRun();
	
	// Reading in the build parameters.
	printf("\n Reading and setting the run parameters.\n");
	readAndSetRunParameters();
	
	// Reading in the branch parameters.
	printf("\n Reading and setting the branch parameters.\n");
	readAndSetBranchParameters();
	
	// Allocating memory for CPU and GPU.
	printf("\n Allocating memory on the GPU and CPU and opening positions and velocities file.\n");
	allocateCPUMemory();
	
	// Reading in the raw stars generated by the build program.
	printf("\n Reading in the stars that were generated in the build program.\n");
	readInTheInitialsStars();
	
	// Setting initial conditions.
	printf("\n Setting initial conditions for the branch run.\n");
	setInitialConditions();
	
	// Draw the intial configuration.
	printf("\n Drawing initial picture.\n");
	drawPicture();
	
	// Seting up the GPUs.
	printf("\n Setting up the GPU.\n");
	gPUsUsed = deviceSetup();
	
	// Running the simulation.
	printf("\n Running the simulation.\n");
	copyStarsUpToGPU(gPUsUsed);
	time = starNbody(time, BranchRunTime, DT, gPUsUsed);
	
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
