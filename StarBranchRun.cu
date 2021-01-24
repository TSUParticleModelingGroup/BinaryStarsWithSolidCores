/*
nvcc StarBranchRun.cu -o StarBranchRun.exe -lglut -lGL -lGLU -lm
nvcc StarBranchRun.cu -o StarBranchRun.exe -lglut -lGL -lGLU -lm --use_fast_math
*/

#include "binaryStarIncludes.h"
#include "binaryStarBranchDefines.h"
#include "binaryStarElementElementForces.h"
		
//Globals to hold the time, positions, velocities, and forces on both the CPU to be read in from the start file.
float StartTime;
float4 *PosCPU, *VelCPU, *ForceCPU;

//Globals to hold positions, velocities, and forces on both the GPU
float4 *PosGPU, *VelGPU, *ForceGPU;

//Globals to setup the kernals
dim3 BlockConfig, GridConfig;

//Root folder to containing the stars to work from.
char RootFolderName[256] = "";

//Globals to be readin from the RunParameters file
double SystemLengthConverterToKilometers;
double SystemMassConverterToKilograms;
double SystemTimeConverterToSeconds;
int NumberElementsStar1;
int NumberElementsStar2;
float CorePushBackReduction, PlasmaPushBackReduction;

//Global to be built from run parameters
int NumberElements;

//Globals read in from the BranchSetup file.
float BranchRunTime;
float GrowthStartTimeStar1, GrowthStopTimeStar1, PercentForceIncreaseStar1;
float GrowthStartTimeStar2, GrowthStopTimeStar2, PercentForceIncreaseStar2;
float4 InitailPosStar1, InitailVelStar1;
float4 InitailPosStar2, InitailVelStar2;
int RecordRate;
int DrawRate;

//Global that will reset your center of view.
float4 CenterOfView;

//File to hold the position and velocity outputs to make videos and analysis of the run.
FILE *PosAndVelFile;

//FIle to hold the ending time Pos vel and forces to continue the run.
FILE *FinalPosVelForceFile;

//FIle to hold the branch run parameters.
FILE *BranchRunParameters;


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
	mkdir(branchFolderName , S_IRWXU|S_IRWXG|S_IRWXO);
	
	//Copying files into the branch folder
	FILE *fileIn;
	FILE *fileOut;
	long sizeOfFile;
  	char *buffer;
	
	//Copying files from the main folder into the branch folder
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
	
	fileIn = fopen("../binaryStarBranchDefines.h", "rb");
	if(fileIn == NULL)
	{
		printf("\n\n The binaryStarBranchDefines.h file does not exist\n\n");
		exit(0);
	}
	fseek (fileIn , 0 , SEEK_END);
  	sizeOfFile = ftell(fileIn);
  	rewind (fileIn);
  	buffer = (char*)malloc(sizeof(char)*sizeOfFile);
  	fread (buffer, 1, sizeOfFile, fileIn);
	fileOut = fopen("binaryStarBranchDefines.h", "wb");
	fwrite (buffer, 1, sizeOfFile, fileOut);
	fclose(fileIn);
	fclose(fileOut);
	
	fileIn = fopen("../binaryStarIncludes.h", "rb");
	if(fileIn == NULL)
	{
		printf("\n\n The binaryStarIncludes.h file does not exist\n\n");
		exit(0);
	}
	fseek (fileIn , 0 , SEEK_END);
  	sizeOfFile = ftell(fileIn);
  	rewind (fileIn);
  	buffer = (char*)malloc(sizeof(char)*sizeOfFile);
  	fread (buffer, 1, sizeOfFile, fileIn);	
	fileOut = fopen("binaryStarIncludes.h", "wb");
	fwrite (buffer, 1, sizeOfFile, fileOut);
	fclose(fileIn);
	fclose(fileOut);
		
	fileIn = fopen("../binaryStarBuildDefines.h", "rb");
	if(fileIn == NULL)
	{
		printf("\n\n The binaryStarBuildDefines.h file does not exist\n\n");
		exit(0);
	}
	fseek (fileIn , 0 , SEEK_END);
  	sizeOfFile = ftell(fileIn);
  	rewind (fileIn);
  	buffer = (char*)malloc(sizeof(char)*sizeOfFile);
  	fread (buffer, 1, sizeOfFile, fileIn);
	fileOut = fopen("binaryStarBuildDefines.h", "wb");
	fwrite (buffer, 1, sizeOfFile, fileOut);
	fclose(fileIn);
	fclose(fileOut);
	
	fileIn = fopen("../binaryStarElementElementForces.h", "rb");
	if(fileIn == NULL)
	{
		printf("\n\n The binaryStarElementElementForces.h file does not exist\n\n");
		exit(0);
	}
	fseek (fileIn , 0 , SEEK_END);
  	sizeOfFile = ftell(fileIn);
  	rewind (fileIn);
  	buffer = (char*)malloc(sizeof(char)*sizeOfFile);
  	fread (buffer, 1, sizeOfFile, fileIn);
	fileOut = fopen("binaryStarElementElementForces.h", "wb");
	fwrite (buffer, 1, sizeOfFile, fileOut);
	fclose(fileIn);
	fclose(fileOut);
			
	fileIn = fopen("../StarBranchRun.cu", "rb");
	if(fileIn == NULL)
	{
		printf("\n\n The StarBranchRun.cu file does not exist\n\n");
		exit(0);
	}
	fseek (fileIn , 0 , SEEK_END);
  	sizeOfFile = ftell(fileIn);
  	rewind (fileIn);
  	buffer = (char*)malloc(sizeof(char)*sizeOfFile);
  	fread (buffer, 1, sizeOfFile, fileIn);
	fileOut = fopen("StarBranchRun.cu", "wb");
	fwrite (buffer, 1, sizeOfFile, fileOut);
	fclose(fileIn);
	fclose(fileOut);
	
	chdir("../");  //Back in main

	//Copying files from the root star folder into the branch folder
	chdir(RootFolderName);	
	fileIn = fopen("BuildSetup", "rb");
	if(fileIn == NULL)
	{
		printf("\n\n The BuildSetup file does not exist\n\n");
		exit(0);
	}
	fseek (fileIn , 0 , SEEK_END);
  	sizeOfFile = ftell(fileIn);
  	rewind (fileIn);
  	buffer = (char*)malloc(sizeof(char)*sizeOfFile);
  	fread (buffer, 1, sizeOfFile, fileIn);
  	chdir("../");	
	chdir(branchFolderName);
	fileOut = fopen("BuildSetup", "wb");
	fwrite (buffer, 1, sizeOfFile, fileOut);
	fclose(fileIn);
	fclose(fileOut);
	chdir("../");  //Back in main
	
	chdir(RootFolderName);	
	fileIn = fopen("RunParameters", "rb");
	if(fileIn == NULL)
	{
		printf("\n\n The RunParameters file does not exist\n\n");
		exit(0);
	}
	fseek (fileIn , 0 , SEEK_END);
  	sizeOfFile = ftell(fileIn);
  	rewind (fileIn);
  	buffer = (char*)malloc(sizeof(char)*sizeOfFile);
  	fread (buffer, 1, sizeOfFile, fileIn);
  	chdir("../");	
	chdir(branchFolderName);
	fileOut = fopen("RunParameters", "wb");
	fwrite (buffer, 1, sizeOfFile, fileOut);
	fclose(fileIn);
	fclose(fileOut);
	chdir("../");	//Back in main
	
	chdir(RootFolderName);	
	fileIn = fopen("StarBuilder.cu", "rb");
	if(fileIn == NULL)
	{
		printf("\n\n The StarBuilder.cu file does not exist\n\n");
		exit(0);
	}
	fseek (fileIn , 0 , SEEK_END);
  	sizeOfFile = ftell(fileIn);
  	rewind (fileIn);
  	buffer = (char*)malloc(sizeof(char)*sizeOfFile);
  	fread (buffer, 1, sizeOfFile, fileIn);
  	chdir("../");	
	chdir(branchFolderName);
	fileOut = fopen("StarBuilder.cu", "wb");
	fwrite (buffer, 1, sizeOfFile, fileOut);
	fclose(fileIn);
	fclose(fileOut);
	chdir("../");	//Back in main
	
	chdir(RootFolderName);	
	fileIn = fopen("StarBuildStats", "rb");
	if(fileIn == NULL)
	{
		printf("\n\n The StarBuildStats file does not exist\n\n");
		exit(0);
	}
	fseek (fileIn , 0 , SEEK_END);
  	sizeOfFile = ftell(fileIn);
  	rewind (fileIn);
  	buffer = (char*)malloc(sizeof(char)*sizeOfFile);
  	fread (buffer, 1, sizeOfFile, fileIn);
  	chdir("../");	
	chdir(branchFolderName);
	fileOut = fopen("StarBuildStats", "wb");
	fwrite (buffer, 1, sizeOfFile, fileOut);
	fclose(fileIn);
	fclose(fileOut);
	chdir("../");   //Back in main
	
	chdir(RootFolderName);		
	fileIn = fopen("StartPosVelForce", "rb");
	if(fileIn == NULL)
	{
		printf("\n\n The StartPosVelForce file does not exist\n\n");
		exit(0);
	}
	fseek (fileIn , 0 , SEEK_END);
  	sizeOfFile = ftell(fileIn);
  	rewind (fileIn);
  	buffer = (char*)malloc(sizeof(char)*sizeOfFile);
  	fread (buffer, 1, sizeOfFile, fileIn);
	chdir("../");	
	chdir(branchFolderName);
	fileOut = fopen("StartPosVelForce", "wb");
	fwrite (buffer, 1, sizeOfFile, fileOut);
	fclose(fileIn);
	fclose(fileOut);
	
	//Creating the positions and velosity file to dump to stuff to make movies out of.
	PosAndVelFile = fopen("PosAndVel", "wb");
	
	//Creating file to hold the ending time Pos vel and forces to continue the run.
	FinalPosVelForceFile = fopen("FinalPosVelForce", "wb");
	
	//Creating the BranchRunParameter file.
	BranchRunParameters = fopen("BranchRunParameters", "wb");

	chdir("../");   //Back in main
	
	free (buffer);
}

void readAndSetRunParameters()
{
	ifstream data;
	string name;
	
	chdir(RootFolderName);   
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
	chdir("../");
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
	
	//Recording info into the BranchRunParameters file
	fprintf(BranchRunParameters, "\n RecordRate = %d", RecordRate);
	fprintf(BranchRunParameters, "\n DrawRate = %d", DrawRate);
	fclose(BranchRunParameters);
}

void errorCheck(const char *message)
{
  cudaError_t  error;
  error = cudaGetLastError();

  if(error != cudaSuccess)
  {
    printf("\n CUDA ERROR: %s = %s\n", message, cudaGetErrorString(error));
    exit(0);
  }
}

void allocateMemory()
{
	PosCPU   = (float4*)malloc(NumberElements*sizeof(float4));
	VelCPU   = (float4*)malloc(NumberElements*sizeof(float4));
	ForceCPU = (float4*)malloc(NumberElements*sizeof(float4));
	
	cudaMalloc((void**)&PosGPU, NumberElements *sizeof(float4));
	errorCheck("cudaMalloc Pos");
	cudaMalloc((void**)&VelGPU, NumberElements *sizeof(float4));
	errorCheck("cudaMalloc Vel");
	cudaMalloc((void**)&ForceGPU, NumberElements *sizeof(float4));
	errorCheck("cudaMalloc Force");
}

void cleanUp()
{
	free(PosCPU);
	free(VelCPU);
	free(ForceCPU);
	
	cudaFree(PosGPU);
	cudaFree(VelGPU);
	cudaFree(ForceGPU);
	
	fclose(PosAndVelFile);
}

void readInTheInitialsStars()
{
	chdir(RootFolderName);   
	FILE *startFile = fopen("StartPosVelForce","rb");
	if(startFile == NULL)
	{
		printf("\n\n The StartPosVelForce file does not exist\n\n");
		exit(0);
	}
	chdir("../");
	fread(&StartTime, sizeof(float), 1, startFile);
	fread(PosCPU, sizeof(float4), NumberElements, startFile);
	fread(VelCPU, sizeof(float4), NumberElements, startFile);
	fread(ForceCPU, sizeof(float4), NumberElements, startFile);
	fclose(startFile);
}

float4 getCenterOfMass()
{
	double totalMass,cmx,cmy,cmz;
	float4 centerOfMass;
	
	cmx = 0.0;
	cmy = 0.0;
	cmz = 0.0;
	totalMass = 0.0;
	
	// This is asuming the mass of each element is 1.
	for(int i = 0; i < NumberElements; i++)
	{
    		cmx += PosCPU[i].x*PosCPU[i].w;
		cmy += PosCPU[i].y*PosCPU[i].w;
		cmz += PosCPU[i].z*PosCPU[i].w;
		totalMass += PosCPU[i].w;
	}
	
	centerOfMass.x = cmx/totalMass;
	centerOfMass.y = cmy/totalMass;
	centerOfMass.z = cmz/totalMass;
	centerOfMass.w = 0.0;
	
	return(centerOfMass);
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
	/*
	for(int i = 0; i < NumberElements; i++)
	{
		printf("\n x p = %f  v = %f  f = %f  i = %d", PosCPU[i].x, VelCPU[i].x, ForceCPU[i].x,i);
		printf("\n y p = %f  v = %f  f = %f  i = %d", PosCPU[i].y, VelCPU[i].y, ForceCPU[i].y,i);
		printf("\n z p = %f  v = %f  f = %f  i = %d", PosCPU[i].z, VelCPU[i].z, ForceCPU[i].z,i);
		printf("\n w p = %f  v = %f  f = %f  i = %d\n", PosCPU[i].w, VelCPU[i].w, ForceCPU[i].w,i);
	}
	*/
}

void deviceSetup()
{
	if(NumberElements%BLOCKSIZE != 0)
	{
		printf("\nTSU Error: Number of Particles is not a multiple of the block size \n\n");
		exit(0);
	}
	
	BlockConfig.x = BLOCKSIZE;
	BlockConfig.y = 1;
	BlockConfig.z = 1;
	
	GridConfig.x = (NumberElements-1)/BlockConfig.x + 1;
	GridConfig.y = 1;
	GridConfig.z = 1;
}

__global__ void getForces(float4 *pos, float4 *vel, float4 *force, int numberElementsStar1, int numberOfElements, float growthMultiplier1, float growthMultiplier2, float corePushBackReduction, float plasmaPushBackReduction)
{
	int id, ids, i, j;
	float4 posMe, velMe, forceMe;
	float4 partialForce;
	double forceSumX, forceSumY, forceSumZ;
	
	__shared__ float4 shPos[BLOCKSIZE];
	__shared__ float4 shVel[BLOCKSIZE];
	__shared__ float4 shForce[BLOCKSIZE];

	id = threadIdx.x + blockDim.x*blockIdx.x;
	if(numberOfElements <= id)
	{
		printf("\n TSU error: id out of bounds in getForcesSeperate. \n");
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
	
	for(j = 0; j < gridDim.x; j++)
	{
		shPos[threadIdx.x] = pos[threadIdx.x + blockDim.x*j];
		shVel[threadIdx.x] = vel[threadIdx.x + blockDim.x*j];
		shForce[threadIdx.x] = force[threadIdx.x + blockDim.x*j];
		__syncthreads();
	   
		#pragma unroll 32
		for(i = 0; i < blockDim.x; i++)	
		{
			ids = i + blockDim.x*j;
			if(id != ids)
			{
				if(id == 0 && ids == numberElementsStar1)
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
	
	force[id].x = (float)forceSumX;
	force[id].y = (float)forceSumY;
	force[id].z = (float)forceSumZ;
	if(0 < id && id < numberElementsStar1)
	{
		vel[id].w *= growthMultiplier1;
	}
	else if(numberElementsStar1 < id)
	{
		vel[id].w *= growthMultiplier2;
	}
}

__global__ void moveBodies(float4 *pos, float4 *vel, float4 *force, float dt)
{  
    	int id = threadIdx.x + blockDim.x*blockIdx.x;

	vel[id].x += (force[id].x/pos[id].w)*dt;
	vel[id].y += (force[id].y/pos[id].w)*dt;
	vel[id].z += (force[id].z/pos[id].w)*dt;

	pos[id].x += vel[id].x*dt;
	pos[id].y += vel[id].y*dt;
	pos[id].z += vel[id].z*dt;
}

void drawPicture()
{	
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	//Drawing the cores spheres
	glPushMatrix();
		glTranslatef(PosCPU[0].x - CenterOfView.x, PosCPU[0].y - CenterOfView.y, PosCPU[0].z - CenterOfView.z);
		glColor3d(1.0,0.0,0.0);
		glutSolidSphere(ForceCPU[0].w*0.2,20,20);  // force.w holds the diameter of an element
	glPopMatrix();
	
	glPushMatrix();
		glTranslatef(PosCPU[NumberElementsStar1].x - CenterOfView.x, PosCPU[NumberElementsStar1].y - CenterOfView.y, PosCPU[NumberElementsStar1].z - - CenterOfView.z);
		glColor3d(0.0,0.0,1.0);
		glutSolidSphere(ForceCPU[NumberElementsStar1].w*0.2,20,20);
	glPopMatrix();
	
	glPointSize(5.0);
	glBegin(GL_POINTS);
		//Drawing all the elements as points
		glColor3d(1.0,0.6,0.0);
 		for(int i = 0; i < NumberElementsStar1; i++)
		{
			glVertex3f(PosCPU[i].x - CenterOfView.x, PosCPU[i].y - CenterOfView.y, PosCPU[i].z - CenterOfView.z);
		}
		glColor3d(1.0,1.0,0.0);
		for(int i = NumberElementsStar1; i < NumberElements; i++)
		{
			glVertex3f(PosCPU[i].x - CenterOfView.x, PosCPU[i].y - CenterOfView.y, PosCPU[i].z - CenterOfView.z);
		}
		//Putting a colored point on the cores so you can track them
		glColor3d(1.0,0.0,0.0);
		glVertex3f(PosCPU[0].x - CenterOfView.x, PosCPU[0].y - CenterOfView.y, PosCPU[0].z - CenterOfView.z);
		glColor3d(0.0,0.0,1.0);
		glVertex3f(PosCPU[NumberElementsStar1].x - CenterOfView.x, PosCPU[NumberElementsStar1].y - CenterOfView.y, PosCPU[NumberElementsStar1].z - CenterOfView.z);
	glEnd();
	
	glutSwapBuffers();
}

void recordPosAndVel(float time)
{
	fwrite(&time, sizeof(float), 1, PosAndVelFile);
	fwrite(PosCPU, sizeof(float4), NumberElements, PosAndVelFile);
	fwrite(VelCPU, sizeof(float4), NumberElements, PosAndVelFile);
}

void recordFinalPosVelForceStars(float time)
{	
	fwrite(&time, sizeof(float), 1, FinalPosVelForceFile);
	fwrite(PosCPU, sizeof(float4), NumberElements, FinalPosVelForceFile);
	fwrite(VelCPU, sizeof(float4), NumberElements, FinalPosVelForceFile);
	fwrite(ForceCPU, sizeof(float4), NumberElements, FinalPosVelForceFile);
	
	fclose(FinalPosVelForceFile);
}

float starNbody(float time, float runTime, float dt)
{ 
	int   tDraw = 0;
	int   tRecord = 0;
	float growthMultiplier1, growthMultiplier2;
	
	growthMultiplier1 = 1.0f;
	growthMultiplier2 = 1.0f;
	
	while(time < runTime)
	{	
		getForces<<<GridConfig, BlockConfig>>>(PosGPU, VelGPU, ForceGPU, NumberElementsStar1, NumberElements, growthMultiplier1, growthMultiplier2, CorePushBackReduction, PlasmaPushBackReduction);
		errorCheck("getForcesSeperate");
		
		moveBodies<<<GridConfig, BlockConfig>>>(PosGPU, VelGPU, ForceGPU, dt);
		errorCheck("moveBodiesDamped");
	
		//Increasing the plasma elements push back. I had to start a dt forward so I could get the blocks to sync.
		if((GrowthStartTimeStar1 - dt) < time && time < (GrowthStopTimeStar1 - dt)) 
		{
			growthMultiplier1 += PercentForceIncreaseStar1;
		}
		else
		{
			growthMultiplier1 = 1.0f;
		}
		if((GrowthStartTimeStar2 - dt) < time && time < (GrowthStopTimeStar2 - dt)) 
		{
			growthMultiplier2 += PercentForceIncreaseStar2;
		}
		else
		{
			growthMultiplier2 = 1.0f;
		}
		
		if(tDraw == DrawRate) 
		{
			cudaMemcpy(PosCPU, PosGPU, (NumberElements)*sizeof(float4), cudaMemcpyDeviceToHost);
			errorCheck("cudaMemcpy Pos draw");
			printf("\n Time in days = %f", time*SystemTimeConverterToSeconds/(60.0*60.0*24.0)); 
			drawPicture();
			tDraw = 0;
		}
		if(tRecord == RecordRate) 
		{
			cudaMemcpy(PosCPU, PosGPU, (NumberElements)*sizeof(float4), cudaMemcpyDeviceToHost);
			errorCheck("cudaMemcpy Pos draw");
			cudaMemcpy(VelCPU, VelGPU, (NumberElements)*sizeof(float4), cudaMemcpyDeviceToHost);
			errorCheck("cudaMemcpy Vel draw");
			recordPosAndVel(time);
			tRecord = 0;
		}
		
		tDraw++;
		tRecord++;
		time += dt;
	}
	return(time - dt);
}

void copyStarsUpToGPU()
{
	cudaMemcpy(PosGPU, PosCPU, NumberElements*sizeof(float4), cudaMemcpyHostToDevice);
	errorCheck("cudaMemcpy Pos up");
	cudaMemcpy(VelGPU, VelCPU, NumberElements*sizeof(float4), cudaMemcpyHostToDevice);
	errorCheck("cudaMemcpy Vel up");
	cudaMemcpy(ForceGPU, ForceCPU, NumberElements*sizeof(float4), cudaMemcpyHostToDevice);
	errorCheck("cudaMemcpy Vel up");
}

void copyStarsDownFromGPU()
{
	cudaMemcpy( PosCPU, PosGPU, NumberElements*sizeof(float4), cudaMemcpyDeviceToHost );
	errorCheck("cudaMemcpy Pos");
	cudaMemcpy( VelCPU, VelGPU, NumberElements*sizeof(float4), cudaMemcpyDeviceToHost );
	errorCheck("cudaMemcpy Vel");
	cudaMemcpy( ForceCPU, ForceGPU, NumberElements*sizeof(float4), cudaMemcpyDeviceToHost );
	errorCheck("cudaMemcpy Vel");
}

static void signalHandler(int signum)
{
	int command;
	
	//exit(0);
    
	cout << "\n\n******************************************************" << endl;
	cout << "Enter:666 to kill the run." << endl;
	cout << "Enter:1 to change the draw rate." << endl;
	cout << "Enter:2 reset view to current center on mass." << endl;
	cout << "Enter:3 reset view to core 1." << endl;
	cout << "Enter:4 reset view to core 2." << endl;
	cout << "Enter:5 to continue the run." << endl;
	
	cout << "******************************************************\n\nCommand: ";
    
	cin >> command;
    
	if(command == 666)
	{
		cout << "\n\n******************************************************" << endl;
		cout << "Are you sure you want to terminate the run?" << endl;
		cout << "Enter:666 again if you are sure. Enter anything else to continue the run." << endl;
		cout << "******************************************************\n\nCommand: ";
		cin >> command;
		
		if(command == 666)
		{
			cleanUp();
			exit(0);
		}
	}
	else if(command == 1)
	{
		cout << "\nEnter the desired draw rate: ";
		cin >> DrawRate;
		cout << "\nDrawRate: " << DrawRate << endl;
	}
	else if(command == 2)
	{
		copyStarsDownFromGPU();
		CenterOfView = getCenterOfMass();
		cout << "\nReset view to current center of mass." << endl;
	}
	else if(command == 3)
	{
		CenterOfView = PosCPU[0];
		copyStarsDownFromGPU();
		cout << "\nReset view to current center of mass." << endl;
	}
	else if(command == 4)
	{
		CenterOfView = PosCPU[NumberElements];
		copyStarsDownFromGPU();
		cout << "\nReset view to current center of mass." << endl;
	}
	else if (command == 5)
	{
		cout << "\nRun continued." << endl;
	}
	else
	{
		cout <<"\n\n Invalid Command\n" << endl;
	}
}

void control()
{	
	struct sigaction sa;
	float time = StartTime;
	
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
	allocateMemory();
	
	// Reading in the raw stars generated by the build program.
	printf("\n Reading in the stars that were generated in the build program.\n");
	readInTheInitialsStars();
	
	// Setting initial conditions.
	printf("\n Setting initial conditions for the branch run.\n");
	setInitialConditions();
	
	// Draw the intial configuration.
	printf("\n Drawing initial picture.\n");
	drawPicture();
	
	// Seting up the GPU.
	printf("\n Setting up the GPU.\n");
	deviceSetup();
	
	// Running the simulation.
	printf("\n Running the simulation.\n");
	copyStarsUpToGPU();
	time = starNbody(time, BranchRunTime, DT);
	
	// Saving the the runs final positions and velosities.	
	printf("\n Saving the the runs final positions and velosities.\n");
	copyStarsDownFromGPU();
	recordFinalPosVelForceStars(time);  
	
	// Saving any wanted stats about the run that you may want. I don't have anything to record as of yet.
	printf("\n Saving any wanted stats about the run that you may want.\n");
	//recordStarStats();	
	
	// Freeing memory. 	
	printf("\n Cleaning up the run.\n");
	cleanUp();

	printf("\n DONE \n");
	exit(0);
}

//Globals for viewing
//Viewing cropped pyrimid
double ViewBoxSize = 60.0;

GLdouble Left = -ViewBoxSize;
GLdouble Right = ViewBoxSize;
GLdouble Bottom = -ViewBoxSize;
GLdouble Top = ViewBoxSize;
GLdouble Front = ViewBoxSize;
GLdouble Back = -ViewBoxSize;

//Where your eye is located
GLdouble EyeX = 5.0;
GLdouble EyeY = 5.0;
GLdouble EyeZ = 5.0;

//Where you are looking
GLdouble CenterX = 0.0;
GLdouble CenterY = 0.0;
GLdouble CenterZ = 0.0;

//Up vector for viewing
GLdouble UpX = 0.0;
GLdouble UpY = 1.0;
GLdouble UpZ = 0.0;

void Display(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(Left, Right, Bottom, Top, Front, Back);
	glMatrixMode(GL_MODELVIEW);
	gluLookAt(EyeX, EyeY, EyeZ, CenterX, CenterY, CenterZ, UpX, UpY, UpZ);
}

void reshape(GLint w, GLint h) 
{
	glViewport(0, 0, w, h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(Left, Right, Bottom, Top, Front, Back);
	glMatrixMode(GL_MODELVIEW);
}

void init()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(Left, Right, Bottom, Top, Front, Back);
	glMatrixMode(GL_MODELVIEW);
	gluLookAt(EyeX, EyeY, EyeZ, CenterX, CenterY, CenterZ, UpX, UpY, UpZ);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

int main(int argc, char** argv)
{ 
	if( argc < 2)
	{
		printf("\n You need to intire a root folder to work from on the comand line\n");
		exit(0);
	}
	else
	{
		strcat(RootFolderName, argv[1]);
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






