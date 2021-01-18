/*
nvcc StarBranchRun.cu -o ContinueStarBranchRun -lglut -lGL -lGLU -lm
nvcc StarBranchRun.cu -o ContinueStarBranchRun -lglut -lGL -lGLU -lm --use_fast_math
*/


#include "binaryStar.h"

#define BLOCKSIZE 512

//Globals to hold positions, velocities, and forces on both the GPU and CPU
float4 *PlaceHolder; //needs to be hard defined for cuda
float4 *Pos, *Vel, *Force;
float4 *PosGPU, *VelGPU, *ForceGPU;

//Globals to setup the kernals
dim3 BlockConfig, GridConfig;

//Root folder to containing the stars to work with.
char RootFolderName[256] = "";

//Globals read in from the BiuldSetup file
double Pi;
double UniversalGravity;			//Universal gravitational constant in kilometersE3 kilogramsE-1 and secondsE-2 (??? source)
double MassOfSun;
double RadiusOfSun;
double FractionSunMassOfStar1;
double FractionSunMassOfStar2;
double DensityOfHydrogenGas;		//Density of hydrogen gas in kilograms meterE-3 (??? source)
float KH;							//Push back of hygrogen gas element
float KRH;							//Push back reduction of hygrogen gas element
int NumberOfElements;
float Dt;


//Globals to be set by the setRunParameters function
double SystemLengthConverterToKilometers;
double SystemMassConverterToKilograms;
double SystemTimeConverterToSeconds;
int NumberOfElementsInStar1;
int NumberOfElementsInStar2; 

//Globals read in from the BranchSetup file.
float BranchRunTime;
float GrowStartTime;
float GrowStopTime;
float DeltaForceIncrease;
int RecordRate;
int DrawRate;

//File to hold the position and velocity outputs to make videos and analysis of the run.
FILE *PosAndVelFile;

void readBuildParameters()
{
	ifstream data;
	string name;
	float temp;
	
	data.open("BuildSetup");
	
	if(data.is_open() == 1)
	{
		getline(data,name,'=');
		data >> Pi;
		getline(data,name,'=');
		data >> UniversalGravity;
		getline(data,name,'=');
		data >> RadiusOfSun;
		getline(data,name,'=');
		data >> MassOfSun;
		
		getline(data,name,'=');
		data >> FractionSunMassOfStar1;
		getline(data,name,'=');
		data >> FractionSunMassOfStar2;
		
		getline(data,name,'=');
		data >> DensityOfHydrogenGas;
		getline(data,name,'=');
		data >> KH;
		getline(data,name,'=');
		data >> KRH;
		
		getline(data,name,'=');
		data >> NumberOfElements;
		
		getline(data,name,'=');
		data >> temp;
		getline(data,name,'=');
		data >> temp;
		getline(data,name,'=');
		data >> temp;
		getline(data,name,'=');
		data >> temp;
		
		getline(data,name,'=');
		data >> temp;
		getline(data,name,'=');
		data >> temp;
		getline(data,name,'=');
		data >> temp;
		getline(data,name,'=');
		data >> temp;
		
		getline(data,name,'=');
		data >> temp;
		getline(data,name,'=');
		data >> temp;
		getline(data,name,'=');
		data >> temp;
		
		getline(data,name,'=');
		data >> temp;
		
		getline(data,name,'=');
		data >> Dt;
		
		getline(data,name,'=');
		data >> temp;
	}
	else
	{
		printf("\nTSU Error could not open run or root Setup file\n");
		exit(0);
	}
	data.close();
}

void generateRunParameters()
{
	double massOfStar1;
	double massOfStar2;
	double massOfHydrogenGasElement;
	double baseDiameterOfHydrogenGasElement;
	
	massOfStar1 = MassOfSun*FractionSunMassOfStar1;
	massOfStar2 = MassOfSun*FractionSunMassOfStar2;
	
	//The mass of an element is just the total mass divided by the number of elements used.
	massOfHydrogenGasElement = (massOfStar1 + massOfStar2)/((double)NumberOfElements);
	
	//We will use the mass of a hydrogen gas element as one unit of mass. 
	//The following constant will convert system masses up to kilograms by multipling 
	//or convert kilograms down to system units by dividing.
	SystemMassConverterToKilograms = massOfHydrogenGasElement;
	
	NumberOfElementsInStar1 = massOfStar1/massOfHydrogenGasElement;
	NumberOfElementsInStar2 = NumberOfElements - NumberOfElementsInStar1;
	
	baseDiameterOfHydrogenGasElement = pow((6.0*massOfHydrogenGasElement)/(Pi*DensityOfHydrogenGas), (1.0/3.0));
	
	//We will use the diameter of a hydrogen gas element as one unit of length. 
	//The following constant will convert system lengths up to kilometers by multipling 
	//or convert kilometers down to system units by dividing.
	SystemLengthConverterToKilometers = baseDiameterOfHydrogenGasElement;
	
	//We will use a time unit so that the universal gravitational constant will be 1. 
	//The following constant will convert system times up to seconds by multipling 
	//or convert seconds down to system units by dividing. Make sure UniversalGravity is fed into the program in kilograms kilometers and seconds!
	SystemTimeConverterToSeconds = sqrt(pow(SystemLengthConverterToKilometers,3)/(SystemMassConverterToKilograms*UniversalGravity));
	
	KH *= SystemTimeConverterToSeconds*SystemTimeConverterToSeconds*SystemLengthConverterToKilometers/SystemMassConverterToKilograms;
	printf("\n KH = %f\n", KH);
}

void readBranchParametersAndSetInitialConditions()
{
	ifstream data;
	string name;
	
	float3 initailPos1, initailPos2, initailVel1, initailVel2;
	
	data.open("BranchSetup");
	
	if(data.is_open() == 1)
	{
		getline(data,name,'=');
		data >> initailPos1.x;
		getline(data,name,'=');
		data >> initailPos1.y;
		getline(data,name,'=');
		data >> initailPos1.z;
		
		getline(data,name,'=');
		data >> initailPos2.x;
		getline(data,name,'=');
		data >> initailPos2.y;
		getline(data,name,'=');
		data >> initailPos2.z;
		
		getline(data,name,'=');
		data >> initailVel1.x;
		getline(data,name,'=');
		data >> initailVel1.y;
		getline(data,name,'=');
		data >> initailVel1.z;
		
		getline(data,name,'=');
		data >> initailVel2.x;
		getline(data,name,'=');
		data >> initailVel2.y;
		getline(data,name,'=');
		data >> initailVel2.z;
		
		getline(data,name,'=');
		data >> BranchRunTime;
		
		getline(data,name,'=');
		data >> GrowStartTime;
		
		getline(data,name,'=');
		data >> GrowStopTime;
		
		getline(data,name,'=');
		data >> DeltaForceIncrease;
		
		getline(data,name,'=');
		data >> RecordRate;
		
		getline(data,name,'=');
		data >> DrawRate;
		
		/*
		printf("\n BranchRunTime = %f\n", BranchRunTime);
		printf("\n GrowStartTime = %f\n", GrowStartTime);
		printf("\n GrowStopTime = %f\n", GrowStopTime);
		printf("\n DeltaForceIncrease = %f\n", DeltaForceIncrease);
		printf("\n RecordRate = %d\n", RecordRate);
		printf("\n drawRate = %d\n", DrawRate);
		*/
		
		// Taking the run times into our units,
		BranchRunTime *= (60.0*60.0*24.0)/SystemTimeConverterToSeconds;
		GrowStartTime *= (60.0*60.0*24.0)/SystemTimeConverterToSeconds;
		GrowStopTime *= (60.0*60.0*24.0)/SystemTimeConverterToSeconds;
		
		/*
		printf("\n BranchRunTime = %f\n", BranchRunTime);
		printf("\n GrowStartTime = %f\n", GrowStartTime);
		printf("\n GrowStopTime = %f\n", GrowStopTime);
		*/
		
		for(int i = 0; i < NumberOfElementsInStar1; i++)	
		{
			Pos[i].x += initailPos1.x/SystemLengthConverterToKilometers;
			Pos[i].y += initailPos1.y/SystemLengthConverterToKilometers;
			Pos[i].z += initailPos1.z/SystemLengthConverterToKilometers;
			
			Vel[i].x += initailVel1.x*SystemTimeConverterToSeconds/SystemLengthConverterToKilometers;
			Vel[i].y += initailVel1.y*SystemTimeConverterToSeconds/SystemLengthConverterToKilometers;
			Vel[i].z += initailVel1.z*SystemTimeConverterToSeconds/SystemLengthConverterToKilometers;
		}
		
		for(int i = NumberOfElementsInStar1; i < NumberOfElements; i++)	
		{
			Pos[i].x += initailPos2.x/SystemLengthConverterToKilometers;
			Pos[i].y += initailPos2.y/SystemLengthConverterToKilometers;
			Pos[i].z += initailPos2.z/SystemLengthConverterToKilometers;
			
			Vel[i].x += initailVel2.x*SystemTimeConverterToSeconds/SystemLengthConverterToKilometers;
			Vel[i].y += initailVel2.y*SystemTimeConverterToSeconds/SystemLengthConverterToKilometers;
			Vel[i].z += initailVel2.z*SystemTimeConverterToSeconds/SystemLengthConverterToKilometers;
		}
	}
	else
	{
		printf("\nTSU Error could not open run or root Setup file\n");
		exit(0);
	}
	data.close();
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
	PlaceHolder = (float4*)malloc(NumberOfElements*sizeof(float4));
	Pos = (float4*)malloc(NumberOfElements*sizeof(float4));
	Vel = (float4*)malloc(NumberOfElements*sizeof(float4));
	Force = (float4*)malloc(NumberOfElements*sizeof(float4));
	
	cudaMalloc((void**)&PosGPU, NumberOfElements *sizeof(float4));
	errorCheck("cudaMalloc Pos");
	cudaMalloc((void**)&VelGPU, NumberOfElements *sizeof(float4));
	errorCheck("cudaMalloc Vel");
	cudaMalloc((void**)&ForceGPU, NumberOfElements *sizeof(float4));
	errorCheck("cudaMalloc Force");
	
	PosAndVelFile = fopen("PosAndVel", "wb");
}

void cleanUp()
{
	free(PlaceHolder);
	free(Pos);
	free(Vel);
	free(Force);
	
	cudaFree(PosGPU);
	cudaFree(VelGPU);
	cudaFree(ForceGPU);
	
	fclose(PosAndVelFile);
}

void readInTheInitialsStars()
{
	FILE *temp = fopen("StartPosAndVel","rb");
	fread(Pos, sizeof(float4), NumberOfElements, temp);
	fread(Vel, sizeof(float4), NumberOfElements, temp);
	fclose(temp);
	
	printf("\n************************************************** The stars have been read in\n");
}

void deviceSetup()
{
	if(NumberOfElements%BLOCKSIZE != 0)
	{
		printf("\nTSU Error: Number of Particles is not a multiple of the block size \n\n");
		exit(0);
	}
	
	BlockConfig.x = BLOCKSIZE;
	BlockConfig.y = 1;
	BlockConfig.z = 1;
	
	GridConfig.x = (NumberOfElements-1)/BlockConfig.x + 1;
	GridConfig.y = 1;
	GridConfig.z = 1;
}

__device__ float4 calculateElementElementForce(float4 posMe, float4 posYou, float4 velMe, float4 velYou, float KH, float KRH, float forceGrowth, float pi)
{
	float4 dp, dv, force;
	float r, r2, r3, invr, inOut, force_mag;
	
	dp.x = posYou.x - posMe.x;
	dp.y = posYou.y - posMe.y;
	dp.z = posYou.z - posMe.z;
	r2 = dp.x*dp.x + dp.y*dp.y + dp.z*dp.z;
	r = sqrt(r2);
	r3 = r2*r;
	
	if(1.0f <= r)
	{
		invr = 1.0f/r3;
		
		force.x = (dp.x*invr);
		force.y = (dp.y*invr);
		force.z = (dp.z*invr);
	}
	else if(0.0f < r)
	{
		invr = 1.0f/r;
		
		dv.x = velYou.x - velMe.x;
		dv.y = velYou.y - velMe.y;
		dv.z = velYou.z - velMe.z;
		inOut = dp.x*dv.x + dp.y*dv.y + dp.z*dv.z;
		
		if(inOut <= 0) 	force_mag  = 1.0f - (forceGrowth+KH)*pi*(1.0f/6.0f + r3/3.0f - r2/2.0f);
		else 			force_mag  = 1.0f - KRH*(forceGrowth+KH)*pi*(1.0f/6.0f + r3/3.0f - r2/2.0f);
		
		force.x = (dp.x*invr)*force_mag;
		force.y = (dp.y*invr)*force_mag;
		force.z = (dp.z*invr)*force_mag;
	}
	else // Hopefully this line of code never gets reached.
	{
		dv.x = velYou.x - velMe.x;
		dv.y = velYou.y - velMe.y;
		dv.z = velYou.z - velMe.z;
		if(0.0f < (dv.x + dv.y + dv.z)) // Hopefully if it have some velocity it will drift past setting right on top of eachother.
		{
			force.x = 0.0f;
			force.y = 0.0f;
			force.z = 0.0f;
		}
		else // If they have no velocity we will have to kick them off of this position. This will be an unnatural force in the x direction
		{
			force.x = KH;
			force.y = 0.0f;
			force.z = 0.0f;
		}
	}
	return(force);
}


__global__ void getForce(float4 *pos, float4 *vel, float4 *force, float KH, float KRH, float forceGrowth, int NumberOfElementsInStar1, float pi)
{
	int id, ids, i, j;
	float4 posMe, velMe;
	float4 elementElementForce, forceSum;
	
	__shared__ float4 shPos[BLOCKSIZE];
	__shared__ float4 shVel[BLOCKSIZE];

	id = threadIdx.x + blockDim.x*blockIdx.x;
		
	forceSum.x = 0.0f;
	forceSum.y = 0.0f;
	forceSum.z = 0.0f;
		
	posMe.x = pos[id].x;
	posMe.y = pos[id].y;
	posMe.z = pos[id].z;
	
	velMe.x = vel[id].x;
	velMe.y = vel[id].y;
	velMe.z = vel[id].z;
	
	for(j = 0; j < gridDim.x; j++)
	{
		shPos[threadIdx.x] = pos[threadIdx.x + blockDim.x*j];
		shVel[threadIdx.x] = vel[threadIdx.x + blockDim.x*j];
		__syncthreads();
	   
		#pragma unroll 32
		for(i = 0; i < blockDim.x; i++)	
		{
			ids = i + blockDim.x*j;
			if(id != ids)
			{
				if(id < NumberOfElementsInStar1 && ids < NumberOfElementsInStar1)
				{
					elementElementForce = calculateElementElementForce(posMe, shPos[i], velMe, shVel[i], KH, KRH, forceGrowth, pi);
				}
				else
				{
					elementElementForce = calculateElementElementForce(posMe, shPos[i], velMe, shVel[i], KH, KRH, 0.0f, pi);
				}
				forceSum.x += elementElementForce.x;
				forceSum.y += elementElementForce.y;
				forceSum.z += elementElementForce.z;
			}
		}
		__syncthreads();
	}
	force[id].x = forceSum.x;
	force[id].y = forceSum.y;
	force[id].z = forceSum.z;
}

__global__ void moveBodies(float4 *pos, float4 *vel, float4 *force, float dt)
{
	int id;
	
    id = threadIdx.x + blockDim.x*blockIdx.x;

	// There is no need to divide by mass in the velocity lines because the mass of an element is 1.
	vel[id].x += (force[id].x)*dt;
	vel[id].y += (force[id].y)*dt;
	vel[id].z += (force[id].z)*dt;

	pos[id].x += vel[id].x*dt;
	pos[id].y += vel[id].y*dt;
	pos[id].z += vel[id].z*dt;
}

void drawPicture(float4 *pos, int NumberOfElementsInStar1, int NumberOfElementsInStar2)
{	
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	glBegin(GL_POINTS);
 		for(int i = 0; i < NumberOfElementsInStar1; i++)
		{
			glColor3d(1.0,1.0,0.0);
			glVertex3f(pos[i].x, pos[i].y, pos[i].z);
		}
		
		for(int i = NumberOfElementsInStar1; i < NumberOfElements; i++)
		{
			glColor3d(1.0,0.5,0.0);
			glVertex3f(pos[i].x, pos[i].y, pos[i].z);
		}
	glEnd();
	
	glutSwapBuffers();
}

void recordPosAndVel()
{
	fwrite(Pos, sizeof(float4), NumberOfElements, PosAndVelFile);
	fwrite(Vel, sizeof(float4), NumberOfElements, PosAndVelFile);
}

void recordFinalPosVelStars(float time)
{
	const char *folderName;
	stringstream streamTime;
	string stringTime;
	string stringFolderName;
	
	streamTime << time;
	stringTime = streamTime.str();
	stringFolderName = "PosAndVelAt" + stringTime + "Days";
	folderName = stringFolderName.c_str();
	
	FILE *finalPosAndVelFile;
	
	//finalPosAndVelFile = fopen("FinalPosAndVel", "wb");
	finalPosAndVelFile = fopen(folderName, "wb");
	cudaMemcpy( Pos, PosGPU, NumberOfElements *sizeof(float4), cudaMemcpyDeviceToHost );
	errorCheck("cudaMemcpy Pos");
	cudaMemcpy( Vel, VelGPU, NumberOfElements *sizeof(float4), cudaMemcpyDeviceToHost );
	errorCheck("cudaMemcpy Vel");
	
	fwrite(Pos, sizeof(float4), NumberOfElements, finalPosAndVelFile);
	fwrite(Vel, sizeof(float4), NumberOfElements, finalPosAndVelFile);
	
	fclose(finalPosAndVelFile);
}

void recordStarStats()
{
	FILE *starStatsFile;
	double massStar1, massStar2;
	double radiusStar1, radiusStar2;
	double densityStar1, densityStar2;
	double radius;
	
	cudaMemcpy( Pos, PosGPU, NumberOfElements *sizeof(float4), cudaMemcpyDeviceToHost );
	errorCheck("cudaMemcpy Pos");
	cudaMemcpy( Vel, VelGPU, NumberOfElements *sizeof(float4), cudaMemcpyDeviceToHost );
	errorCheck("cudaMemcpy Vel");
	
	massStar1 = NumberOfElementsInStar1*SystemMassConverterToKilograms;
	massStar2 = NumberOfElementsInStar2*SystemMassConverterToKilograms;
	
	radiusStar1 = 0.0;
	for(int i = 0; i < NumberOfElementsInStar1; i++)
	{
		radius = sqrt(Pos[i].x*Pos[i].x + Pos[i].y*Pos[i].y + Pos[i].z*Pos[i].z);
		if(radiusStar1 < radius) radiusStar1 = radius;
	}
	
	radiusStar2 = 0.0;
	for(int i = NumberOfElementsInStar1; i < NumberOfElements; i++)
	{
		radius = sqrt(Pos[i].x*Pos[i].x + Pos[i].y*Pos[i].y + Pos[i].z*Pos[i].z);
		if(radiusStar2 < radius) radiusStar2 = radius;
	}
	
	radiusStar1 *= SystemLengthConverterToKilometers;
	radiusStar2 *= SystemLengthConverterToKilometers;
	
	densityStar1 = massStar1/((4.0/3.0)*Pi*radiusStar1*radiusStar1*radiusStar1);
	densityStar2 = massStar2/((4.0/3.0)*Pi*radiusStar2*radiusStar2*radiusStar2);
	
	starStatsFile = fopen("StarStats", "wb");
		fprintf(starStatsFile, " The conversion parameters to take you to and from our units to kilograms, kilometers, seconds follow\n");
		fprintf(starStatsFile, " Mass in our units is the mass of an element. In other words the mass of an element is one.\n");
		fprintf(starStatsFile, " Length in our units is the starting diameter of an element. In other words the staring base diameter of an element is one.\n");
		fprintf(starStatsFile, " Time in our units is set so that the universal gravitational constant is 1.");
		fprintf(starStatsFile, "\n ");
		fprintf(starStatsFile, "\n Our length unit is this many kilometers: %e", SystemLengthConverterToKilometers);
		fprintf(starStatsFile, "\n Our mass unit is this many kilograms: %e", SystemMassConverterToKilograms);
		fprintf(starStatsFile, "\n Our time unit is this many seconds: %e or days %e", SystemTimeConverterToSeconds, SystemTimeConverterToSeconds/(60*60*24));
		fprintf(starStatsFile, "\n ");
		fprintf(starStatsFile, "\n Total number of elements in star1: %d", NumberOfElementsInStar1);
		fprintf(starStatsFile, "\n Total number of elements in star2: %d", NumberOfElementsInStar2);
		fprintf(starStatsFile, "\n ");
		fprintf(starStatsFile, "\n Mass of Star1 = %e kilograms", massStar1);
		fprintf(starStatsFile, "\n Radius of Star1 = %e kilometers", radiusStar1);
		fprintf(starStatsFile, "\n Density of star1 = %e kilograms/(cubic kilometer)", densityStar1);
		fprintf(starStatsFile, "\n ");
		fprintf(starStatsFile, "\n Mass of Star2 = %e kilograms", massStar2);
		fprintf(starStatsFile, "\n Radius of Star2 = %e kilometers", radiusStar2);
		fprintf(starStatsFile, "\n Density of star2 = %e kilograms/(cubic kilometer)", densityStar2);
	fclose(starStatsFile);
}

float nBody()
{ 
	float time;
	int   draw, record;
	float forceGrowth = 0.0;
	
	cudaMemcpy(PosGPU, Pos, (NumberOfElements)*sizeof(float4), cudaMemcpyHostToDevice);
	errorCheck("cudaMemcpy Pos up");
	cudaMemcpy(VelGPU, Vel, NumberOfElements*sizeof(float4), cudaMemcpyHostToDevice);
	errorCheck("cudaMemcpy Vel up");
	
	printf("\n************************************************** Simulation is on\n");
	time = 0.0;
	draw = 0;
	record = 0;
	while(time < BranchRunTime)
	{	
		getForce<<<GridConfig, BlockConfig>>>(PosGPU, VelGPU, ForceGPU, KH, KRH, forceGrowth, NumberOfElementsInStar1, (float)Pi);
		errorCheck("getForce");
		moveBodies<<<GridConfig, BlockConfig>>>(PosGPU, VelGPU, ForceGPU, Dt);
		errorCheck("moveBodiesDamped");
		
		if(GrowStartTime < time && time < GrowStopTime) 
		{
			forceGrowth += DeltaForceIncrease;
		}
		
		if(draw == DrawRate) 
		{
			cudaMemcpy(Pos, PosGPU, (NumberOfElements)*sizeof(float4), cudaMemcpyDeviceToHost);
			errorCheck("cudaMemcpy Pos draw");
	    	drawPicture(Pos, NumberOfElementsInStar1, NumberOfElementsInStar2);
	    	printf("\n Time in days = %f\n", time*SystemTimeConverterToSeconds/(60.0*60.0*24.0));
			draw = 0;
		}
		draw++;
		
		if(record == RecordRate) 
		{
			cudaMemcpy(Pos, PosGPU, (NumberOfElements)*sizeof(float4), cudaMemcpyDeviceToHost);
			errorCheck("cudaMemcpy Pos draw");
	    	recordPosAndVel();
			record = 0;
		}
		record++;
		
		time += Dt;
	}
	return(time -= Dt);
	//while(1);
}

static void signalHandler(int signum)
{
	int command;
    
	cout << "\n\n******************************************************" << endl;
	cout << "Enter:666 to kill the run." << endl;
	cout << "Enter:1 to change the draw rate." << endl;
	cout << "Enter:2 to continue the run." << endl;
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
	else if (command == 2)
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
	float time;
	
	// Handling input from the screen.
	sa.sa_handler = signalHandler;
	sigemptyset(&sa.sa_mask);
	sa.sa_flags = SA_RESTART; // Restart functions if interrupted by handler
	if (sigaction(SIGINT, &sa, NULL) == -1)
	{
		printf("\nTSU Error: sigaction error\n");
	}

	// Creating folder to hold the newly created stars and moving into that folder. It also makes a copy of the BiuldSetup file in this folder.
	printf("\n Creating folders for the branch run.\n");
	createFolderForNewBranchRun();
	
	// Reading in the build parameters to a file.
	printf("\n Reading in the build parameters.\n");
	readBuildParameters();
	
	// Creating and saving the run the run parameters to a file.
	printf("\n Creating and saving the run parameters file.\n");
	generateRunParameters();
	
	// Allocating memory for CPU and GPU.
	printf("\n Allocating memory on the GPU and CPU and opening positions and velocities file.\n");
	allocateMemory();
	
	// Reading in the raw stars from the build generated by the build program.
	printf("\n Reading in the stars that were generated in the build probram.\n");
	readInTheInitialsStars();
	
	// Reading in Branch parameters and seting inintial conditions.
	printf("\n Reading in Branch parameters and seting inintial conditions.\n");
	readBranchParametersAndSetInitialConditions();
	
	// Seting up the GPU.
	printf("\n Setting up the GPU.\n");
	deviceSetup();
	
	// Running the simulation.
	printf("\n Running the simulation.\n");
	time = nBody();
	
	// Saving the the runs final positions and velosities.	
	printf("\n Saving the the runs final positions and velosities.\n");
	time = time*SystemTimeConverterToSeconds/(60.0*60.0*24.0); // Converting to hours.
	recordFinalPosVelStars(time);  
	
	// Saving any wanted stats about the run that you may want.
	printf("\n Saving any wanted stats about the run that you may want.\n");
	recordStarStats();	
	
	// Freeing memory. 	
	printf("\n Cleaning up the run.\n");
	cleanUp();

	printf("\n DONE \n");
	exit(0);
}

//Globals for viewing
//Viewing cropped pyrimid
double ViewBoxSize = 10.0;

GLdouble Left = -ViewBoxSize;
GLdouble Right = ViewBoxSize;
GLdouble Bottom = -ViewBoxSize;
GLdouble Top = ViewBoxSize;
GLdouble Front = ViewBoxSize;
GLdouble Back = -ViewBoxSize;

//Where your eye is located
GLdouble EyeX = 0.0;
GLdouble EyeY = 1.0;
GLdouble EyeZ = 1.0;

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
	int xWindowSize = 1500;
	int yWindowSize = 1500; 
	
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






