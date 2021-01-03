/*
nvcc StarBuilder.cu -o StarBuilder -lglut -lGL -lGLU -lm -arch=sm_60
 nvcc StarBuilder.cu -o StarBuilder -lglut -lGL -lGLU -lm -arch=compute_60 -code=sm_60
nvcc StarBuilder.cu -o StarBuilder -lglut -lGL -lGLU -lm --use_fast_math
*/

#include "binaryStar.h"

#define BLOCKSIZE 512
//#define PI 3.141592654f

//Globals to hold positions, velocities, and forces on both the GPU and CPU
float4 *PlaceHolder; //needs to be hard defined for cuda
float4 *PosCPU, *VelCPU; // *ForceCPU;
float4 *PosGPU, *VelGPU, *ForceGPU;
float4 *PosCoreCPU, *VelCoreCPU, *ForceCoreCPU;
float4 *PosCoreGPU, *VelCoreGPU, *ForceCoreGPU;
//double *CoreForceSumCPU;
double *CoreForceSumGPU;

//Globals to setup the kernals
dim3 BlockConfig, GridConfig;

//Globals read in from the BiuldSetup file
double Pi;
double UniversalGravity;			//Universal gravitational constant in kilometersE3 kilogramsE-1 and secondsE-2 (??? source)
double MassOfSun, RadiusOfSun;
double MassOfStar1, MassOfCore1, RadiusOfCore1;
double MassOfStar2, MassOfCore2, RadiusOfCore2;
double RadiusTolerance, TargetRadiusStar1, TargetRadiusStar2;
double DensityOfHydrogenGas;		//Density of hydrogen gas in kilograms meterE-3 (??? source)
float KH1;				//Push back of hygrogen star1
float KH2;				//Push back of hygrogen star2
float KH1RadiusAdjustmentFactor;		//How fast kH1 adjusts with delta radius
float KH2RadiusAdjustmentFactor;		//How fast kH2 adjusts with delta radius
float KRH;				//Push back reduction of hygrogen
int NumberOfElements;
float MaxInitialElementSpeed;		//This shakes up the cubic starting positions so the star can settle down into a natural configuration.
float4 InitialSpin1, InitialSpin2;	//Initial spins of the 2 stars
float DampAmount;
float DampTime;
int NumberOfDampIncriments;
float DampRestTime;
float SpinRestTime;
float RadiusAdjustTime;
float RadiusAdjustRestTime;
float Dt;
int DrawRate;

//Globals to be set by the setRunParameters function
double SystemLengthConverterToKilometers;
double SystemMassConverterToKilograms;
double SystemTimeConverterToSeconds;
int NumberOfElementsInStar1, NumberOfElementsInStar2;

void createFolderForNewStars()
{   	
	//Create output folder to store the stars
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
	else					monthday = smonth.str() + "-" + sday.str() + "-" + stimeHour.str() + ":" + stimeMin.str();
	string foldernametemp = "Stars:" + monthday;
	const char *foldername = foldernametemp.c_str();
	mkdir(foldername , S_IRWXU|S_IRWXG|S_IRWXO);
	chdir(foldername);
	
	FILE *fileIn;
	FILE *fileOut;
	long sizeOfFile;
  	char *buffer;
    
    	//Copying the BuildSetup file into the stars' folder
	fileIn = fopen("../BuildSetup", "rb");
	fseek (fileIn , 0 , SEEK_END);
  	sizeOfFile = ftell(fileIn);
  	rewind (fileIn);
  	buffer = (char*)malloc(sizeof(char)*sizeOfFile);
  	fread (buffer, 1, sizeOfFile, fileIn);
	fileOut = fopen("BuildSetup", "wb");
	fwrite (buffer, 1, sizeOfFile, fileOut);
	fclose(fileIn);
	fclose(fileOut);

	//Copying the build code into the stars' folder
	fileIn = fopen("../StarBuilder.cu", "rb");
	fseek (fileIn , 0 , SEEK_END);
  	sizeOfFile = ftell(fileIn);
  	rewind (fileIn);
  	buffer = (char*)malloc(sizeof(char)*sizeOfFile);
  	fread (buffer, 1, sizeOfFile, fileIn);
	fileOut = fopen("StarBuilder.cu", "wb");
	fwrite (buffer, 1, sizeOfFile, fileOut);
	fclose(fileIn);
	fclose(fileOut);

	//Copying the build code header file into the stars' folder
	fileIn = fopen("../binaryStar.h", "rb");
	fseek (fileIn , 0 , SEEK_END);
  	sizeOfFile = ftell(fileIn);
  	rewind (fileIn);
  	buffer = (char*)malloc(sizeof(char)*sizeOfFile);
  	fread (buffer, 1, sizeOfFile, fileIn);
	fileOut = fopen("binaryStar.h", "wb");
	fwrite (buffer, 1, sizeOfFile, fileOut);
	fclose(fileIn);
	fclose(fileOut);
	free (buffer);
}

void readBuildParameters()
{
	ifstream data;
	string name;
	
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
		data >> TargetRadiusStar1;
		getline(data,name,'=');
		data >> RadiusOfCore1;
		getline(data,name,'=');
		data >> TargetRadiusStar2;
		getline(data,name,'=');
		data >> RadiusOfCore2;
		getline(data,name,'=');
		data >> RadiusTolerance;
		
		getline(data,name,'=');
		data >> MassOfSun;
		getline(data,name,'=');
		data >> MassOfStar1;
		getline(data,name,'=');
		data >> MassOfCore1;
		getline(data,name,'=');
		data >> MassOfStar2;
		getline(data,name,'=');
		data >> MassOfCore2;
		
		getline(data,name,'=');
		data >> DensityOfHydrogenGas;
		getline(data,name,'=');
		data >> KH1;
		getline(data,name,'=');
		data >> KH2;
		getline(data,name,'=');
		data >> KH1RadiusAdjustmentFactor;
		getline(data,name,'=');
		data >> KH2RadiusAdjustmentFactor;
		getline(data,name,'=');
		data >> KRH;
		
		getline(data,name,'=');
		data >> NumberOfElements;
		
		getline(data,name,'=');
		data >> MaxInitialElementSpeed;
		
		getline(data,name,'=');
		data >> InitialSpin1.x;
		getline(data,name,'=');
		data >> InitialSpin1.y;
		getline(data,name,'=');
		data >> InitialSpin1.z;
		getline(data,name,'=');
		data >> InitialSpin1.w;
		
		getline(data,name,'=');
		data >> InitialSpin2.x;
		getline(data,name,'=');
		data >> InitialSpin2.y;
		getline(data,name,'=');
		data >> InitialSpin2.z;
		getline(data,name,'=');
		data >> InitialSpin2.w;
		
		getline(data,name,'=');
		data >> DampAmount;
		getline(data,name,'=');
		data >> DampTime;
		getline(data,name,'=');
		data >> NumberOfDampIncriments;
		getline(data,name,'=');
		data >> DampRestTime;
		
		getline(data,name,'=');
		data >> SpinRestTime;
		
		getline(data,name,'=');
		data >> RadiusAdjustTime;
		getline(data,name,'=');
		data >> RadiusAdjustRestTime;
		
		getline(data,name,'=');
		data >> Dt;
		
		getline(data,name,'=');
		data >> DrawRate;
	}
	else
	{
		printf("\nTSU Error could not open run or root Setup file\n");
		exit(0);
	}
	data.close();
}

void generateAndSaveRunParameters()
{
	double massOfHydrogenGasElement;
	double baseDiameterOfHydrogenGasElement;
	double totalMassOfElements, massOfElementsStar1, massOfElementsStar2;
	
	MassOfStar1 *= MassOfSun;
	MassOfStar2 *= MassOfSun;
	MassOfCore1 *= MassOfSun;
	MassOfCore2 *= MassOfSun;
	TargetRadiusStar1 *= RadiusOfSun;
	TargetRadiusStar2 *= RadiusOfSun;
	RadiusOfCore1 *= RadiusOfSun;
	RadiusOfCore2 *= RadiusOfSun;
	
	massOfElementsStar1 = MassOfStar1 - MassOfCore1;
	massOfElementsStar2 = MassOfStar2 - MassOfCore2;
	
	totalMassOfElements = massOfElementsStar1 + massOfElementsStar2;
	
	//The mass of an element is just the total mass divided by the number of elements used.
	massOfHydrogenGasElement = totalMassOfElements/((double)NumberOfElements);
	
	//We will use the mass of a hydrogen gas element as one unit of mass. 
	//The following constant will convert system masses up to kilograms by multipling 
	//or convert kilograms down to system units by dividing.
	SystemMassConverterToKilograms = massOfHydrogenGasElement;
	
	NumberOfElementsInStar1 = (massOfElementsStar1/totalMassOfElements)*(double)NumberOfElements;
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
	
	//Putting things with mass into our units. Taking kilograms into our units.
	MassOfStar1 /= SystemMassConverterToKilograms;
	MassOfCore1 /= SystemMassConverterToKilograms;
	MassOfStar2 /= SystemMassConverterToKilograms;
	MassOfCore2 /= SystemMassConverterToKilograms;
	
	//Putting things with length into our units. Taking kilometers into our units.
	TargetRadiusStar1 /= SystemLengthConverterToKilometers;
	RadiusOfCore1 /= SystemLengthConverterToKilometers;
	TargetRadiusStar2 /= SystemLengthConverterToKilometers;
	RadiusOfCore2 /= SystemLengthConverterToKilometers;
	
	//Putting times into our units. Taking days (24 hours) to our units.
	DampTime *= (60.0*60.0*24.0)/SystemTimeConverterToSeconds;
	DampRestTime *= (60.0*60.0*24.0)/SystemTimeConverterToSeconds;
	SpinRestTime *= (60.0*60.0*24.0)/SystemTimeConverterToSeconds;
	
	//Putting things dealing with linear velosity into our units. Taking kilometers/second into our units.
	MaxInitialElementSpeed *= SystemTimeConverterToSeconds/SystemLengthConverterToKilometers;
	
	
	//Putting Initial Angular Velocities into our units. Taking revolutions/hour into our units.
	InitialSpin1.w *= SystemTimeConverterToSeconds/3600.0;
	InitialSpin2.w *= SystemTimeConverterToSeconds/3600.0;
	
	//Putting push back parameters into our units. kilograms*kilometersE-1*secondsE-2 into our units.
	KH1 *= SystemTimeConverterToSeconds*SystemTimeConverterToSeconds*SystemLengthConverterToKilometers/SystemMassConverterToKilograms;
	KH2 *= SystemTimeConverterToSeconds*SystemTimeConverterToSeconds*SystemLengthConverterToKilometers/SystemMassConverterToKilograms;
	
	FILE *RunParametersFile;
	RunParametersFile = fopen("RunParameters", "wb");
		fprintf(RunParametersFile, "\n SystemLengthConverterToKilometers = %e", SystemLengthConverterToKilometers);
		fprintf(RunParametersFile, "\n SystemMassConverterToKilograms = %e", SystemMassConverterToKilograms);
		fprintf(RunParametersFile, "\n SystemTimeConverterToSeconds = %e", SystemTimeConverterToSeconds);
	
		fprintf(RunParametersFile, "\n NumberOfElementsInStar1 = %d", NumberOfElementsInStar1);
		fprintf(RunParametersFile, "\n NumberOfElementsInStar2 = %d", NumberOfElementsInStar2);
	fclose(RunParametersFile);
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
	PosCPU = (float4*)malloc(NumberOfElements*sizeof(float4));
	VelCPU = (float4*)malloc(NumberOfElements*sizeof(float4));
	//ForceCPU = (float4*)malloc(NumberOfElements*sizeof(float4));
	
	PosCoreCPU = (float4*)malloc(2*sizeof(float4));
	VelCoreCPU = (float4*)malloc(2*sizeof(float4));
	//ForceCoreCPU = (float4*)malloc(2*sizeof(float4));
	
	cudaMalloc((void**)&PosGPU, NumberOfElements*sizeof(float4));
	errorCheck("cudaMalloc Pos");
	cudaMalloc((void**)&VelGPU, NumberOfElements*sizeof(float4));
	errorCheck("cudaMalloc Vel");
	cudaMalloc((void**)&ForceGPU, NumberOfElements*sizeof(float4));
	
	errorCheck("cudaMalloc Force");
	cudaMalloc((void**)&PosCoreGPU, 2*sizeof(float4));
	errorCheck("cudaMalloc Pos");
	cudaMalloc((void**)&VelCoreGPU, 2*sizeof(float4));
	errorCheck("cudaMalloc Vel");
	//cudaMalloc((void**)&ForceCoreGPU, 2*sizeof(float4));
	//errorCheck("cudaMalloc Force");
	
	cudaMalloc((void**)&CoreForceSumGPU, 6*sizeof(double));
	errorCheck("cudaMalloc CoreForceSumGPU");
}

void cleanUp()
{
	free(PlaceHolder);
	free(PosCPU);
	free(VelCPU);
	free(PosCoreCPU);
	free(VelCoreCPU);
	//free(ForceCoreCPU); 
	//free(ForceCPU);
	
	cudaFree(PosGPU);
	cudaFree(VelGPU);
	cudaFree(ForceGPU);
	cudaFree(PosCoreGPU);
	cudaFree(VelCoreGPU);
	//cudaFree(ForceCoreGPU);
	cudaFree(CoreForceSumGPU);
}

int createRawStar(int starNumber)
{
	//int cubeStart;
	int elementStart, elementStop;
	float coreRadius;
	int x, y, z;
	float mag, speed;
	//float elementDiameter;
	int element, cubeLayer;
	float kh;
	time_t t;
	
	if(starNumber == 1)
	{
		elementStart = 0;
		elementStop = NumberOfElementsInStar1;
		coreRadius = RadiusOfCore1;
		kh = KH1;
		PosCoreCPU[0].x = 0.0;
		PosCoreCPU[0].y = 0.0;
		PosCoreCPU[0].z = 0.0;
		PosCoreCPU[0].w = MassOfCore1;
		VelCoreCPU[0].x = 0.0;
		VelCoreCPU[0].y = 0.0;
		VelCoreCPU[0].z = 0.0;
		VelCoreCPU[0].w = RadiusOfCore1;
	}
	if(starNumber == 2)
	{
		elementStart = NumberOfElementsInStar1;
		elementStop = NumberOfElements;
		coreRadius = RadiusOfCore2;
		kh = KH2;
		PosCoreCPU[1].x = 0.0;
		PosCoreCPU[1].y = 0.0;
		PosCoreCPU[1].z = 0.0;
		PosCoreCPU[1].w = MassOfCore2;
		VelCoreCPU[1].x = 0.0;
		VelCoreCPU[1].y = 0.0;
		VelCoreCPU[1].z = 0.0;
		VelCoreCPU[1].w = RadiusOfCore2;
	}
	
	//elementDiameter = 1.0;
	// We are going to set the core at (0,0,0) then place elements in a cubic grid around it. Each element radius is 1 so we will walk out in units of 1.
	cubeLayer = (int)coreRadius + 1; // This is the size of the cube the core takes up.    
	
	element = elementStart;
	while(element < elementStop)
	{
		cubeLayer++;
		x = -cubeLayer;
		for(y = -cubeLayer; y <= cubeLayer; y++)
		{
			for(z = -cubeLayer; z <= cubeLayer; z++)
			{
				if(element < elementStop)
				{
					PosCPU[element].x = (float)x;
					PosCPU[element].y = (float)y;
					PosCPU[element].z = (float)z;
					PosCPU[element].w = kh;
					element++;
				}
				else break;
			}
		}
	
		x = cubeLayer;
		for(y = -cubeLayer; y <= cubeLayer; y++)
		{
			for(z = -cubeLayer; z <= cubeLayer; z++)
			{
				if(element < elementStop)
				{
					PosCPU[element].x = (float)x;
					PosCPU[element].y = (float)y;
					PosCPU[element].z = (float)z;
					PosCPU[element].w = kh;
					element++;
				}
				else break;
			}
		}
	
		y = -cubeLayer;
		for(x = -cubeLayer + 1; x <= cubeLayer - 1; x++)
		{
			for(z = -cubeLayer; z <= cubeLayer; z++)
			{
				if(element < elementStop)
				{
					PosCPU[element].x = (float)x;
					PosCPU[element].y = (float)y;
					PosCPU[element].z = (float)z;
					PosCPU[element].w = kh;
					element++;
				}
				else break;
			}
		}
	
		y = cubeLayer;
		for(x = -cubeLayer + 1; x <= cubeLayer - 1; x++)
		{
			for(z = -cubeLayer; z <= cubeLayer; z++)
			{
				if(element < elementStop)
				{
					PosCPU[element].x = (float)x;
					PosCPU[element].y = (float)y;
					PosCPU[element].z = (float)z;
					PosCPU[element].w = kh;
					element++;
				}
				else break;
			}
		}
	
		z = -cubeLayer;
		for(x = -cubeLayer + 1; x <= cubeLayer - 1; x++)
		{
			for(y = -cubeLayer + 1; y <= cubeLayer - 1; y++)
			{
				if(element < elementStop)
				{
					PosCPU[element].x = (float)x;
					PosCPU[element].y = (float)y;
					PosCPU[element].z = (float)z;
					PosCPU[element].w = kh;
					element++;
				}
				else break;
			}
		}
	
		z = cubeLayer;
		for(x = -cubeLayer + 1; x <= cubeLayer - 1; x++)
		{
			for(y = -cubeLayer + 1; y <= cubeLayer - 1; y++)
			{
				if(element < elementStop)
				{
					PosCPU[element].x = (float)x;
					PosCPU[element].y = (float)y;
					PosCPU[element].z = (float)z;
					PosCPU[element].w = kh;
					element++;
				}
				else break;
			}
		}
	}
	
	// Setting the randum number generater seed.
	srand((unsigned) time(&t));
	
	// Giving each particle a randium velocity to shake things up a little.
	for(int i = elementStart; i < elementStop; i++)
	{
		VelCPU[i].x = ((float)rand()/(float)RAND_MAX)*2.0 - 1.0;;
		VelCPU[i].y = ((float)rand()/(float)RAND_MAX)*2.0 - 1.0;;
		VelCPU[i].z = ((float)rand()/(float)RAND_MAX)*2.0 - 1.0;;
		mag = sqrt(VelCPU[i].x*VelCPU[i].x + VelCPU[i].y*VelCPU[i].y + VelCPU[i].z*VelCPU[i].z);
		speed = ((float)rand()/(float)RAND_MAX)*MaxInitialElementSpeed;
		VelCPU[i].x *= speed/mag;
		VelCPU[i].y *= speed/mag;
		VelCPU[i].z *= speed/mag;
		
		VelCPU[i].w = 0.0;
	}
	
	return(1);
}

int centerStar(int starNumber)
{
	float4 posAdjust, velAdjust;
	int elementStart, elementStop;
	
	if(starNumber == 1)
	{
		elementStart = 0;
		elementStop = NumberOfElementsInStar1;
		posAdjust.x = PosCoreCPU[0].x;
		posAdjust.y = PosCoreCPU[0].y;
		posAdjust.z = PosCoreCPU[0].z;
		velAdjust.x = VelCoreCPU[0].x;
		velAdjust.y = VelCoreCPU[0].y;
		velAdjust.z = VelCoreCPU[0].z;
		PosCoreCPU[0].x = 0.0;
		PosCoreCPU[0].y = 0.0;
		PosCoreCPU[0].z = 0.0;
		VelCoreCPU[0].x = 0.0;
		VelCoreCPU[0].y = 0.0;
		VelCoreCPU[0].z = 0.0;
	}
	if(starNumber == 2)
	{
		elementStart = NumberOfElementsInStar1;
		elementStop = NumberOfElements;
		posAdjust.x = PosCoreCPU[0].x;
		posAdjust.y = PosCoreCPU[0].y;
		posAdjust.z = PosCoreCPU[0].z;
		velAdjust.x = VelCoreCPU[0].x;
		velAdjust.y = VelCoreCPU[0].y;
		velAdjust.z = VelCoreCPU[0].z;
		PosCoreCPU[1].x = 0.0;
		PosCoreCPU[1].y = 0.0;
		PosCoreCPU[1].z = 0.0;
		VelCoreCPU[1].x = 0.0;
		VelCoreCPU[1].y = 0.0;
		VelCoreCPU[1].z = 0.0;
	}
	
	for(int i = elementStart; i < elementStop; i++)
	{
    		PosCPU[i].x -= posAdjust.x;
    		PosCPU[i].y -= posAdjust.y;
    		PosCPU[i].z -= posAdjust.z;
    		
    		VelCPU[i].x -= velAdjust.x;
    		VelCPU[i].y -= velAdjust.y;
    		VelCPU[i].z -= velAdjust.z;
	}
	
	return(1);
}

void spinStar(int starNumber)
{
	float3	n;	//Unit vector perpendicular to the plane of spin
	float 	mag;
	float 	assumeZero = 0.0000001;
	float4 spinVector;
	int elementStart, elementStop;
	
	if(starNumber == 1)
	{
		elementStart = 0;
		elementStop = NumberOfElementsInStar1;
		spinVector.x = InitialSpin1.x;
		spinVector.y = InitialSpin1.y;
		spinVector.z = InitialSpin1.z;
		spinVector.w = InitialSpin1.w;
	}
	if(starNumber == 2)
	{
		elementStart = NumberOfElementsInStar1;
		elementStop = NumberOfElements;
		spinVector.x = InitialSpin2.x;
		spinVector.y = InitialSpin2.y;
		spinVector.z = InitialSpin2.z;
		spinVector.w = InitialSpin2.w;
	}
	
	//Making sure the spin vector is a unit vector
	mag = sqrt(spinVector.x*spinVector.x + spinVector.y*spinVector.y + spinVector.z*spinVector.z);
	if(assumeZero < mag)
	{
		spinVector.x /= mag;
		spinVector.y /= mag;
		spinVector.z /= mag;
	}
	else 
	{
		printf("\nTSU Error: In spinStar. The spin direction vector is zero.\n");
		exit(0);
	}
	
	for(int i = elementStart; i < elementStop; i++)
	{
		//Creating a vector from the center of mass to the point
		float magsquared = PosCPU[i].x*PosCPU[i].x + PosCPU[i].y*PosCPU[i].y + PosCPU[i].z*PosCPU[i].z;
		float spinDot = spinVector.x*PosCPU[i].x + spinVector.y*PosCPU[i].y + spinVector.z*PosCPU[i].z;
		float perpendicularDistance = sqrt(magsquared - spinDot*spinDot);
		float perpendicularVelocity = spinVector.w*2.0*Pi*perpendicularDistance;
		
		//finding unit vector perpendicular to both the position vector and the spin vector
		n.x =  (spinVector.y*PosCPU[i].z - spinVector.z*PosCPU[i].y);
		n.y = -(spinVector.x*PosCPU[i].z - spinVector.z*PosCPU[i].x);
		n.z =  (spinVector.x*PosCPU[i].y - spinVector.y*PosCPU[i].x);
		mag = sqrt(n.x*n.x + n.y*n.y + n.z*n.z);
		if(mag != 0.0)
		{
			n.x /= mag;
			n.y /= mag;
			n.z /= mag;
				
			//Spining the element
			VelCPU[i].x += perpendicularVelocity*n.x;
			VelCPU[i].y += perpendicularVelocity*n.y;
			VelCPU[i].z += perpendicularVelocity*n.z;
		}
	}		
}

double getStarRadius(int starNumber)
{
	double starRadius;
	double radius, radiusSum, tempRadius;
	int used[NumberOfElements];
	int i,j;
	int elementStart, elementStop;
	int count, numberToSum;
	float coreRadius;
	
	if(starNumber == 1)
	{
		elementStart = 0;
		elementStop = NumberOfElementsInStar1;
		coreRadius = RadiusOfCore1;
	}
	if(starNumber == 2)
	{
		elementStart = NumberOfElementsInStar1;
		elementStop = NumberOfElements;
		coreRadius = RadiusOfCore2;
	}
	
	for(i = 0; i < NumberOfElements; i++)
	{
		used[i] = 0;
	}
	
	numberToSum = 100;
	radiusSum = 0.0;
	count = 0;
	for(j = 0; j < numberToSum; j++)
	{
		radius = -1.0;
		for(i = elementStart; i < elementStop; i++)
		{
			tempRadius = sqrt(PosCPU[i].x*PosCPU[i].x + PosCPU[i].y*PosCPU[i].y + PosCPU[i].z*PosCPU[i].z);
			if(radius < tempRadius && used[i] == 0) 
			{
				radius = tempRadius;
			}
		}
		if(radius != -1) 
		{
			count++;
			radiusSum += radius;
		}
	}
	if(count == 0)
	{
	 	starRadius = coreRadius; // If this happens there are not hydrogen elements in the star.
	}
	else
	{
		starRadius = radiusSum/((float)count);
	}
	
	return(starRadius);
}

void drawPicture()
{	
	float seperation = 10.0;
	
	glPointSize(2.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	glColor3d(1.0,1.0,0.0);
	glBegin(GL_POINTS);
 		for(int i = 0; i < NumberOfElementsInStar1; i++)
		{
			glVertex3f(PosCPU[i].x + seperation, PosCPU[i].y, PosCPU[i].z);
		}
	glEnd();
	
	glColor3d(0.0,1.0,0.0);
	glBegin(GL_POINTS);
 		for(int i = NumberOfElementsInStar1; i < NumberOfElements; i++)
		{
			glVertex3f(PosCPU[i].x - seperation, PosCPU[i].y, PosCPU[i].z);
		}
	glEnd();
	
	glPointSize(5.0);
	glColor3d(1.0,1.0,1.0);
	glBegin(GL_POINTS);
		glVertex3f(PosCoreCPU[0].x + seperation, PosCoreCPU[0].y, PosCoreCPU[0].z);
		glVertex3f(PosCoreCPU[1].x - seperation, PosCoreCPU[1].y, PosCoreCPU[1].z);
	glEnd();
	
	glutSwapBuffers();
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

__device__ float4 calculateElementElementForce(float4 posMe, float4 posYou, float4 velMe, float4 velYou, float KRH, float pi)
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
		/* 
		invr = 1.0f/r;
		
		force_mag = 1.0f/r2;  // G = 1 and mass of each elemnet =1. So G*mass1*mass2 = 1.
		
		force.x = (dp.x*invr)*force_mag;
		force.y = (dp.y*invr)*force_mag;
		force.z = (dp.z*invr)*force_mag;
		*/ 
		// All the above code can be replaced with dividing by r3. I left the code for readablity.
		
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
		
		//if(inOut <= 0) 	force_mag  = 1.0f - KH*(1.0f - r2);  // Area push back what we used on the Moon code
		//else 			force_mag  = 1.0f - KRH*KH*(1.0f - r2);
		
		//if(inOut <= 0) 	force_mag  = 1.0f - KH*(1.0f - r3);  // Volume push back analogus to what we did on the Moon code
		//else 			force_mag  = 1.0f - KRH*KH*(1.0f - r3);
		
		//if(inOut <= 0) 	force_mag  = 1.0f - KH*(pi/4.0f)*(1.0f - r2);  // Exact area push back
		//else 			force_mag  = 1.0f - KRH*KH*(pi/4.0f)*(1.0f - r2);
		
		if(inOut <= 0) 		force_mag  = 1.0f - 0.5*(posYou.w + posMe.w)*pi*(1.0f/6.0f + r3/3.0f - r2/2.0f);
		else 			force_mag  = 1.0f - KRH*0.5*(posYou.w + posMe.w)*pi*(1.0f/6.0f + r3/3.0f - r2/2.0f);
		
		force.x = (dp.x*invr)*force_mag;
		force.y = (dp.y*invr)*force_mag;
		force.z = (dp.z*invr)*force_mag;
	}
	else // Hopefully this line of code never gets reached.
	{
		dv.x = velYou.x - velMe.x;
		dv.y = velYou.y - velMe.y;
		dv.z = velYou.z - velMe.z;
		if(0.0f < (dv.x + dv.y + dv.z)) // Hopefully if it they do not have the same velocity they will drift past setting right on top of eachother.
		{
			force.x = 0.0f;
			force.y = 0.0f;
			force.z = 0.0f;
			printf("\n TSU error: Elements on top of each other in calculateElementElementForce \n");
		}
		else // If they have the same velocity we will need to kick them off of died center. This sceem will work but I think I'll kill the program and check it.
		{
			force.x = 0.0000001f;
			force.y = 0.0f;
			force.z = 0.0f;
			printf("\n TSU error: Elements stuck on top of each other in calculateElementElementForce \n");
		}
	}
	return(force);
}

__device__ float4 calculateElementCoreForce(float4 posMe, float4 posCore, float4 velMe, float4 velCore, float KRH, float pi)
{
	float4 dp, dv, force;
	float r, r2, r3, invr, inOut, force_mag;
	
	dp.x = posCore.x - posMe.x;
	dp.y = posCore.y - posMe.y;
	dp.z = posCore.z - posMe.z;
	r2 = dp.x*dp.x + dp.y*dp.y + dp.z*dp.z;
	r = sqrt(r2);
	r3 = r2*r;
	
	if(velCore.w + 0.5f <= r)
	{
		/* 
		invr = 1.0f/r;
		
		force_mag = massCore/r2;  // G = 1 and mass of each elemnet =1. So G*mass1*mass2 = massCore.
		
		force.x = (dp.x*invr)*force_mag;
		force.y = (dp.y*invr)*force_mag;
		force.z = (dp.z*invr)*force_mag;
		*/ 
		// All the above code can be replaced with dividing by r3. I left the code for readablity.
		
		invr = 1.0f/r3;
		
		force.x = posCore.w*(dp.x*invr);  // posCore.w holds the mass of the core
		force.y = posCore.w*(dp.y*invr);
		force.z = posCore.w*(dp.z*invr);
		
	}
	else if(0.0f < r)
	{
		invr = 1.0f/r;
		
		dv.x = velCore.x - velMe.x;
		dv.y = velCore.y - velMe.y;
		dv.z = velCore.z - velMe.z;
		inOut = dp.x*dv.x + dp.y*dv.y + dp.z*dv.z;
		float overLap = r - velCore.w;  // velCore.w holds the radius of the core
		if(-0.5f < overLap)
		{
			if(inOut <= 0) 		force_mag  = posCore.w/r2 - posMe.w*pi*(4.0f/3.0f)*(0.5f-overLap);
			else 			force_mag  = posCore.w/r2 - KRH*posMe.w*pi*(4.0f/3.0f)*(0.5f-overLap);
		}
		else
		{
			if(inOut <= 0) 		force_mag  = posCore.w/r2 - posMe.w*pi*(4.0f/3.0f);
			else 			force_mag  = posCore.w/r2 - KRH*posMe.w*pi*(4.0f/3.0f);
		}
		
		force.x = (dp.x*invr)*force_mag;
		force.y = (dp.y*invr)*force_mag;
		force.z = (dp.z*invr)*force_mag;
	}
	else // Hopefully this line of code never gets reached.
	{
		dv.x = velCore.x - velMe.x;
		dv.y = velCore.y - velMe.y;
		dv.z = velCore.z - velMe.z;
		if(0.0f < (dv.x + dv.y + dv.z)) // Hopefully if it they do not have the same velocity they will drift past setting right on top of eachother.
		{
			force.x = 0.0f;
			force.y = 0.0f;
			force.z = 0.0f;
			printf("\n TSU error: Elements on top of each other in calculateElementCoreForce \n");
		}
		else // If they have the same velocity we will need to kick them off of died center. This sceem will work but I think I'll kill the program to see if I need to patch the code.
		{
			force.x = 0.0000001f;
			force.y = 0.0f;
			force.z = 0.0f;
			printf("\n TSU error: Elements stuck on top of each other in calculateElementCoreForce \n");
		}
	}
	return(force);
}

__global__ void getForcesSeperate(float4 *pos, float4 *vel, float4 *force, float4 *corePos, float4 *coreVel, double *coreForceSum, float KRH, int numberOfElementsInStar1, int numberOfElements, float pi)
{
	int id, ids, i, j;
	float4 posMe, velMe;
	float4 partialForce;
	double forceSumX, forceSumY, forceSumZ;
	
	__shared__ float4 shPos[BLOCKSIZE];
	__shared__ float4 shVel[BLOCKSIZE];

	id = threadIdx.x + blockDim.x*blockIdx.x;
	if(numberOfElements <= id)
	{
		printf("\n TSU error: id out of bounds in getForcesSeperate. \n");
	}
		
	forceSumX = 0.0f;
	forceSumY = 0.0f;
	forceSumZ = 0.0f;
		
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
			if((id < numberOfElementsInStar1 && ids < numberOfElementsInStar1) || (numberOfElementsInStar1 <= id && numberOfElementsInStar1 <= ids))
			{
				if(id != ids)
				{
					partialForce = calculateElementElementForce(posMe, shPos[i], velMe, shVel[i], KRH, pi);
					forceSumX += partialForce.x;
					forceSumY += partialForce.y;
					forceSumZ += partialForce.z;
				}
			}
		}
		__syncthreads();
	}
	
	// Adding the force of the cores. I'm atomic summing the cores themselves so I do not have to write a kernal for them.
	if(id < numberOfElementsInStar1) 
	{
		partialForce = calculateElementCoreForce(posMe, corePos[0], velMe, coreVel[0], KRH, pi);
		atomicAdd( &coreForceSum[0], -(double)partialForce.x );
		atomicAdd( &coreForceSum[1], -(double)partialForce.y );
		atomicAdd( &coreForceSum[2], -(double)partialForce.z );
	}
	else
	{
		partialForce = calculateElementCoreForce(posMe, corePos[1], velMe, coreVel[1], KRH, pi);
		atomicAdd( &coreForceSum[3], -(double)partialForce.x );
		atomicAdd( &coreForceSum[4], -(double)partialForce.y );
		atomicAdd( &coreForceSum[5], -(double)partialForce.z );
	}
	forceSumX += partialForce.x;
	forceSumY += partialForce.y;
	forceSumZ += partialForce.z;
	
	force[id].x = forceSumX;
	force[id].y = forceSumY;
	force[id].z = forceSumZ;
}

__global__ void moveBodiesDamped(float4 *pos, float4 *vel, float4 *force, float4 *posCore, float4 *velCore, double *coreForceSum, float dt, float damp)
{  
    	int id = threadIdx.x + blockDim.x*blockIdx.x;

	// There is no need to divide by mass in the velocity lines because the mass of an element is 1.
	vel[id].x += (force[id].x-damp*vel[id].x)*dt;
	vel[id].y += (force[id].y-damp*vel[id].y)*dt;
	vel[id].z += (force[id].z-damp*vel[id].z)*dt;

	pos[id].x += vel[id].x*dt;
	pos[id].y += vel[id].y*dt;
	pos[id].z += vel[id].z*dt;
	
	// Getting two threads to do a little extra work and do the core too.
	if(id == 0)
	{
		velCore[0].x += (coreForceSum[0] - damp*velCore[0].x)*dt/posCore[0].w;
		velCore[0].y += (coreForceSum[1] - damp*velCore[0].y)*dt/posCore[0].w;
		velCore[0].z += (coreForceSum[2] - damp*velCore[0].z)*dt/posCore[0].w;

		posCore[0].x += velCore[0].x*dt;
		posCore[0].y += velCore[0].y*dt;
		posCore[0].z += velCore[0].z*dt;
	}
	else if(id == 1)
	{
		velCore[1].x += (coreForceSum[3] - damp*velCore[1].x)*dt/posCore[1].w;
		velCore[1].y += (coreForceSum[4] - damp*velCore[1].y)*dt/posCore[1].w;
		velCore[1].z += (coreForceSum[5] - damp*velCore[1].z)*dt/posCore[1].w;

		posCore[1].x += velCore[1].x*dt;
		posCore[1].y += velCore[1].y*dt;
		posCore[1].z += velCore[1].z*dt;
	}
}

void starNbody(float time, float damp)
{ 
	int   tdraw = 0;
	
	while(time < DampTime)
	{	
		cudaMemset(CoreForceSumGPU, 0, 6*sizeof(double));
		errorCheck("cudaMemset CoreForceSumGPU");
		
		getForcesSeperate<<<GridConfig, BlockConfig>>>(PosGPU, VelGPU, ForceGPU, PosCoreGPU, VelCoreGPU, CoreForceSumGPU, KRH, NumberOfElementsInStar1, NumberOfElements, (float)Pi);
		errorCheck("getForcesSeperate");
		
		moveBodiesDamped<<<GridConfig, BlockConfig>>>(PosGPU, VelGPU, ForceGPU, PosCoreGPU, VelCoreGPU, CoreForceSumGPU, Dt, damp);
		errorCheck("moveBodiesDamped");
	
		if(tdraw == DrawRate) 
		{
			cudaMemcpy(PosCPU, PosGPU, (NumberOfElements)*sizeof(float4), cudaMemcpyDeviceToHost);
			errorCheck("cudaMemcpy Pos draw");
			cudaMemcpy(PosCoreCPU, PosCoreGPU, 2*sizeof(float4), cudaMemcpyDeviceToHost);
			errorCheck("cudaMemcpy PosCore draw");
			drawPicture();
			tdraw = 0;
		}
		tdraw++;
		time += Dt;
	}
}

void recordStartPosVelOfCreatedStars()
{
	FILE *startPosAndVelFile;
	
	startPosAndVelFile = fopen("StartPosAndVel", "wb");
	cudaMemcpy( PosCPU, PosGPU, NumberOfElements *sizeof(float4), cudaMemcpyDeviceToHost );
	errorCheck("cudaMemcpy Pos");
	cudaMemcpy( VelCPU, VelGPU, NumberOfElements *sizeof(float4), cudaMemcpyDeviceToHost );
	errorCheck("cudaMemcpy Vel");
	
	fwrite(PosCPU, sizeof(float4), NumberOfElements, startPosAndVelFile);
	fwrite(VelCPU, sizeof(float4), NumberOfElements, startPosAndVelFile);
	
	fclose(startPosAndVelFile);
}

void recordStarStats()
{
	FILE *starStatsFile;
	double massStar1, radiusStar1, densityStar1;
	
	cudaMemcpy( PosCPU, PosGPU, NumberOfElements *sizeof(float4), cudaMemcpyDeviceToHost );
	errorCheck("cudaMemcpy Pos");
	cudaMemcpy( VelCPU, VelGPU, NumberOfElements *sizeof(float4), cudaMemcpyDeviceToHost );
	errorCheck("cudaMemcpy Vel");
	
	massStar1 = NumberOfElementsInStar1*SystemMassConverterToKilograms;
	radiusStar1 = getStarRadius(1);
	radiusStar1 *= SystemLengthConverterToKilometers;
	densityStar1 = massStar1/((4.0/3.0)*Pi*radiusStar1*radiusStar1*radiusStar1);
	
	starStatsFile = fopen("StarStats", "wb");
		fprintf(starStatsFile, " The conversion parameters to take you to and from our units to kilograms, kilometers, seconds follow\n");
		fprintf(starStatsFile, " Mass in our units is the mass of an element. In other words the mass of an element is one.\n");
		fprintf(starStatsFile, " Length in our units is the starting diameter of an element. In other words the staring base diameter of an element is one.\n");
		fprintf(starStatsFile, " Time in our units is set so that the universal gravitational constant is 1.");
		fprintf(starStatsFile, "\n ");
		fprintf(starStatsFile, "\n Our length unit is this many kilometers: %e", SystemLengthConverterToKilometers);
		fprintf(starStatsFile, "\n Our mass unit is this many kilograms: %e", SystemMassConverterToKilograms);
		fprintf(starStatsFile, "\n Our time unit is this many seconds: %e or days %e", SystemTimeConverterToSeconds, SystemTimeConverterToSeconds/(60*60*24));
		fprintf(starStatsFile, "\n KH1 in our units is: %e", KH1);
		fprintf(starStatsFile, "\n ");
		fprintf(starStatsFile, "\n Total number of elements in star1: %d", NumberOfElementsInStar1);
		fprintf(starStatsFile, "\n ");
		fprintf(starStatsFile, "\n Mass of Star1 = %e kilograms", massStar1);
		fprintf(starStatsFile, "\n Radius of Star1 = %e kilometers", radiusStar1);
		fprintf(starStatsFile, "\n Density of star1 = %e kilograms/(cubic kilometers)", densityStar1);
	fclose(starStatsFile);
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
	float damp, time;
	
	// Handling input from the screen.
	sa.sa_handler = signalHandler;
	sigemptyset(&sa.sa_mask);
	sa.sa_flags = SA_RESTART; // Restart functions if interrupted by handler
	if (sigaction(SIGINT, &sa, NULL) == -1)
	{
		printf("\nTSU Error: sigaction error\n");
	}

	// Creating folder to hold the newly created stars and moving into that folder. It also makes a copy of the BiuldSetup file in this folder.
	printf("\n Creating folders for new stars. \n");
	createFolderForNewStars();
	
	// Reading in and saving the build parameters to a file.
	printf("\n Reading build parameters. \n");
	readBuildParameters();
	
	// Creating and saving the run parameters to a file.
	printf("\n Saving run parameters. \n");
	generateAndSaveRunParameters();
	
	// Allocating memory for CPU and GPU.
	printf("\n Allocating memory. \n");
	allocateMemory();

	// Generating raw stars
	printf("\n Generating raw star1. \n");
	createRawStar(1);	
	printf("\n Generating raw star2. \n");
	createRawStar(2);	
	 
	drawPicture();
	//while(1);
	
	// Seting up the GPU.
	printf("\n Setting up GPUs \n");
	deviceSetup();
	
	// Coping the raw stars up to the GPU
	cudaMemcpy(PosGPU, PosCPU, NumberOfElements*sizeof(float4), cudaMemcpyHostToDevice);
	errorCheck("cudaMemcpy Pos up");
	cudaMemcpy(VelGPU, VelCPU, NumberOfElements*sizeof(float4), cudaMemcpyHostToDevice);
	errorCheck("cudaMemcpy Vel up");
	
	cudaMemcpy(PosCoreGPU, PosCoreCPU, 2*sizeof(float4), cudaMemcpyHostToDevice);
	errorCheck("cudaMemcpy PosCore up");
	cudaMemcpy(VelCoreGPU, VelCoreCPU, 2*sizeof(float4), cudaMemcpyHostToDevice);
	errorCheck("cudaMemcpy VelCore up");
	
	// The raw stars are in unnatural positions and have unnatural velocities. The stars need to be run with a damping factor turned on 
	// to let the stars move into naturl configurations. The damp will start high and be reduced to zero
	for(int i = 0; i < NumberOfDampIncriments; i++)
	{
		damp = DampAmount - float(i)*DampAmount/((float)NumberOfDampIncriments);
		time = DampTime/NumberOfDampIncriments;
		printf("\n Damping raw stars interation %d out of %d \n", i, NumberOfDampIncriments);
		starNbody(time, damp);
	}
	
	// Letting any residue from the damping settle out.
	printf("\n Running damp rest\n");
	starNbody(DampRestTime, 0.0);
	
	// Setting the star's core pos and vel to zero and adjusting all it's elements relative to this.
	centerStar(1);
	centerStar(2);
	
	// Spinning stars
	printf("\n Spinning star1. \n");
	spinStar(1);	
	printf("\n Spinning star2. \n");
	spinStar(2);
	
	// Letting any residue from the spinning settle out.
	printf("\n Running spin rest.\n");
	starNbody(SpinRestTime, 0.0);
	
	// Now we need to set KH so that the radii of the stars is correct.
	printf("\n Running radius adjustment. \n");
	float radiusStar1 = getStarRadius(1);
	float radiusStar2 = getStarRadius(2);
	while(RadiusTolerance < abs(TargetRadiusStar1 - radiusStar1) || RadiusTolerance < abs(TargetRadiusStar2 - radiusStar2))
	{
		damp = 0.0;
		time = RadiusAdjustTime;
		for(int i = 0; i < NumberOfElementsInStar1; i++)
		{
			PosCPU[i].w += KH1RadiusAdjustmentFactor*(TargetRadiusStar1 - radiusStar1);
		}
		for(int i = NumberOfElementsInStar1; i < NumberOfElements; i++)
		{
			PosCPU[i].w += KH2RadiusAdjustmentFactor*(TargetRadiusStar2 - radiusStar2);
		}
		starNbody(time, damp);
		radiusStar1 = getStarRadius(1);
		radiusStar2 = getStarRadius(2);
	}
	
	// Letting any residue from the radius adjustment settle out.
	printf("\n Running radius adjustment rest.\n");
	starNbody(RadiusAdjustRestTime, 0.0);
	
	// Setting the star's core pos and vel to zero and adjusting all it's elements relative to this.
	centerStar(1);
	centerStar(2);
	
	// Saving the stars positions and velocities to a file.	
	printf("\n Saving positions and velocities \n");
	recordStartPosVelOfCreatedStars();  
	
	// Saving any wanted stats about the stars into thier folder.
	printf("\n Recording stats \n");
	recordStarStats();	
	
	// Freeing memory. 	
	printf("\n Cleaning up \n");
	cleanUp();

	printf("\n DONE \n");
	exit(0);
}

//Globals for viewing
//Viewing cropped pyrimid
double ViewBoxSize = 20.0;

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






