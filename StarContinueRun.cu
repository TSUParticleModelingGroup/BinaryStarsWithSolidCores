/*
nvcc StarContinueRun.cu -o StarContinueRun.exe -lglut -lGL -lGLU -lm
nvcc StarContinueRun.cu -o StarContinueRun.exe -lglut -lGL -lGLU -lm --use_fast_math
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

//Root folder to containing the stars to work from. Read in from the comand line.
char BranchFolderName[256] = "";

//Time to add on to the run. Readin from the comand line.
float ContinueRunTime;

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
int RecordRate;
int DrawRate;

//Global that will reset your center of view.
float4 CenterOfView;

//File to hold the position and velocity outputs to make videos and analysis of the run.
FILE *PosAndVelFile;


void openAndReadFiles()
{
	ifstream data;
	string name;
	
	//Moving to the branch run folder.	
	chdir(BranchFolderName);
	
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
	
	//Move back to the main directory.
	chdir("../");
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
	chdir(BranchFolderName);   
	FILE *startFile = fopen("FinalPosVelForce","rb");
	if(startFile == NULL)
	{
		printf("\n\n The FinalPosVelForce file does not exist\n\n");
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

void recordFinalPosVelForce(float time)
{	
	chdir(BranchFolderName);   
	FILE *endFile = fopen("FinalPosVelForce","wb");
	if(endFile == NULL)
	{
		printf("\n\n The FinalPosVelForce could not be created\n\n");
		exit(0);
	}
	fseek(endFile,0,SEEK_SET);
	chdir("../");
	fwrite(&time, sizeof(float), 1, endFile);
	fwrite(PosCPU, sizeof(float4), NumberElements, endFile);
	fwrite(VelCPU, sizeof(float4), NumberElements, endFile);
	fwrite(ForceCPU, sizeof(float4), NumberElements, endFile);
	
	fclose(endFile);
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
	clock_t start_t, end_t;
	double total_t;
	
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
	allocateMemory();
	
	// Reading in the raw stars generated by the build program.
	printf("\n Reading in the stars that were generated in the build program.\n");
	readInTheInitialsStars();
	
	// Draw the intial configuration.
	printf("\n Drawing initial picture.\n");
	drawPicture();
	
	// Seting up the GPU.
	printf("\n Setting up the GPU.\n");
	deviceSetup();
	
	// Running the simulation.
	start_t = clock();
	printf("\n Running the simulation.\n");
	copyStarsUpToGPU();
	time = starNbody(StartTime, StartTime + ContinueRunTime, DT);
	end_t = clock();
	total_t = (double)(end_t - start_t) / CLOCKS_PER_SEC;
   	printf("\nTotal time taken for simulation in this run: %f seconds\n", total_t);
	
	// Saving the the runs final positions and velosities.	
	printf("\n Saving the the runs final positions and velosities.\n");
	copyStarsDownFromGPU();
	recordFinalPosVelForce(time);  
	
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
	else if( argc < 3)
	{
		printf("\n You need to intire an amount of time to add to the run on the comand line\n");
		exit(0);
	}
	else
	{
		strcat(BranchFolderName, argv[1]);
		ContinueRunTime = atof(argv[2]); //Reading time in as days. Need to put in our units after paranter file is read in.
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






