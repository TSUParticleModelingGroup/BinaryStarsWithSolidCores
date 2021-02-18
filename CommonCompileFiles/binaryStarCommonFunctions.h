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

void allocateCPUMemory()
{
	PosCPU   = (float4*)malloc(NumberElements*sizeof(float4));
	VelCPU   = (float4*)malloc(NumberElements*sizeof(float4));
	ForceCPU = (float4*)malloc(NumberElements*sizeof(float4));
}

void cleanUp(int gPUsUsed)
{
	free(PosCPU);
	free(VelCPU);
	free(ForceCPU);
	
	for(int i = 0; i < gPUsUsed; i++)
	{
		cudaFree(PosGPU[i]);
		cudaFree(VelGPU[i]);
		cudaFree(ForceGPU[i]);
	}
}

void copyStarsUpToGPU(int gPUsUsed)
{
	for(int i = 0; i < gPUsUsed; i++)
	{
		cudaSetDevice(i);
		errorCheck("cudaSetDevice");
		cudaMemcpy(PosGPU[i], PosCPU, NumberElements*sizeof(float4), cudaMemcpyHostToDevice);
		errorCheck("cudaMemcpy Pos up");
		cudaMemcpy(VelGPU[i], VelCPU, NumberElements*sizeof(float4), cudaMemcpyHostToDevice);
		errorCheck("cudaMemcpy Vel up");
		cudaMemcpy(ForceGPU[i], ForceCPU, NumberElements*sizeof(float4), cudaMemcpyHostToDevice);
		errorCheck("cudaMemcpy Force up");
	}
}

void copyStarsDownFromGPU()
{
	//Because it is shared above it will only need to be copied from one GPU.
	cudaSetDevice(0);
	errorCheck("cudaSetDevice");
	cudaMemcpy( PosCPU, PosGPU[0], NumberElements*sizeof(float4), cudaMemcpyDeviceToHost );
	errorCheck("cudaMemcpy Pos");
	cudaMemcpy( VelCPU, VelGPU[0], NumberElements*sizeof(float4), cudaMemcpyDeviceToHost );
	errorCheck("cudaMemcpy Vel");
	cudaMemcpy( ForceCPU, ForceGPU[0], NumberElements*sizeof(float4), cudaMemcpyDeviceToHost );
	errorCheck("cudaMemcpy Force");
}

int deviceSetup()
{
	//int access;
	int numberOfGpus;
	int gPUsUsed;
	
	if(NumberElements%BLOCKSIZE != 0)
	{
		printf("\nTSU Error: Number of Particles is not a multiple of the block size \n\n");
		exit(0);
	}
	
	cudaGetDeviceCount(&numberOfGpus);
	errorCheck("cudaGetDeviceCount");
	printf("\n You have %d GPU(s) available\n", numberOfGpus);
	
	if(numberOfGpus == 0)
	{
		printf("\nTSU error: You do not have a GPU. Sorry but you can't run this code.\n");
		exit(0);
	}
	else if(numberOfGpus == 1)
	{
		printf("\n You have 1 GPU in this machine and we will be using it to accelerated this simulation.\n");
		gPUsUsed = 1;
	}
	else if(numberOfGpus == 2 || numberOfGpus == 3) //We are only using even numbers of GPUs so 3 will be down graded to 2
	{
		if(numberOfGpus == 3) 
		{
			printf("\n You have 3 GPUs but we will only use 2. You may need to check and make sure you are using the best GPUs\n");
			gPUsUsed = 2;
		}
		else
		{
			gPUsUsed = 2;
		}
		
		//Seeing if the GPUs can talk to each other.
		/*
		for(int i = 0; i < gPUsUsed; i++)
		{
			for(int j = 0; j < gPUsUsed; j++)
			{
				if(i != j)
				{
					cudaDeviceCanAccessPeer(&access,i,j);
					errorCheck("cudaDeviceCanAccessPeer");
					if(access == 0)
					{
						printf("\nTSU Error: Device %d can not do peer to peer to device %d. Dropping to using only 1 GPU\n", i, j);
						gPUsUsed = 1;
					}
				}
			}
		}
		*/
	}
	else
	{
		if(numberOfGpus != 4) 
		{
			printf("\n You have more than 4 GPUs but we will only use 4. You may need to check and make sure you are using the best GPUs\n");
			gPUsUsed = 4;
		}
		else
		{
			gPUsUsed = 4;
		}
		
		//Seeing if the GPUs can talk to each other.
		/*
		for(int i = 0; i < gPUsUsed; i++)
		{
			for(int j = 0; j < gPUsUsed; j++)
			{
				if(i != j)
				{
					cudaDeviceCanAccessPeer(&access,i,j);
					errorCheck("cudaDeviceCanAccessPeer");
					if(access == 0)
					{
						printf("\nTSU Error: Device %d can not do peer to peer to device %d. Dropping to using only 1 GPU\n", i, j);
						gPUsUsed = 1;
					}
				}
			}
		}
		*/
	}
	
	//gPUsUsed = 2;
	printf("\n You will be using %d GPUs\n", gPUsUsed);
	
	BlockConfig.x = BLOCKSIZE;
	BlockConfig.y = 1;
	BlockConfig.z = 1;
	
	GridConfig.x = ((NumberElements/gPUsUsed)-1)/BlockConfig.x + 1;
	GridConfig.y = 1;
	GridConfig.z = 1;
	
	for(int i = 0; i < gPUsUsed; i++)
	{
		cudaSetDevice(i);
		errorCheck("cudaSetDevice");
		
		//Allocating memory on each GPU.
		cudaMalloc((void**)&PosGPU[i], NumberElements *sizeof(float4));
		errorCheck("cudaMalloc Pos");
		cudaMalloc((void**)&VelGPU[i], NumberElements *sizeof(float4));
		errorCheck("cudaMalloc Vel");
		cudaMalloc((void**)&ForceGPU[i], NumberElements *sizeof(float4));
		errorCheck("cudaMalloc Force");
		
		//Telling current GPU to talk to all the other GPUs.
		/*
		for(int j = 0; j < gPUsUsed; j++)
		{
			if(i != j)
			{
				cudaDeviceEnablePeerAccess(j,0);  //Second value most be zero. It is held by NVIDIA to do future work.
				errorCheck("cudaDeviceEnablePeerAccess");
			}
		}
		*/
	}
	return(gPUsUsed);
}

//Force between two plasma elements
__device__ float4 calculatePlasmaPlasmaForce(float4 posMe, float4 posYou, float4 velMe, float4 velYou, float plasmaPushBackReduction)
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
		// G = 1 and mass of each plasma elemnet = 1. So G*mass1*mass2 = 1.
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
		
		// mass1*mass2*G = 1 is the pull force just holding it constant after they touch.
		// average push times the volume lost is the push force. 0.5*(push1+push2)2pi*(2/3+r+r*r8r/3)
		// (1.0f/6.0f + r3/3.0f - r2/2.0f)
		//if(inOut <= 0) 		force_mag  = 1.0f - (velYou.w + velMe.w)*PI*(1.0f/6.0f + r/4.0 - r3/12.0f);
		//else 			force_mag  = 1.0f - PLASMA_PUSH_BACK_REDUCTION*(velYou.w + velMe.w)*PI*(1.0f/6.0f + r/4.0 - r3/12.0f);
		//if(inOut <= 0) 		force_mag  = 1.0f - (velYou.w + velMe.w)*PI*(1.0f/6.0f + r3/3.0f - r2/2.0f);
		//else 			force_mag  = 1.0f - plasmaPushBackReduction*(velYou.w + velMe.w)*PI*(1.0f/6.0f + r3/3.0f - r2/2.0f);
		
		// *********** doing an area so need to say something about that ****************************
		
		if(inOut <= 0) 		force_mag  = 1.0f - ((velYou.w + velMe.w)*0.5f)*(PI*0.25f)*(1.0f - r2);
		else 			force_mag  = 1.0f - plasmaPushBackReduction*((velYou.w + velMe.w)*0.5f)*(PI*0.25f)*(1.0f - r2);
		
		invr = 1.0f/r;
		force.x = (dp.x*invr)*force_mag;
		force.y = (dp.y*invr)*force_mag;
		force.z = (dp.z*invr)*force_mag;
	}
	else // Hopefully this line of code never gets reached.
	{
		dv.x = velYou.x - velMe.x;
		dv.y = velYou.y - velMe.y;
		dv.z = velYou.z - velMe.z;
		if(0.0f < (dv.x*dv.x + dv.y*dv.y + dv.z*dv.z)) // Hopefully if it they do not have the same velocity they will drift past setting right on top of eachother.
		{
			force.x = 0.0f;
			force.y = 0.0f;
			force.z = 0.0f;
			printf("\n TSU error:Plasma Elements on top of each other in calculatePlasmaPlasmaForce \n");
		}
		else // If they have the same velocity we will need to kick them off of died center. This sceem will work but I think I'll kill the program and check it.
		{
			force.x = 0.0001f;
			force.y = 0.0f;
			force.z = 0.0f;
			printf("\n TSU error: Plasma Elements stuck on top of each other in calculatePlasmaPlasmaForce \n");
		}
	}
	
	return(force);
}

//Force between plasma and core elements
__device__ float4 calculateCorePlasmaForce(int coreFlag, float4 posMe, float4 posYou, float4 velMe, float4 velYou, float4 forceMe, float4 forceYou, float corePushBackReduction)
{
	float4 dp, dv, force;
	float r, r2, r3, invr, inOut, force_mag;
	float gravity, firstTouch, coreMass, corePushBack, maxPushBack, plasmaPushBack;
	
	dp.x = posYou.x - posMe.x;
	dp.y = posYou.y - posMe.y;
	dp.z = posYou.z - posMe.z;
	r2 = dp.x*dp.x + dp.y*dp.y + dp.z*dp.z;
	r = sqrt(r2);
	r3 = r2*r;
	
	firstTouch = (forceYou.w + forceMe.w)*0.5f;  // Distance where the elements first touch
	
	if(coreFlag == 0) 	// Me is the core element.
	{
		coreMass = posMe.w;  
		corePushBack = velMe.w;
		plasmaPushBack = velYou.w;
	}
	else 			// You is the core element.
	{
		coreMass = posYou.w; 
		corePushBack = velYou.w;
		plasmaPushBack = velMe.w;
	}
	
	if(firstTouch <= r)  // force.w holds the diameters so this is when the just touch or greater (only gravity).
	{
		// G = 1 and mass of each plasma elemnet = 1. So G*mass1*mass2 = coreMass.
		invr = 1.0f/r3;
		force.x = coreMass*(dp.x*invr);  // pos.w holds the mass
		force.y = coreMass*(dp.y*invr);
		force.z = coreMass*(dp.z*invr);
	}
	else if(0.0f < r)
	{	
		dv.x = velYou.x - velMe.x;
		dv.y = velYou.y - velMe.y;
		dv.z = velYou.z - velMe.z;
		inOut = dp.x*dv.x + dp.y*dv.y + dp.z*dv.z;
		
		gravity = coreMass/(firstTouch*firstTouch);  //Holding gravity constant at where they touched. G =1 amd mass of plasma = 1.
		maxPushBack = corePushBack + plasmaPushBack*(PI/6.0f);
		
		// Making a linear increasing function from 0 to max.
		if(inOut <= 0) 	force_mag  = gravity - (-maxPushBack*r/firstTouch + maxPushBack);
		else 		force_mag  = gravity - corePushBackReduction*(-maxPushBack*r/firstTouch + maxPushBack);
		
		invr = 1.0f/r;
		force.x = (dp.x*invr)*force_mag;
		force.y = (dp.y*invr)*force_mag;
		force.z = (dp.z*invr)*force_mag;
	}
	else // Hopefully this line of code never gets reached.
	{
		dp.x = posYou.x - posMe.x;
		dp.y = posYou.y - posMe.y;
		dp.z = posYou.z - posMe.z;
		if(0.0f < (dv.x*dv.x + dv.y*dv.y + dv.z*dv.z)) // Hopefully if they do not have the same velocity they will drift past setting right on top of eachother.
		{
			force.x = 0.0f;
			force.y = 0.0f;
			force.z = 0.0f;
			//printf("\n TSU error: Core Elements on top of each other in calculatePlasmaCoreForce \n");
		}
		else 	// If they have the same velocity we will need to kick them off of died center. 
			//This will work but I think I'll should kill the program to see if I need to patch the code.
		{
			force.x = 0.0001f;
			force.y = 0.0f;
			force.z = 0.0f;
			//printf("\n TSU error: Core Elements stuck on top of each other in calculatePlasmaCoreForce \n");
		}
	}
	
	return(force);
}

//Force between two core elements
__device__ float4 calculateCoreCoreForce(float4 posMe, float4 posYou, float4 velMe, float4 velYou, float4 forceMe, float4 forceYou, float corePushBackReduction)
{
	float4 dp, dv, force;
	float r, r2, r3, invr, inOut, force_mag;
	float gravity, firstTouch, maxPushBack;
	
	dp.x = posYou.x - posMe.x;
	dp.y = posYou.y - posMe.y;
	dp.z = posYou.z - posMe.z;
	r2 = dp.x*dp.x + dp.y*dp.y + dp.z*dp.z;
	r = sqrt(r2);
	r3 = r2*r;
	
	firstTouch = (forceYou.w + forceMe.w)*0.5f;  // Distance where the elements first touch
	
	if(firstTouch <= r)  // force.w holds the diameters so this is when the just touch or greater (only gravity).
	{
		// G = 1 and mass of each plasma elemnet = 1. So G*mass1*mass2 = coreMass1*coreMass2.
		invr = 1.0f/r3;
		force.x = posYou.w*posMe.w*(dp.x*invr);  // pos.w holds the mass
		force.y = posYou.w*posMe.w*(dp.y*invr);
		force.z = posYou.w*posMe.w*(dp.z*invr);
	}
	else if(0.0f < r)
	{	
		dv.x = velYou.x - velMe.x;
		dv.y = velYou.y - velMe.y;
		dv.z = velYou.z - velMe.z;
		inOut = dp.x*dv.x + dp.y*dv.y + dp.z*dv.z;
		
		gravity = posYou.w*posMe.w/(firstTouch*firstTouch);  //Holding gravity constant at where they touched. G =1 amd mass of plasma = 1.
		maxPushBack = (velYou.w + velMe.w)*0.5;  //vel.w hold push back
		
		// Making a linear increasing function from 0 to max.
		if(inOut <= 0) 	force_mag  = gravity - (-maxPushBack*r/firstTouch + maxPushBack);
		else 		force_mag  = gravity - corePushBackReduction*(-maxPushBack*r/firstTouch + maxPushBack);
		
		invr = 1.0f/r;
		force.x = (dp.x*invr)*force_mag;
		force.y = (dp.y*invr)*force_mag;
		force.z = (dp.z*invr)*force_mag;
	}
	else // Hopefully this line of code never gets reached.
	{
		dp.x = posYou.x - posMe.x;
		dp.y = posYou.y - posMe.y;
		dp.z = posYou.z - posMe.z;
		if(0.0f < (dv.x*dv.x + dv.y*dv.y + dv.z*dv.z)) // Hopefully if they do not have the same velocity they will drift past setting right on top of eachother.
		{
			force.x = 0.0f;
			force.y = 0.0f;
			force.z = 0.0f;
			//printf("\n TSU error: Core Core Elements on top of each other in calculatePlasmaCoreForce \n");
		}
		else 	// If they have the same velocity we will need to kick them off of died center. 
			//This will work but I think I'll should kill the program to see if I need to patch the code.
		{
			force.x = 0.0001f;
			force.y = 0.0f;
			force.z = 0.0f;
			//printf("\n TSU error: Core Core Elements stuck on top of each other in calculatePlasmaCoreForce \n");
		}
	}
	return(force);
}


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

