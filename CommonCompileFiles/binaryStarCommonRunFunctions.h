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
	centerOfMass.w = totalMass;
	
	return(centerOfMass);
}

void drawPicture()
{	
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	//Drawing the cores spheres
	
	glPushMatrix();
		glTranslatef(PosCPU[NumberElementsStar1].x - CenterOfView.x, PosCPU[NumberElementsStar1].y - CenterOfView.y, PosCPU[NumberElementsStar1].z - CenterOfView.z);
		glColor3d(0.0,0.0,1.0);
		glutSolidSphere(ForceCPU[NumberElementsStar1].w,20,20);
	glPopMatrix();
	
	glPushMatrix();
		glTranslatef(PosCPU[0].x - CenterOfView.x, PosCPU[0].y - CenterOfView.y, PosCPU[0].z - CenterOfView.z);
		glColor3d(1.0,0.0,0.0);
		glutSolidSphere(ForceCPU[0].w,20,20);  // force.w holds the diameter of an element
	glPopMatrix();
	
	glBegin(GL_POINTS);
		//Drawing all the elements as points
		glPointSize(5.0);
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
		glPointSize(1.0);
		glColor3d(1.0,0.0,0.0);
		//glVertex3f(PosCPU[0].x - CenterOfView.x, PosCPU[0].y - CenterOfView.y, PosCPU[0].z - CenterOfView.z);
		glColor3d(0.0,0.0,1.0);
		//glVertex3f(PosCPU[NumberElementsStar1].x - CenterOfView.x, PosCPU[NumberElementsStar1].y - CenterOfView.y, PosCPU[NumberElementsStar1].z - CenterOfView.z);
	glEnd();
	
	glutSwapBuffers();
}

void recordPosAndVel(float time)
{
	fwrite(&time, sizeof(float), 1, PosAndVelFile);
	fwrite(PosCPU, sizeof(float4), NumberElements, PosAndVelFile);
	fwrite(VelCPU, sizeof(float4), NumberElements, PosAndVelFile);
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
		copyStarsDownFromGPU();
		CenterOfView = PosCPU[0];
		cout << "\nReset view to the center of core 1." << endl;
	}
	else if(command == 4)
	{
		copyStarsDownFromGPU();
		CenterOfView = PosCPU[NumberElementsStar1];
		cout << "\nReset view to the center of core 2." << endl;
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

void recordFinalPosVelForceStars(float time)
{	
	FILE *finalPosVelForceFile;
	finalPosVelForceFile = fopen("FinalPosVelForce", "wb");
	
	fwrite(&time, sizeof(float), 1, finalPosVelForceFile);
	fwrite(PosCPU, sizeof(float4), NumberElements, finalPosVelForceFile);
	fwrite(VelCPU, sizeof(float4), NumberElements, finalPosVelForceFile);
	fwrite(ForceCPU, sizeof(float4), NumberElements, finalPosVelForceFile);
	
	fclose(finalPosVelForceFile);
}
