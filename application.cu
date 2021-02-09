#include <iostream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <cuda_gl_interop.h>
#include "graphicsIncludes.h"
#include "vendorIncludes.h"
#include "Renderer.h"
#include "Grid.h"
#include "Camera.h"
#include "Input.h"
#include "GUI.h"
#include "Window.h"
#include <stdio.h>
#include <stdlib.h>

//CUDA
const int N = 262144;

// camera
Camera camera(glm::vec3(70.0f, 30.0f, 260.0f));
// timing
float deltaTime = 0.0f;	
float lastFrame = 0.0f;
const int width = 1600;
const int height = 1000;
float lastX = width/2.0f;
float lastY = height/2.0f;
bool firstMouse= true;

void mouse_callback(GLFWwindow* window, double xpos, double ypos){
    if (firstMouse){
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top
    lastX = xpos;
    lastY = ypos;

    camera.ProcessMouseMovement(xoffset, yoffset);
}


int main(void){
	
	Window window(width, height, mouse_callback);
	glm::mat4 proj;
	proj = glm::perspective(glm::radians(45.0f), (float)width/(float)height, 0.1f, 400.0f);
	glm::mat4 view = camera.GetViewMatrix();
	glm::mat4 model = glm::mat4(1.0f);//glm::rotate(glm::mat4(1.0f), glm::radians(-100.0f), glm::vec3(1.0f,0.0f,0.0f));
	glm::mat4 mvp = proj * view * model;
	Renderer renderer;	
	Grid grid(5.0f, mvp);
	GUI gui(window.ptr);
	//set_initail_conditions();

	unsigned int index[N];
	for(unsigned int i = 0; i < N; i++){
		index[i] = i;
	}
	float particles_CPU[N*4];
	
	VertexArray va;
	VertexBuffer vb(N);
	VertexBufferLayout layout;		
	layout.Push<float>(4);
	va.AddBuffer(vb, layout);
	IndexBuffer ib(index, N);
	Shader shader("../res/shaders/particle.shader");
	

	FILE *posFile = fopen("../res/PosAndVel","rb");
	float fileTime;
	//FILE *posFile;
	
    while(!window.shouldClose()){
        float currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
		lastFrame = currentFrame;
		gui.NewFrame();
		Input::processInput(window.ptr, camera, deltaTime);

		glm::mat4 view = camera.GetViewMatrix();
		mvp = proj*view*model;

		fread(&fileTime, sizeof(float), 1,posFile);
		fread(particles_CPU,sizeof(float4),N,posFile);

		renderer.Clear();
		vb.Update(particles_CPU);
		ib.Bind();
		shader.Bind();
		shader.SetUniformMat4f("u_MVP", mvp);
		renderer.Draw(va,ib,shader,GL_POINTS);
		for (int i = 0; i < 3000; i++){
		
		}
		fread(particles_CPU,sizeof(float4),N,posFile);
		
		grid.Update(mvp);	
		renderer.Draw(grid);

		gui.CameraWindowUpdate(camera);
		gui.Render();
		window.Update();
    }
    return 0;
}
