#pragma once

#include <iostream>
#include <cstdlib>
#include <string>
#include <sstream>
#include <fstream>
#include <cmath>
#include <optional>

#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_gl_interop.h>

#include "kernel.h"
#include "glslUtility.hpp"

/**********
* OpenGL *
**********/

struct GL {

	//GLuint posLocation = 0;
	//GLuint velLocation = 1;
	//const char* attributeLocations[2] = { "Position", "Velocity" };

	GLuint triLocation = 0;
	const char* attributeLocations[1] = { "Triangle" };

	GLuint fishVAO = 0;
	GLuint fishVBO_tri = 0;
	//GLuint fishVBO_pos = 0;
	//GLuint fishVBO_vel = 0;
	GLuint fishIBO = 0;
	GLuint program = 0;

	// window
	GLFWwindow* window = nullptr;
	std::string windowTitle;
};


/********
* Main *
********/
int main(int argc, char* argv[]); 

/******************
* Initialization *
******************/
std::optional<std::string> getTitle();
bool init(int argc, char* argv[], Global::Parameters& params, Global::Tables& tabs, GL& props);
void initVAO(const int& N, GL& props);
void initShaders(GL& props);

/*************
* Main Loop *
*************/
void mainLoop(Global::Parameters& params, Global::Tables& tabs, GL& props);
void runSimulation(Global::Parameters& params, Global::Tables& tabs, GL& props);

void errorCallback(int error, const char* description);
void frameSizeCallback(GLFWwindow* window, int width, int height);
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
void mousePositionCallback(GLFWwindow* window, double xpos, double ypos);

