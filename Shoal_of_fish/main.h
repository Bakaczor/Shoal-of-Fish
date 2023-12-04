#pragma once

#include <cstdlib>
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <optional>

#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_gl_interop.h>

#include "global.h"
#include "glslUtility.hpp"

struct GL {
	// OpenGL
	GLuint triLocation = 0;
	GLuint shoalLocation = 1;
	const char* attributeLocations[2] = { "Point", "Id" };
	GLuint fishVAO = 0;
	GLuint fishVBO_tri = 0;
	GLuint fishVBO_sho = 0;
	GLuint fishEBO = 0;
	GLuint program = 0;

	// GLFW
	const int WIDTH = 1280;
	const int HEIGHT = 720;
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
bool init(int argc, char* argv[], Parameters& params, Tables& tabs, GL& props);
void initVAO(const int& N, GL& props);
void initShaders(GL& props);

/*************
* Main Loop *
*************/
void mainLoop(Parameters& params, Tables& tabs, GL& props);
void runStep(Parameters& params, Tables& tabs, GL& props);

/*************
* Callbacks *
*************/
void errorCallback(int error, const char* description);
void frameBufferSizeCallback(GLFWwindow* window, int width, int height);
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
void cursorPosCallback(GLFWwindow* window, double xpos, double ypos);
