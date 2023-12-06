#pragma once

#include <cstdlib>
#include <iostream>
#include <string>
#include <sstream>
#include <optional>

#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_gl_interop.h>

#include "global.h"
#include "glslUtility/glslUtility.hpp"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

constexpr bool VISUALIZE = true;
constexpr bool HOST = true;

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

	// VBO BINDED
	std::unique_ptr<GLfloat[]> bodies;
	std::unique_ptr<GLuint[]> shoals;
};


/********
* Main *
********/
int main(int argc, char* argv[]);
bool readArgs(int argc, char* argv[], Parameters& params);

/******************
* Initialization *
******************/
std::optional<std::string> getTitle();
bool init(Parameters& params, Tables& tabs, GL& props);
void initVAO(const int& N, GL& props);
void initShaders(GL& props);
void renderUI(Parameters& params);

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
