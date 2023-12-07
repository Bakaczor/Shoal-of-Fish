#pragma once

#include <cstdlib>
#include <iostream>
#include <string>
#include <sstream>
#include <optional>
#include <chrono>
#include <vector>

#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_gl_interop.h>

#include "global.h"
#include "glslUtility/glslUtility.hpp"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

using hrc = std::chrono::high_resolution_clock;

// Turn off if you want to only measure calculations
constexpr bool VISUALIZE = true;

// Turn on if you want to run algorithm on CPU
constexpr bool HOST = false;

struct GL {
	// === OpenGL ===

	GLuint triLocation = 0;
	GLuint shoalLocation = 1;
	const char* attributeLocations[2] = { "Point", "Id" };
	GLuint fishVAO = 0;
	GLuint fishVBO_tri = 0;
	GLuint fishVBO_sho = 0;
	GLuint fishEBO = 0;
	GLuint program = 0;

	// === Window ===

	const int WIDTH = 1280;
	const int HEIGHT = 720;
	GLFWwindow* window = nullptr;
	std::string windowTitle;

	// === VBO ===

	std::unique_ptr<GLfloat[]> bodies;
	std::unique_ptr<GLuint[]> shoals;

	// === Measurements ===
	std::vector<long long> steps;
	std::vector<long long> copying;
};


/********
* Main *
********/

// Entry point for the application
int main(int argc, char* argv[]);

// Read and set arguments from the command line
bool readArgs(int argc, char* argv[], Parameters& params);

// Calculates the average from vector
double average(const std::vector<long long>& vec);

/******************
* Initialization *
******************/

// Retrives information about GPU and returns a title for the window
std::optional<std::string> getTitle();

// Initializes the window and the simulation
bool init(Parameters& params, Tables& tabs, GL& props);

// Initializes VAO and necessary VBOs
void initVAO(const int& N, GL& props);

// Compiles and loads shares, using glslUtility
void initShaders(GL& props);

/*************
* Main Loop *
*************/

// Renders the control window with the sliders
void renderUI(Parameters& params);

// Performs a singles step, including calculations and moving data
void runStep(Parameters& params, Tables& tabs, GL& props);

// Main loop of the application, procesing events, running simulation and swapping frames
void mainLoop(Parameters& params, Tables& tabs, GL& props);

/*************
* Callbacks *
*************/

// Handles GLFW errors
void errorCallback(int error, const char* description);

// Handles chnages in the size of window
void frameBufferSizeCallback(GLFWwindow* window, int width, int height);

// Handles the key input
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);

// Handles the mouse buttons
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);

// Handles changes in cursor position
void cursorPosCallback(GLFWwindow* window, double xpos, double ypos);
