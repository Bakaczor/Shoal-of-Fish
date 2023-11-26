#pragma once

#include <iostream>
#include <cstdlib>
#include <string>
#include <sstream>
#include <fstream>
#include <cmath>
#include <optional>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "kernel.h"

/********
* Main *
********/
int main(int argc, char* argv[]); 

/******************
* Initialization *
******************/
bool init(int argc, char* argv[], Global::Parameters& params, Global::Tables& tabs);
std::optional<std::string> getTitle();

/*************
* Main Loop *
*************/
void mainLoop(Global::Parameters& params, Global::Tables& tabs);
void run(Global::Parameters& params, Global::Tables& tabs);

