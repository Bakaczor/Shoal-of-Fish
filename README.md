# Shoal of fish
This CUDA application simulates the movement and reactions between fish in 2D space, based on Craig W. Reynolds "Flocks, Herds, and Schools: A Distributed Behavioral Model" work.

In addition to that, with correct parameters the fish from the same shoal (with the same color) may join and swim together. What's more, with the right mouse button you can gather the fish in one place, using blackhole-like pulling force.

The project was built using OpenGL 3.3 and C++20.

## Command line arguments
```
--wrap
The fish will wrap their position around, when crossing the border. If not set, they will bounce of it.

--fish_num <number>
The number of fish to simulate.
By default 500.

--shoal_num <number>
The number of the fish shoals.
By default 3.

--cell_n <number>
The number of the cells of the uniform grid in one row (and number of rows at the same time).
By default 50.
```

## Constants
At the top of the `main.h` file you can find:
```cpp
// Turn off if you want to only measure calculations
constexpr bool VISUALIZE = true;

// Turn on if you want to run algorithm on CPU
constexpr bool HOST = false;
```

## Dependencies
The applications relies on several libraries and frameworks. Some are included in the project and some are not.

### Included
- Dear ImGui - bloat-free graphical user interface for C++ with minimal dependencies. It was used to create the control window with sliders for simulation parameters.
  
    Availible at: https://github.com/ocornut/imgui

- GLSL Utility - a utility namespace for loading GLSL shaders. It was only used to compile and load the shaders.

    Availible at: https://github.com/CIS565-Fall-2022/Project1-CUDA-Flocking

### Not included
- glm - header only C++ mathematics library for graphics software. Contains `glm::vec2` and many useful methods like `glm::distance`, `glm::dot`, `glm::normalize` etc.

    Availible at: https://glm.g-truc.net/0.9.9/

- GLFW - open source, multi-platform library for OpenGL, OpenGL ES and Vulkan development on the desktop. It was used to create create the application window, handle callbacks etc.

    Availible at: https://www.glfw.org/

- GLEW - cross-platform open-source C/C++ extension loading library. It is necessar in order to use OpenGL and its extensions. Glad would be the alternative.

    Availible at: https://glew.sourceforge.net/

## Building
### Compiler flags
The CUDA compiler (nvcc) requiers the `--extended-lambda ` flag to build. In Visual Studio you can set it at:
```
Debug > Properties > CUDA C/C++ > Command Line
```
### Including
To include the necessary, not included libraries, in Visual Studio go to:
```
Debug > Properties > C/C++ > General > Additional Include Directories
```
and add there a path with `include` directory, containing `glm`, `GLFW` and `GL` folders.

### Linking
To link the necessary, not included libraries, in Visual Studio go to:
```
Debug > Properties > Linker > General > Additional Library Directories
```
and add there a path to `lib` directory.

Then go to `Linker > Input > Additional Dependencies` and add there `opengl32.lib`, `glew32s.lib` and `glfw3.lib`.

### Dynamic libraries
Go to directory with the executable (in Visual Studio `x64 > Debug`) and copy there `glew32.dll` and `shaders` directory (if it haven't been copied by itself).


## Contact
In case of any problems, contact me at my university mail: bartosz.kaczorowski2.stud@pw.edu.pl.
