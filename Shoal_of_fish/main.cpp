#include "main.h"

/********
* Main *
********/

int main(int argc, char* argv[]) {
    Global::Parameters params;
    Global::Tables tabs;
    GL props;
    if (init(argc, argv, params, tabs, props)) {
        mainLoop(params, tabs, props);
        Global::endSimulation(tabs);
        glfwTerminate();
        return 0;
    } else { return 1; }
}

/******************
* Initialization *
******************/

std::optional<std::string> getTitle() {
    cudaDeviceProp deviceProp;
    int gpuDevice = 0;
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (gpuDevice > device_count) {
        std::cout
            << "Error: GPU device number is greater than the number of devices!"
            << " Perhaps a CUDA-capable GPU is not installed?"
            << std::endl;
        return {};
    }
    cudaGetDeviceProperties(&deviceProp, gpuDevice);
    int major = deviceProp.major;
    int minor = deviceProp.minor;

    std::ostringstream ss;
    ss << "Shoal of Fish" << " [" << major << "." << minor << " " << deviceProp.name << "]";
    return ss.str();
}

bool init(int argc, char* argv[], Global::Parameters& params, Global::Tables& tabs, GL& props) {
    
    // TODO : reading from console
    std::optional<std::string> title = getTitle();
    if (title.has_value()) {
        props.windowTitle = title.value();
    } else {
        return false;
    }

    glfwSetErrorCallback(errorCallback);
    if (!glfwInit()) {
        std::cout
            << "Error: Could not initialize GLFW!"
            << " Perhaps OpenGL 3.3 isn't available?"
            << std::endl;
        return false;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    props.window = glfwCreateWindow(params.WIDTH, params.HEIGHT, props.windowTitle.c_str(), nullptr, nullptr);
    if (!props.window) {
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(props.window);
    glfwSetFramebufferSizeCallback(props.window, frameSizeCallback);
    glfwSetKeyCallback(props.window, keyCallback);
    glfwSetCursorPosCallback(props.window, mousePositionCallback);
    glfwSetMouseButtonCallback(props.window, mouseButtonCallback);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) { return false; }

    initVAO(params.FISH_NUM, props);

    cudaGLSetGLDevice(0);
    cudaGLRegisterBufferObject(props.fishVBO_tri);
    //cudaGLRegisterBufferObject(props.fishVBO_pos);
    //cudaGLRegisterBufferObject(props.fishVBO_vel);

    Global::initSimulation(params, tabs);

    initShaders(props);
    glEnable(GL_DEPTH_TEST);
    return true;
}

void initVAO(const int& N, GL& props) {
    std::unique_ptr<GLfloat[]> bodies(new GLfloat[2 * 3 * N]);
    std::unique_ptr<GLuint[]> bindices(new GLuint[N]);

    for (int i = 0; i < N; i++) {
        const int j = 2 * 3 * i;
        bodies[j + 0] = 0.0f;
        bodies[j + 1] = 0.0f;
        bodies[j + 2] = 0.0f;
        bodies[j + 3] = 0.0f;
        bodies[j + 4] = 0.0f;
        bodies[j + 5] = 0.0f;
        bindices[i] = i;
    }

    glGenVertexArrays(1, &props.fishVAO); // Attach everything needed to draw to this id array
    glGenBuffers(1, &props.fishVBO_tri);
    //glGenBuffers(1, &props.fishVBO_pos);
    //glGenBuffers(1, &props.fishVBO_vel);
    glGenBuffers(1, &props.fishIBO);

    glBindVertexArray(props.fishVAO);

    /*
    // Bind the positions array to the boidVAO by way of the boidVBO_positions
    glBindBuffer(GL_ARRAY_BUFFER, props.fishVBO_pos); // bind the buffer
    glBufferData(GL_ARRAY_BUFFER, 2 * N * sizeof(GLfloat), bodies.get(), GL_DYNAMIC_DRAW); // transfer data
    glVertexAttribPointer(props.posLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(props.posLocation);

    // Bind the velocities array to the boidVAO by way of the boidVBO_velocities
    glBindBuffer(GL_ARRAY_BUFFER, props.fishVBO_vel);
    glBufferData(GL_ARRAY_BUFFER, 2 * N * sizeof(GLfloat), bodies.get(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(props.velLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(props.velLocation);
    */

    glBindBuffer(GL_ARRAY_BUFFER, props.fishVBO_tri);
    glBufferData(GL_ARRAY_BUFFER, 2 * 3 * N * sizeof(GLfloat), bodies.get(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(props.triLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(props.triLocation);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, props.fishIBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, N * sizeof(GLuint), bindices.get(), GL_STATIC_DRAW);

    glBindVertexArray(0);
}

void initShaders(GL& props) {
    props.program = glslUtility::createProgram("shaders/vert.glsl", "shaders/frag.glsl", props.attributeLocations, 1);
    glUseProgram(props.program);
}

/*************
* Main Loop *
*************/
void runSimulation(Global::Parameters& params, Global::Tables& tabs, GL& props) {
    /*
    float* d_vboPositions = nullptr;
    float* d_vboVelocities = nullptr;
    cudaGLMapBufferObject(reinterpret_cast<void**>(&d_vboPositions), props.fishVBO_pos);
    cudaGLMapBufferObject(reinterpret_cast<void**>(&d_vboVelocities), props.fishVBO_vel);
    */

    float* d_vboTriangles = nullptr;
    cudaGLMapBufferObject(reinterpret_cast<void**>(&d_vboTriangles), props.fishVBO_tri);

    Global::stepSimulation(params, tabs);
    if (params.VISUALIZE) {
        //Global::copyToVBO(params, tabs, d_vboPositions, d_vboVelocities);
        Global::copyTrianglesToVBO(params, tabs, d_vboTriangles);
    }
    cudaGLUnmapBufferObject(props.fishVBO_tri);
    /*
    cudaGLUnmapBufferObject(props.fishVBO_pos);
    cudaGLUnmapBufferObject(props.fishVBO_vel);
    */
}

void mainLoop(Global::Parameters& params, Global::Tables& tabs, GL& props) {
    double fps = 0;
    int frame = 0;
    double timebase = 0;

    while (!glfwWindowShouldClose(props.window)) {
        glfwPollEvents();

        frame++;
        double time = glfwGetTime();
        double elapsed = time - timebase;
        if (elapsed > 1.0) {
            fps = frame / elapsed;
            timebase = time;
            frame = 0;
        }

        runSimulation(params, tabs, props);

        std::ostringstream ss;
        ss << "[";
        ss.precision(1);
        ss << std::fixed << fps;
        ss << " fps] " << props.windowTitle;

        glfwSetWindowTitle(props.window, ss.str().c_str());
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        if (params.VISUALIZE) {
            glUseProgram(props.program);
            glBindVertexArray(props.fishVAO);
            //glPointSize(3.0f);
            //glDrawElements(GL_POINTS, params.FISH_NUM, GL_UNSIGNED_INT, 0);
            //glDrawElements(GL_TRIANGLES, 3 * params.FISH_NUM, GL_UNSIGNED_INT, 0);
            glDrawArrays(GL_TRIANGLES, 0, 3 * params.FISH_NUM);
            
            glUseProgram(0);
            glBindVertexArray(0);
            //glPointSize(1.0f);
            
            glfwSwapBuffers(props.window);
        }
    }
}

// CALLBACKS
bool leftMousePressed = false;
bool rightMousePressed = false;
double lastX;
double lastY;

void errorCallback(int error, const char* description) {
    fprintf(stderr, "error %d: %s\n", error, description);
}

void frameSizeCallback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    // zamknij program przyciskiem ESC
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GL_TRUE);
    }
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
    rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
}

void mousePositionCallback(GLFWwindow* window, double xpos, double ypos) {
    lastX = xpos;
    lastY = ypos;
}