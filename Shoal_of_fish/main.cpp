#include "main.h"

/********
* Main *
********/

int main(int argc, char* argv[]) {
    Global::Parameters params;
    Global::Tables tabs;
    GL glProperties;
    if (init(argc, argv, params, tabs, glProperties)) {
        mainLoop(params, tabs, glProperties);
        Global::endSimulation(tabs);
        return 0;
    } else { return 1; }
}

/******************
* Initialization *
******************/

GLFWwindow* window = nullptr;
std::string windowTitle;

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

bool init(int argc, char* argv[], Global::Parameters& params, Global::Tables& tabs, GL& glProperties) {
    
    // TODO : reading from console
    std::optional<std::string> title = getTitle();
    if (title.has_value()) {
        windowTitle = title.value();
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

    window = glfwCreateWindow(params.WIDTH, params.HEIGHT, windowTitle.c_str(), nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetCursorPosCallback(window, mousePositionCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) { return false; }

    initVAO(params.FISH_NUM, glProperties);

    cudaGLSetGLDevice(0);
    cudaGLRegisterBufferObject(glProperties.fishVBO_pos);
    cudaGLRegisterBufferObject(glProperties.fishVBO_vel);

    Global::initSimulation(params, tabs);

    initShaders(glProperties);
    glEnable(GL_DEPTH_TEST);
    return true;
}

void initVAO(const int& N, GL& glProperties) {
    std::unique_ptr<GLfloat[]> bodies(new GLfloat[2 * N]);
    std::unique_ptr<GLuint[]> bindices(new GLuint[N]);

    for (int i = 0; i < N; i++) {
        bodies[2 * i + 0] = 0.0f;
        bodies[2 * i + 1] = 0.0f;
        bindices[i] = i;
    }

    glGenVertexArrays(1, &glProperties.fishVAO); // Attach everything needed to draw a particle to this
    glGenBuffers(1, &glProperties.fishVBO_pos);
    glGenBuffers(1, &glProperties.fishVBO_vel);
    glGenBuffers(1, &glProperties.fishIBO);

    glBindVertexArray(glProperties.fishVAO);

    // Bind the positions array to the boidVAO by way of the boidVBO_positions
    glBindBuffer(GL_ARRAY_BUFFER, glProperties.fishVBO_pos); // bind the buffer
    glBufferData(GL_ARRAY_BUFFER, 2 * N * sizeof(GLfloat), bodies.get(), GL_DYNAMIC_DRAW); // transfer data

    glEnableVertexAttribArray(glProperties.posLocation);
    glVertexAttribPointer(glProperties.posLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);

    // Bind the velocities array to the boidVAO by way of the boidVBO_velocities
    glBindBuffer(GL_ARRAY_BUFFER, glProperties.fishVBO_vel);
    glBufferData(GL_ARRAY_BUFFER, 2 * N * sizeof(GLfloat), bodies.get(), GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(glProperties.velLocation);
    glVertexAttribPointer(glProperties.velLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, glProperties.fishIBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, N * sizeof(GLuint), bindices.get(), GL_STATIC_DRAW);

    glBindVertexArray(0);
}

void initShaders(GL& glProperties) {
    glProperties.program = glslUtility::createProgram("shaders/vert.glsl", "shaders/frag.glsl", glProperties.attributeLocations, 2);
    glUseProgram(glProperties.program);
}

/*************
* Main Loop *
*************/
void run(Global::Parameters& params, Global::Tables& tabs, GL& glProperties) {
    float* d_vboPositions = nullptr;
    float* d_vboVelocities = nullptr;
    cudaGLMapBufferObject(reinterpret_cast<void**>(&d_vboPositions), glProperties.fishVBO_pos);
    cudaGLMapBufferObject(reinterpret_cast<void**>(&d_vboVelocities), glProperties.fishVBO_vel);

    Global::stepSimulation(params, tabs);
    if (params.VISUALIZE) {
        Global::copyToVBO(params, tabs, d_vboPositions, d_vboVelocities);
    }

    cudaGLUnmapBufferObject(glProperties.fishVBO_pos);
    cudaGLUnmapBufferObject(glProperties.fishVBO_vel);
}

void mainLoop(Global::Parameters& params, Global::Tables& tabs, GL& glProperties) {
    double fps = 0;
    int frame = 0;
    double timebase = 0;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        frame++;
        double time = glfwGetTime();
        double elapsed = time - timebase;
        if (elapsed > 1.0) {
            fps = frame / elapsed;
            timebase = time;
            frame = 0;
        }

        run(params, tabs, glProperties);

        std::ostringstream ss;
        ss << "[";
        ss.precision(1);
        ss << std::fixed << fps;
        ss << " fps] " << windowTitle;

        glfwSetWindowTitle(window, ss.str().c_str());
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        if (params.VISUALIZE) {
            glUseProgram(glProperties.program);
            glBindVertexArray(glProperties.fishVAO);
            glPointSize(3.0f);
            glDrawElements(GL_POINTS, params.FISH_NUM, GL_UNSIGNED_INT, 0);
            
            glUseProgram(0);
            glBindVertexArray(0);
            glPointSize(1.0f);
            
            glfwSwapBuffers(window);
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