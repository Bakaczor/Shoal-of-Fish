#include "main.h"

/********
* Main *
********/

int main(int argc, char* argv[]) {
    Parameters params;
    Tables tabs;
    GL props;
    if (!readArgs(argc, argv, params)) { return 1; }
    if (init(params, tabs, props)) {
        mainLoop(params, tabs, props);
        GPU::endSimulation(tabs);
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        glfwTerminate();
        return 0;
    } else { return 1; }
}

bool readArgs(int argc, char* argv[], Parameters& params) {
    bool flag = true;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--wrap") {
            params.WRAP = true;
        } else if (arg == "--fish_num" && i + 1 < argc) {
            try {
                params.FISH_NUM = std::stoi(argv[++i]);
            } catch (const std::invalid_argument& e) {
                flag = false;
                std::cerr << "The argument {" << argv[i] << "} is invalid." << std::endl << e.what();
            }
        } else if (arg == "--shoal_num" && i + 1 < argc) {
            try {
                params.SHOAL_NUM = std::stoi(argv[++i]);
            } catch (const std::invalid_argument& e) {
                flag = false;
                std::cerr << "The argument {" << argv[i] << "} is invalid." << std::endl << e.what();
            }
        } else if (arg == "--cell_n" && i + 1 < argc) {
            try {
                params.setCELL_N(std::stoi(argv[++i]));
            } catch (const std::invalid_argument& e) {
                flag = false;
                std::cerr << "The argument {" << argv[i] << "} is invalid." << std::endl << e.what();
            }
        }
    }
    return flag;
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

bool init(Parameters& params, Tables& tabs, GL& props) {
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

    props.window = glfwCreateWindow(props.WIDTH, props.HEIGHT, props.windowTitle.c_str(), nullptr, nullptr);
    if (!props.window) {
        glfwTerminate();
        return false;
    }

    glfwMakeContextCurrent(props.window);
    glfwSetFramebufferSizeCallback(props.window, frameBufferSizeCallback);
    glfwSetKeyCallback(props.window, keyCallback);
    glfwSetMouseButtonCallback(props.window, mouseButtonCallback);
    glfwSetCursorPosCallback(props.window, cursorPosCallback);
    glfwSetWindowUserPointer(props.window, &params.BLACKHOLE);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) { return false; }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui_ImplGlfw_InitForOpenGL(props.window, true);
    ImGui_ImplOpenGL3_Init();

    initVAO(params.FISH_NUM, props);

    cudaGLSetGLDevice(0);
    cudaGLRegisterBufferObject(props.fishVBO_tri);
    cudaGLRegisterBufferObject(props.fishVBO_sho);
    GPU::initSimulation(params, tabs);

    initShaders(props);
    glEnable(GL_DEPTH_TEST);
    return true;
}

void initVAO(const int& N, GL& props) {
    std::unique_ptr<GLfloat[]> bodies(new GLfloat[2 * 3 * N]);
    std::unique_ptr<GLuint[]> shoals(new GLuint[3 * N]);

    for (int i = 0; i < N; i++) {
        const int j = 2 * 3 * i;
        bodies[j + 0] = 0.0f;
        bodies[j + 1] = 0.0f;
        bodies[j + 2] = 0.0f;
        bodies[j + 3] = 0.0f;
        bodies[j + 4] = 0.0f;
        bodies[j + 5] = 0.0f;
        const int k = 3 * i;
        shoals[k + 0] = 0U;
        shoals[k + 1] = 0U;
        shoals[k + 2] = 0U;
    }

    glGenVertexArrays(1, &props.fishVAO);
    glGenBuffers(1, &props.fishVBO_tri);
    glGenBuffers(1, &props.fishVBO_sho);
    glGenBuffers(1, &props.fishEBO);
    glBindVertexArray(props.fishVAO);

    glBindBuffer(GL_ARRAY_BUFFER, props.fishVBO_tri);
    glBufferData(GL_ARRAY_BUFFER, 2 * 3 * N * sizeof(GLfloat), bodies.get(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(props.triLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(props.triLocation);

    glBindBuffer(GL_ARRAY_BUFFER, props.fishVBO_sho);
    glBufferData(GL_ARRAY_BUFFER, 3 * N * sizeof(GLuint), shoals.get(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(props.shoalLocation, 1, GL_UNSIGNED_INT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(props.shoalLocation);

    glBindVertexArray(0);
}

void initShaders(GL& props) {
    props.program = glslUtility::createProgram("shaders/vert.glsl", "shaders/frag.glsl", props.attributeLocations, 2);
    glUseProgram(props.program);
}

void renderUI(Parameters& params) {
    ImGui::Begin("Simulation Control Window");
    ImGui::SliderFloat("Time delta", &params.DT, 0.0f, 1.0f);
    ImGui::SliderFloat("View range", &params.R, 0.0f, params.CELL_LEN);
    ImGui::SliderFloat("Field of view (cosine)", &params.COS_PHI, -1.0f, 1.0f);
    ImGui::SliderFloat("Separation weight", &params.W_SEP, 0.0f, 0.01f);
    ImGui::SliderFloat("Alignment weight", &params.W_ALI, 0.0f, 10.0f);
    ImGui::SliderFloat("Coherence weight", &params.W_COH, 0.0f, 10.0f);
    ImGui::End();
}

/*************
* Main Loop *
*************/

void runStep(Parameters& params, Tables& tabs, GL& props) {
    float* d_vboTriangles = nullptr;
    uint* d_vboShoals = nullptr;
    cudaGLMapBufferObject(reinterpret_cast<void**>(&d_vboTriangles), props.fishVBO_tri);
    cudaGLMapBufferObject(reinterpret_cast<void**>(&d_vboShoals), props.fishVBO_sho);

    GPU::stepSimulation(params, tabs);
    if (VISUALIZE) {
        GPU::copyToVBO(params, tabs, d_vboTriangles, d_vboShoals);
    }
    cudaGLUnmapBufferObject(props.fishVBO_tri);
    cudaGLUnmapBufferObject(props.fishVBO_sho);
}

void mainLoop(Parameters& params, Tables& tabs, GL& props) {
    while (!glfwWindowShouldClose(props.window)) {
        glfwPollEvents();

        runStep(params, tabs, props);

        std::ostringstream ss;
        ss << "[";
        ss.precision(1);
        ss << std::fixed << ImGui::GetIO().Framerate;
        ss << " FPS] " << props.windowTitle;

        glfwSetWindowTitle(props.window, ss.str().c_str());
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        renderUI(params);

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        if (VISUALIZE) {
            glUseProgram(props.program);
            glBindVertexArray(props.fishVAO);
            glPointSize(2.0f);
            glDrawArrays(GL_TRIANGLES, 0, 3 * params.FISH_NUM);
            
            glUseProgram(0);
            glBindVertexArray(0);
            glPointSize(1.0f);
            glfwSwapBuffers(props.window);
        }
    }
}

/*************
* Callbacks *
*************/

void errorCallback(int error, const char* description) {
    fprintf(stderr, "error %d: %s\n", error, description);
}

void frameBufferSizeCallback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GL_TRUE);
    }
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    Parameters::Blackhole* bh = static_cast<Parameters::Blackhole*>(glfwGetWindowUserPointer(window));
    if (action == GLFW_PRESS && button == GLFW_MOUSE_BUTTON_RIGHT) {
        bh->PULL = true;
    } else if (action == GLFW_RELEASE) {
        bh->PULL = false;
    }
}

void cursorPosCallback(GLFWwindow* window, double xpos, double ypos) {
    Parameters::Blackhole* bh = static_cast<Parameters::Blackhole*>(glfwGetWindowUserPointer(window));
    int width, height;
    glfwGetWindowSize(window, &width, &height);
    bh->X = static_cast<float>(xpos / width);
    bh->Y = static_cast<float>((height - ypos) / height);
}