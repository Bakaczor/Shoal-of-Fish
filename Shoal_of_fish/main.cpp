#include "main.h"

/********
* Main *
********/

int main(int argc, char* argv[]) {
    Global::Parameters params;
    Global::Tables tabs;
    if (init(argc, argv, params, tabs)) {
        mainLoop(params, tabs);
        Global::endSimulation(tabs);
        return 0;
    } else { return 1; }
}

/******************
* Initialization *
******************/

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
    ss << "Shoal of Fish" << " [ " << major << "." << minor << " " << deviceProp.name << "]";
    return ss.str();
}

bool init(int argc, char* argv[], Global::Parameters& params, Global::Tables& tabs) {

    std::optional<std::string> title = getTitle();
    if (title.has_value()) {
        windowTitle = title.value();
    } else {
        return false;
    }

    // TODO : reading from console

    Global::initSimulation(
        params.FISH_NUM,
        params.CELL_N * params.CELL_N,
        params.SHOAL_NUM,
        params.BOUNDS.MAX_X - params.BOUNDS.MIN_X,
        tabs
    );

    return true;
}

/*************
* Main Loop *
*************/
void run(Global::Parameters& params, Global::Tables& tabs) {
    Global::stepSimulation(params, tabs);
}

void mainLoop(Global::Parameters& params, Global::Tables& tabs) {
    double fps = 0;
    int frame = 0;
    auto timebase = hrClock::now();

    while (true) {
        frame++;
        auto time = hrClock::now();
        std::chrono::duration<double> elapsed = time - timebase;
        if (elapsed.count() > 1.0) {
            fps = frame / elapsed.count();
            timebase = time;
            frame = 0;
            fprintf(stderr, "[%lf]\n", fps);
        }

        run(params, tabs);

        /*
        std::ostringstream ss;
        ss << "[";
        ss.precision(1);
        ss << std::fixed << fps;
        ss << " fps] " << windowTitle;
        */
    }
}