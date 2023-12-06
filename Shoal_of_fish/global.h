#pragma once

#include <glm/glm.hpp>

typedef unsigned int uint;

// This structure holds all data used during computations.
struct Tables {
    uint* d_fishId = nullptr;
    uint* d_cellId = nullptr;
    int* d_cellStart = nullptr;
    int* d_cellEnd = nullptr;

    glm::vec2* d_pos = nullptr;
    glm::vec2* d_pos_g = nullptr;
    glm::vec2* d_vel = nullptr;
    glm::vec2* d_vel_g = nullptr;
    uint* d_shoalId = nullptr;
    uint* d_shoalId_g = nullptr;

    glm::vec2* d_newVel = nullptr;
};

// This structure holds all parameters necessary for simulation.
struct Parameters {
    // === CONSTANT ===

    const float MIN_VEL = 0.0005f;
    const float MAX_VEL = 0.005f;

    struct Bounds {
        const int MIN_X = 0;
        const int MIN_Y = 0;
        const int MAX_X = 1;
        const int MAX_Y = 1;
        const float SCALE = 1.0f;
    } BOUNDS;

    // === SET DURING INITALIZATION ===

    bool WRAP = false;
    int FISH_NUM = 500;
    int SHOAL_NUM = 3;
    int CELL_N = 50;

    // === SET AUTOMATICALLY ===

    float CELL_LEN = 0.02f;
    float CELL_LEN_INV = 50.0f;

    void setCELL_N(int cell_n) {
        CELL_N = cell_n;
        CELL_LEN = BOUNDS.SCALE / CELL_N;
        CELL_LEN_INV = 1 / CELL_LEN;
    }

    // === MODIFIABLE DURING SIMULATION ===

    float DT = 0.2f;
    float R = 0.01f;
    float COS_PHI = 0.0f;
    float W_SEP = 0.005f;
    float W_ALI = 2.0f;
    float W_COH = 2.0f;

    struct Blackhole {
        const float VEL = 0.01f;
        float X = 0.5f;
        float Y = 0.5f;
        bool PULL = false;
    } BLACKHOLE;
};

namespace GPU {
    // Allocates memory and fills tables with randomized data
    void initSimulation(const Parameters& params, Tables& tabs);

    // Performs calculations for one frame of the simulation
    void stepSimulation(const Parameters& params, Tables& tabs);

    // Creates triangles from positions and velocities, copies them and shoals to VBO binded buffers
    void copyToVBO(const Parameters& params, Tables& tabs, float* vboTriangles, uint* vboShoals);

    // Deallocates memory
    void endSimulation(Tables& tabs);
}

namespace CPU {
    // Allocates memory and fills tables with randomized data
    void initSimulation(const Parameters& params, Tables& tabs);

    // Performs calculations for one frame of the simulation
    void stepSimulation(const Parameters& params, Tables& tabs);

    // Creates triangles from positions and velocities, copies them and shoals to VBO binded buffers
    void copyToVBO(const Parameters& params, Tables& tabs, float* vboTriangles, uint* vboShoals);

    // Deallocates memory
    void endSimulation(Tables& tabs);
}