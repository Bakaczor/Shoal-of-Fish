#pragma once

#include <glm/glm.hpp>

typedef unsigned int uint;

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

struct Parameters {
    // CONSTANT
    const float MIN_VEL = 0.0005f;
    const float MAX_VEL = 0.005f;

    struct Bounds {
        const int MIN_X = 0;
        const int MIN_Y = 0;
        const int MAX_X = 1;
        const int MAX_Y = 1;
        const float SCALE = 1.0f;
    } BOUNDS;

    // SET DURING INITALIZATION
    bool WRAP = false;
    int FISH_NUM = 100;
    int SHOAL_NUM = 2;
    int CELL_N = 50;

    // SET AUTOMATICALLY
    float CELL_LEN = 0.02f;
    float CELL_LEN_INV = 50.0f;

    void setCELL_N(int cell_n) {
        CELL_N = cell_n;
        CELL_LEN = BOUNDS.SCALE / CELL_N;
        CELL_LEN_INV = 1 / CELL_LEN;
    }

    // MODIFIABLE DURING SIMULATION
    float DT = 0.2f;
    float R = 0.01f;
    float COS_PHI = 0.3f;
    float W_SEP = 0.005f;
    float W_ALI = 1.0f;
    float W_COH = 1.0f;

    struct Blackhole {
        const float VEL = 0.01f;
        float X = 0.5f;
        float Y = 0.5f;
        bool PULL = false;
    } BLACKHOLE;
};

namespace GPU {
    void initSimulation(const Parameters& params, Tables& tabs);
    void stepSimulation(const Parameters& params, Tables& tabs);
    void copyToVBO(const Parameters& params, Tables& tabs, float* d_vboTriangles, uint* d_vboShoals);
    void endSimulation(Tables& tabs);
}

namespace CPU {
    void initSimulation(const Parameters& params, Tables& tabs);
    void stepSimulation(const Parameters& params, Tables& tabs);
    void copyToVBO(const Parameters& params, Tables& tabs, float* vboTriangles, uint* vboShoals);
    void endSimulation(Tables& tabs);
}