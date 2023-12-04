#pragma once

#include <cstdio>
#include <cmath>
#include <ctime> 

#include <thrust/random.h>
#include <thrust/sort.h>
#include <thrust/gather.h>
#include <thrust/transform.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

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
    // NIEZMIENIALNE
    const float MIN_VEL = 0.0005f;
    const float MAX_VEL = 0.005f;

    struct Bounds {
        const int MIN_X = 0;
        const int MIN_Y = 0;
        const int MAX_X = 1;
        const int MAX_Y = 1;
        const float SCALE = 1.0f;
    } BOUNDS;

    // ZMIENIALNE PRZY INICJALIZACJI
    bool VISUALIZE = true;
    bool WRAP = true;
    int SHOAL_NUM = 5;
    int FISH_NUM = 500;
    int CELL_N = 50;

    // USTAWIANE AUTOMATYCZNIE
    float CELL_LEN = 0.02f;
    float CELL_LEN_INV = 50.0f;

    void setCELL_N(int cell_n) {
        CELL_N = cell_n;
        CELL_LEN = BOUNDS.SCALE / CELL_N;
        CELL_LEN_INV = 1 / CELL_LEN;
    }

    // ZMIENIALNE W TRAKCIE
    float DT = 0.1f;
    float R = 0.01f; // from 0 up to CELL_LEN
    float COS_PHI = 0.5f; // from -1 to 1
    float W_SEP = 0.05f; // from 0 to 0.1
    float W_ALI = 5.0f; // from 0 to 10
    float W_COH = 5.0f; // from 0 to 10

    struct Blackhole {
        const float VEL = 0.005f;
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
}