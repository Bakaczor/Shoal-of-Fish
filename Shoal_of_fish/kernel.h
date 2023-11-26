#pragma once

#include <stdio.h>
#include <cmath>
#include <chrono>
#include <algorithm>

#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/gather.h>
#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>

#include <glm/glm.hpp>

typedef unsigned int uint;
typedef std::chrono::high_resolution_clock hrClock;

namespace Global {
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
        const float DT = 0.1f;
        const float MIN_VEL = 0.5f;
        const float MAX_VEL = 2.0f;

        struct Bounds {
            const int MIN_X = 0;
            const int MIN_Y = 0;
            const int MAX_X = 100;
            const int MAX_Y = 100;
        } BOUNDS;

        // ZMIENIALNE PRZY INICJALIZACJI
        bool VISUALIZE = false;
        int SHOAL_NUM = 5;
        int FISH_NUM = 1000;
        int WIDTH = 1280;
        int HEIGHT = 720;
        int CELL_N = 50;

        // USTAWIANE AUTOMATYCZNIE
        float CELL_LEN = 2.0f;
        float CELL_LEN_INV = 0.5f;

        // ZMIENIALNE W TRAKCIE
        float R = 1.0f; // from 0 up to CELL_LEN
        float COS_PHI = 0.0f; // from -1 to 1
        float W_SEP = 0.5f; // from 0 to 1
        float W_ALI = 0.5f; // from 0 to 1
        float W_COH = 0.5f; // from 0 to 1
        //float BH_X; 
        //float BH_Y;
        //float W_BH;

        void setCELL_N(int cell_n) {
            CELL_N = cell_n;
            CELL_LEN = (BOUNDS.MAX_X - BOUNDS.MIN_X) / CELL_LEN;
            CELL_LEN_INV = 1 / CELL_LEN;
        }
    };

    void initSimulation(const int& N, const int& M, const int& shoals, const int& scale, Tables& tabs);
    void stepSimulation(const Parameters& params, Tables& tabs);
    void endSimulation(Tables& tabs);
}
