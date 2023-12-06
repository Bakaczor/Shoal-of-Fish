#include "global.h"

#include <random>

/******************
* initSimulation *
******************/

glm::vec2 generateRandomVec2(const int& rand, const int& index, const float& a, const float& b) {
    std::default_random_engine rng(index * rand);
    std::uniform_real_distribution<float> U(a, b);
    return glm::vec2(U(rng), U(rng));
}

uint generateRandomUint(const int& rand, const int& index, const float& scale) {
    std::default_random_engine rng(index * rand);
    std::uniform_real_distribution<float> U(0.0f, scale);
    return static_cast<uint>(U(rng));
}

void generateRandomFloatArray(int rand, int N, float a, float b, glm::vec2* array) {
    for (int index = 0; index < N; index++) {
        glm::vec2 random = generateRandomVec2(rand, index, a, b);
        array[index].x = random.x;
        array[index].y = random.y;
    }
}

void generateRandomUintArray(int rand, int N, float scale, uint* array) {
    for (int index = 0; index < N; index++) {
        array[index] = generateRandomUint(rand, index, scale);
    }
}

void CPU::initSimulation(const Parameters& params, Tables& tabs) {
    const int N = params.FISH_NUM;
    const int M = params.CELL_N * params.CELL_N;

    tabs.d_fishId = new uint[N];
    if (!tabs.d_fishId) {
        fprintf(stderr, "Error: new d_fishId failed!: %d.\n", __LINE__);
    }

    tabs.d_cellId = new uint[N];
    if (!tabs.d_cellId) {
        fprintf(stderr, "Error: new d_cellId failed!: %d.\n", __LINE__);
    }

    tabs.d_cellStart = new int[M];
    if (!tabs.d_cellStart) {
        fprintf(stderr, "Error: new d_cellStart failed!: %d.\n", __LINE__);
    }

    tabs.d_cellEnd = new int[M];
    if (!tabs.d_cellEnd) {
        fprintf(stderr, "Error: new d_cellEnd failed!: %d.\n", __LINE__);
    }

    tabs.d_pos = new glm::vec2[N];
    if (!tabs.d_pos) {
        fprintf(stderr, "Error: new d_pos failed!: %d.\n", __LINE__);
    }

    tabs.d_pos_g = new glm::vec2[N];
    if (!tabs.d_pos_g) {
        fprintf(stderr, "Error: new d_pos failed!: %d.\n", __LINE__);
    }
    tabs.d_vel = new glm::vec2[N];
    if (!tabs.d_vel) {
        fprintf(stderr, "Error: new d_vel failed!: %d.\n", __LINE__);
    }
    tabs.d_vel_g = new glm::vec2[N];
    if (!tabs.d_vel_g) {
        fprintf(stderr, "Error: new d_vel_g failed!: %d.\n", __LINE__);
    }

    tabs.d_shoalId = new uint[N];
    if (!tabs.d_shoalId) {
        fprintf(stderr, "Error: new d_shoalId failed!: %d.\n", __LINE__);
    }

    tabs.d_shoalId_g = new uint[N];
    if (!tabs.d_shoalId_g) {
        fprintf(stderr, "Error: new d_shoalId_g failed!: %d.\n", __LINE__);
    }

    tabs.d_newVel = new glm::vec2[N];
    if (!tabs.d_newVel) {
        fprintf(stderr, "Error: new d_newVel failed!: %d.\n", __LINE__);
    }

    const float max_6 = params.BOUNDS.MAX_X / 6.0f;
    srand(time(0));

    generateRandomFloatArray(rand(), N, max_6, 5 * max_6, tabs.d_pos);
    generateRandomFloatArray(rand(), N, -params.MAX_VEL, params.MAX_VEL, tabs.d_vel);

    generateRandomUintArray(rand(), N, params.SHOAL_NUM, tabs.d_shoalId);
}

/******************
* stepSimulation *
******************/

// TODO


/**************
 * copyToVBO *
 **************/

void copyTriangleToVBO(int N, glm::vec2* pos, glm::vec2* vel, float* vbo) {
    for (int index = 0; index < N; index++) {
        glm::vec2 posScaled = 2.0f * pos[index] - 1.0f;
        glm::vec2 velNormed = 0.01f * glm::normalize(vel[index]);
        glm::vec2 p1 = posScaled + 0.5f * glm::vec2(-velNormed.y, velNormed.x);
        glm::vec2 p2 = posScaled + 2.0f * velNormed;
        glm::vec2 p3 = posScaled + 0.5f * glm::vec2(velNormed.y, -velNormed.x);
        vbo[6 * index + 0] = p1.x;
        vbo[6 * index + 1] = p1.y;
        vbo[6 * index + 2] = p2.x;
        vbo[6 * index + 3] = p2.y;
        vbo[6 * index + 4] = p3.x;
        vbo[6 * index + 5] = p3.y;
    }
}

void copyShoalToVBO(int N, uint* shoal, uint* vbo) {
    for (int index = 0; index < N; index++) {
        vbo[3 * index + 0] = shoal[index];
        vbo[3 * index + 1] = shoal[index];
        vbo[3 * index + 2] = shoal[index];
    }
}

void CPU::copyToVBO(const Parameters& params, Tables& tabs, float* d_vboTriangles, uint* d_vboShoals) {
    copyTriangleToVBO(params.FISH_NUM, tabs.d_pos, tabs.d_vel, d_vboTriangles);
    copyShoalToVBO(params.FISH_NUM, tabs.d_shoalId, d_vboShoals);
}

 /******************
 * endSimulation *
 ******************/

void CPU::endSimulation(Tables& tabs) {
    delete[] tabs.d_fishId;
    delete[] tabs.d_cellId;
    delete[] tabs.d_cellStart;
    delete[] tabs.d_cellEnd;
    delete[] tabs.d_pos;
    delete[] tabs.d_pos_g;
    delete[] tabs.d_vel;
    delete[] tabs.d_vel_g;
    delete[] tabs.d_shoalId;
    delete[] tabs.d_shoalId_g;
    delete[] tabs.d_newVel;
}