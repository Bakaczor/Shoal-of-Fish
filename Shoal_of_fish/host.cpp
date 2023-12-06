#include "global.h"

#include <utility>
#include <random>
#include <algorithm>

#include <thrust/sort.h>
#include <thrust/gather.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>

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

std::pair<int, int> getCellIndices(const float& x, const float& y, const float& L, const int& N) {
    int i = static_cast<int>(x * L);
    int j = static_cast<int>(y * L);
    // skrajny przypadek, w którym x lub y le¿¹ na górnej granicy
    if (i >= N) {
        i = N - 1;
    }
    if (j >= N) {
        j = N - 1;
    }
    return std::make_pair(i, j);
}

int getCellId(const int& i, const int& j, const int& N) {
    return i * N + j;
}

glm::vec2 reflectVelocity(const glm::vec2& vel, const float& val, const float& min, const float& max, const bool& side) {
    if (val < min || val > max) {
        if (side) {
            return glm::vec2(-vel.x, vel.y);
        }
        else {
            return glm::vec2(vel.x, -vel.y);
        }
    }
    return vel;
}

float wrapAround(float val, const float& min, const float& max) {
    if (val < min) {
        val = max - 1.0E-16f;
    }
    else if (val > max) {
        val = min + 1.0E-16f;
    }
    return val;
}

glm::vec2 wrapPosition(glm::vec2 position, const Parameters::Bounds& bounds) {
    position.x = wrapAround(position.x, bounds.MIN_X, bounds.MAX_X);
    position.y = wrapAround(position.y, bounds.MIN_Y, bounds.MAX_Y);
    return position;
}

void AssignIds(int N, int cellLineNumber, float inverseCellWidth, glm::vec2* positions, uint* fishIndices, uint* cellIndices) {
    for (int index = 0; index < N; index++) {
        const auto pair = getCellIndices(positions[index].x, positions[index].y, inverseCellWidth, cellLineNumber);
        int id = getCellId(pair.first, pair.second, cellLineNumber);
        cellIndices[index] = id;
        fishIndices[index] = index;
    }
}

void resetTable(int N, int* array, int value = -1) {
    for (int index = 0; index < N; index++) {
        array[index] = value;
    }
}

void AssignCellStartEnd(int N, uint* cellIndices, int* cellStarts, int* cellEnds) {
    for (int index = 0; index < N; index++) {

        const uint id = cellIndices[index];

        if (index == 0) {
            cellStarts[id] = index;
            if (cellIndices[index + 1] != id) {
                cellEnds[id] = index;
            }
            return;
        }

        if (index == N - 1) {
            cellEnds[id] = index;
            if (cellIndices[index - 1] != id) {
                cellStarts[id] = index;
            }
            return;
        }

        if (cellIndices[index - 1] != id) {
            cellStarts[id] = index;
        }

        if (cellIndices[index + 1] != id) {
            cellEnds[id] = index;
        }
    }
}

void updateVelocity(Parameters params, int* cellStarts, int* cellEnds,
    glm::vec2* positions, glm::vec2* velocities, uint* shoals, glm::vec2* newVelocities) {
    for (int index = 0; index < params.FISH_NUM; index++) {

        const auto pair = getCellIndices(positions[index].x, positions[index].y, params.CELL_LEN_INV, params.CELL_N);
        const int iStart = std::max(pair.first - 1, 0);
        const int iEnd = std::min(pair.first + 1, params.CELL_N - 1);
        const int jStart = std::max(pair.second - 1, 0);
        const int jEnd = std::min(pair.second + 1, params.CELL_N - 1);

        const float maxVelSq = params.MAX_VEL * params.MAX_VEL;
        const glm::vec2 current_pos = positions[index];
        const glm::vec2 current_vel = velocities[index];
        const uint current_shoal = shoals[index];
        glm::vec2 separation(0, 0), alignment(0, 0), coherence(0, 0), center(0, 0);
        int count = 0;

        for (int i = iStart; i < iEnd; i++) {
            for (int j = jStart; j < jEnd; j++) {
                const auto id = getCellId(i, j, params.CELL_N);

                const int start = cellStarts[id];
                const int end = cellEnds[id];

                if (start == -1) continue;

                for (int k = start; k <= end; k++) {
                    if (k == index) continue;
                    const glm::vec2 neighbour_pos = positions[k];
                    const glm::vec2 neighbour_vel = velocities[k];
                    const uint neighbour_shoal = shoals[k];

                    glm::vec2 vector = neighbour_pos - current_pos;
                    float distance = glm::distance(neighbour_pos, current_pos);
                    float distanceSq = distance * distance;
                    if (distanceSq > maxVelSq) {
                        vector = glm::normalize(vector) * params.MAX_VEL;
                        distanceSq = maxVelSq;
                    }

                    const float cosine = glm::dot(glm::normalize(current_vel), glm::normalize(vector));
                    if (distance < params.R && cosine > params.COS_PHI) {
                        if (distance == 0) {
                            distance = 1.0E-16f;
                        }
                        if (neighbour_shoal == current_shoal) {
                            separation -= 0.1f * vector / distanceSq;
                            alignment += neighbour_vel;
                            center += neighbour_pos;
                        }
                        else {
                            separation -= vector / distanceSq;
                        }
                        ++count;
                    }
                }
            }
        }

        glm::vec2 newVelocity = current_vel;
        if (count != 0) {
            separation /= count;
            newVelocity += separation * params.W_SEP;

            alignment /= count;
            newVelocity += alignment * params.W_ALI;

            center /= count;
            coherence = (center - current_pos);
            newVelocity += coherence * params.W_COH;

            newVelocity /= 4;
        }

        if (params.BLACKHOLE.PULL) {
            glm::vec2 bh = glm::vec2(params.BLACKHOLE.X, params.BLACKHOLE.Y) - current_pos;
            const float R2 = glm::dot(bh, bh);
            bh = glm::normalize(bh) * params.BLACKHOLE.VEL / R2;
            newVelocity += bh;
            newVelocity /= 2;
        }

        const float norm = glm::dot(newVelocity, newVelocity);
        if (norm > maxVelSq) {
            newVelocity = glm::normalize(newVelocity) * params.MAX_VEL;
        }
        else if (norm < params.MIN_VEL * params.MIN_VEL) {
            newVelocity = glm::normalize(newVelocity) * params.MIN_VEL;
        }

        if (!params.WRAP) {
            const glm::vec2 prediction = current_pos + newVelocity * params.DT;
            newVelocity = reflectVelocity(newVelocity, prediction.x, params.BOUNDS.MIN_X, params.BOUNDS.MAX_X, true);
            newVelocity = reflectVelocity(newVelocity, prediction.y, params.BOUNDS.MIN_Y, params.BOUNDS.MAX_Y, false);
        }
        newVelocities[index] = newVelocity;
    }
}

void CPU::stepSimulation(const Parameters& params, Tables& tabs) {
    const int& N = params.FISH_NUM;
    const int& M = params.CELL_N * params.CELL_N;

    AssignIds(N, params.CELL_N, params.CELL_LEN_INV, tabs.d_pos, tabs.d_fishId, tabs.d_cellId);

    thrust::sort_by_key(thrust::host, tabs.d_cellId, tabs.d_cellId + N, tabs.d_fishId);

    resetTable(M, tabs.d_cellStart);
    resetTable(M, tabs.d_cellEnd);

    AssignCellStartEnd(N, tabs.d_cellId, tabs.d_cellStart, tabs.d_cellEnd);

    thrust::gather(thrust::host, tabs.d_fishId, tabs.d_fishId + N, tabs.d_pos, tabs.d_pos_g);
    thrust::gather(thrust::host, tabs.d_fishId, tabs.d_fishId + N, tabs.d_vel, tabs.d_vel_g);
    thrust::gather(thrust::host, tabs.d_fishId, tabs.d_fishId + N, tabs.d_shoalId, tabs.d_shoalId_g);

    updateVelocity(params, tabs.d_cellStart, tabs.d_cellEnd, tabs.d_pos_g, tabs.d_vel_g, tabs.d_shoalId_g, tabs.d_newVel);

    const float& dt = params.DT;
    const bool& wrap = params.WRAP;
    const Parameters::Bounds& bounds = params.BOUNDS;

    std::transform(tabs.d_pos_g, tabs.d_pos_g + N, tabs.d_newVel, tabs.d_pos,
        [wrap, dt, bounds] (const glm::vec2 & pos, const glm::vec2 & vel) {
            if (wrap) {
                return wrapPosition(pos + vel * dt, bounds);
            }
            return pos + vel * dt;
        });

    std::copy(tabs.d_newVel, tabs.d_newVel + N, tabs.d_vel);
    std::copy(tabs.d_shoalId_g, tabs.d_shoalId_g + N, tabs.d_shoalId);
}

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

void CPU::copyToVBO(const Parameters& params, Tables& tabs, float* vboTriangles, uint* vboShoals) {
    copyTriangleToVBO(params.FISH_NUM, tabs.d_pos, tabs.d_vel, vboTriangles);
    copyShoalToVBO(params.FISH_NUM, tabs.d_shoalId, vboShoals);
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