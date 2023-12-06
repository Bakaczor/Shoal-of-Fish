#define GLM_FORCE_CUDA

#include <thrust/random.h>
#include <thrust/sort.h>
#include <thrust/gather.h>
#include <thrust/transform.h>
#include <thrust/device_vector.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "global.h"

/******************
* Error handling *
******************/

// Handles CUDA errors
void checkCUDAError(const char *msg, int line = -1) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        if (line >= 0) {
            fprintf(stderr, "Line %d: ", line);
        }
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

/*****************
* Configuration *
*****************/

constexpr int blockSize = 64;
const dim3 threadsPerBlock(blockSize);

/******************
* initSimulation *
******************/

// Generates random vec2 with values between a and b
__device__ glm::vec2 d_generateRandomVec2(const int& rand, const int& index, const float& a, const float& b) {
    thrust::default_random_engine rng(index * rand);
    thrust::uniform_real_distribution<float> U(a, b);
    return glm::vec2(U(rng), U(rng));
}

// Generates random unsigned int with values between 0 and scale
__device__ uint d_generateRandomUint(const int& rand, const int& index, const float& scale) {
    thrust::default_random_engine rng(index * rand);
    thrust::uniform_real_distribution<float> U(0.0f, scale);
    return static_cast<uint>(U(rng));
}

// Fills given array with randomized vec2s
__global__ void k_generateRandomFloatArray(int rand, int N, float a, float b, glm::vec2* array) {
    const int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= N) return;
    glm::vec2 random = d_generateRandomVec2(rand, index, a, b);
    array[index].x = random.x;
    array[index].y = random.y;
}

// Fills given array with randomized unsigned ints
__global__ void k_generateRandomUintArray(int rand, int N, float scale, uint* array) {
    const int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= N) return;
    array[index] = d_generateRandomUint(rand, index, scale);
}

void GPU::initSimulation(const Parameters& params, Tables& tabs) {
    const int N = params.FISH_NUM;
    const int M = params.CELL_N * params.CELL_N;
    const dim3 blocksPerGrid((N + 1) / blockSize);

    // --- Memory allocation ---

    cudaMalloc(reinterpret_cast<void**>(&tabs.d_fishId), N * sizeof(uint));
    checkCUDAError("cudaMalloc d_fishId failed!", __LINE__);

    cudaMalloc(reinterpret_cast<void**>(&tabs.d_cellId), N * sizeof(uint));
    checkCUDAError("cudaMalloc d_cellId failed!", __LINE__);

    cudaMalloc(reinterpret_cast<void**>(&tabs.d_cellStart), M * sizeof(int));
    checkCUDAError("cudaMalloc d_cellStart failed!", __LINE__);

    cudaMalloc(reinterpret_cast<void**>(&tabs.d_cellEnd), M * sizeof(int));
    checkCUDAError("cudaMalloc d_cellEnd failed!", __LINE__);

    cudaMalloc(reinterpret_cast<void**>(&tabs.d_pos), N * sizeof(glm::vec2));
    checkCUDAError("cudaMalloc d_pos failed!", __LINE__);

    cudaMalloc(reinterpret_cast<void**>(&tabs.d_pos_g), N * sizeof(glm::vec2));
    checkCUDAError("cudaMalloc d_pos_g failed!", __LINE__);

    cudaMalloc(reinterpret_cast<void**>(&tabs.d_vel), N * sizeof(glm::vec2));
    checkCUDAError("cudaMalloc d_vel failed!", __LINE__);

    cudaMalloc(reinterpret_cast<void**>(&tabs.d_vel_g), N * sizeof(glm::vec2));
    checkCUDAError("cudaMalloc d_vel_g failed!", __LINE__);

    cudaMalloc(reinterpret_cast<void**>(&tabs.d_shoalId), N * sizeof(uint));
    checkCUDAError("cudaMalloc d_shoalId failed!", __LINE__);

    cudaMalloc(reinterpret_cast<void**>(&tabs.d_shoalId_g), N * sizeof(uint));
    checkCUDAError("cudaMalloc d_shoalId_g failed!", __LINE__);

    cudaMalloc(reinterpret_cast<void**>(&tabs.d_newVel), N * sizeof(glm::vec2));
    checkCUDAError("cudaMalloc d_newVel failed!", __LINE__);

    srand(time(0));

    // --- Filling the arrays ---

    k_generateRandomFloatArray <<< blocksPerGrid, threadsPerBlock >>> (rand(), N, 1.0E-16f, params.BOUNDS.MAX_X - 1.0E-16f, tabs.d_pos);
    checkCUDAError("k_generateRandomFloatArray failed!", __LINE__);
    cudaDeviceSynchronize();
    checkCUDAError("synchronize failed!", __LINE__);

    k_generateRandomFloatArray <<< blocksPerGrid, threadsPerBlock >>> (rand(), N, -params.MAX_VEL, params.MAX_VEL, tabs.d_vel);
    checkCUDAError("k_generateRandomFloatArray failed!", __LINE__);
    cudaDeviceSynchronize();
    checkCUDAError("synchronize failed!", __LINE__);

    k_generateRandomUintArray <<<blocksPerGrid, threadsPerBlock >>> (rand(), N, params.SHOAL_NUM, tabs.d_shoalId);
    checkCUDAError("k_generateRandomUintArray failed!", __LINE__);
    cudaDeviceSynchronize();
    checkCUDAError("synchronize failed!", __LINE__);
}

/******************
* stepSimulation *
******************/

// Gets the (i, j) indices of the cell the fish with position (x, y) is currently in
__device__ thrust::pair<int, int> d_getCellIndices(const float& x, const float& y, const float& L, const int& N) {
    int i = static_cast<int>(x * L);
    int j = static_cast<int>(y * L);

    // This is a edge case, where x or y are at the top border of the space
    if (i >= N) {
        i = N - 1;
    }
    if (j >= N) {
        j = N - 1;
    }
    return thrust::make_pair(i, j);
}

// Returns the line index based (i, j) indices
__device__ int d_getCellId(const int& i, const int& j, const int& N) {
    return i * N + j;
}

// Reflects the velocity, so that the fish will bounce of the border
__device__ glm::vec2 d_reflectVelocity(const glm::vec2& vel, const float& val, const float& min, const float& max, const bool& side) {
    if (val < min || val > max) {
        if (side) {
            return glm::vec2(-vel.x, vel.y);
        } else {
            return glm::vec2(vel.x, -vel.y);
        }
    }
    return vel;
}

// Wraps the point on the other side of the space
__device__ float d_wrapAround(float val, const float& min, const float& max) {
    if (val < min) {
        val = max - 1.0E-16f;
    } else if (val > max) {
        val = min + 1.0E-16f;
    }
    return val;
}

// Returns wrapped (x, y) position of the fish leaving the space
__device__ glm::vec2 d_wrapPosition(glm::vec2 position, const Parameters::Bounds& bounds) {
    position.x = d_wrapAround(position.x, bounds.MIN_X, bounds.MAX_X);
    position.y = d_wrapAround(position.y, bounds.MIN_Y, bounds.MAX_Y);
    return position;
}

// Assigns ids to fish based on thread indices and corresponding ids to cell they are contained within
__global__ void k_AssignIds(int N, int cellLineNumber, float inverseCellWidth, glm::vec2* positions, uint* fishIndices, uint* cellIndices) {
    const int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= N) return;
    const auto pair = d_getCellIndices(positions[index].x, positions[index].y, inverseCellWidth, cellLineNumber);
    int id = d_getCellId(pair.first, pair.second, cellLineNumber);
    cellIndices[index] = id;
    fishIndices[index] = index;  
}

// Sets the table to a given value
__global__ void k_resetTable(int N, int* array, int value = -1) {
    const int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= N) return;
    array[index] = value;
}

// 
__global__ void k_AssignCellStartEnd(int N, uint* cellIndices, int* cellStarts, int* cellEnds) {
    const int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= N) return;

    const uint id = cellIndices[index];

    // Beginning of the table
    if (index == 0) {
        cellStarts[id] = index;
        if (cellIndices[index + 1] != id) {
            cellEnds[id] = index;
        }
        return;
    }

    // Ending of the table
    if (index == N - 1) {
        cellEnds[id] = index;
        if (cellIndices[index - 1] != id) {
            cellStarts[id] = index;
        }
        return;
    }

    // Standard case

    if (cellIndices[index - 1] != id) {
        cellStarts[id] = index;
    }

    if (cellIndices[index + 1] != id) {
        cellEnds[id] = index;
    }
}

// Core function, updates fish's velocity based on its surrounding
__global__ void k_updateVelocity(Parameters params, int* cellStarts, int* cellEnds,
    glm::vec2* positions, glm::vec2* velocities, uint* shoals, glm::vec2* newVelocities) {
    const int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= params.FISH_NUM) return;

    // Determine the indices of the cell the fish is in and the surrounding cells
    const auto pair = d_getCellIndices(positions[index].x, positions[index].y, params.CELL_LEN_INV, params.CELL_N);
    const int iStart = thrust::max(pair.first - 1, 0);
    const int iEnd = thrust::min(pair.first + 1, params.CELL_N - 1);
    const int jStart = thrust:: max(pair.second - 1, 0);
    const int jEnd = thrust::min(pair.second + 1, params.CELL_N - 1);

    const float maxVelSq = params.MAX_VEL * params.MAX_VEL;
    const glm::vec2 current_pos = positions[index];
    const glm::vec2 current_vel = velocities[index];
    const uint current_shoal = shoals[index];
    glm::vec2 separation(0, 0), alignment(0, 0), coherence(0, 0), center(0, 0);
    int count = 0;

    // Loop through the determined cells
    for (int i = iStart; i < iEnd; i++) {
        for (int j = jStart; j < jEnd; j++) {
            const auto id = d_getCellId(i, j, params.CELL_N);

            // Get the beginning and ending of the list of fish in the current cell
            const int start = cellStarts[id];
            const int end = cellEnds[id];

            if (start == -1) continue;

            // Loop through the fish in current cell
            for (int k = start; k <= end; k++) {
                if (k == index) continue;

                const glm::vec2 neighbour_pos = positions[k];
                const glm::vec2 neighbour_vel = velocities[k];
                const uint neighbour_shoal = shoals[k];

                // The vector between the fish and its neighbour
                glm::vec2 vector = neighbour_pos - current_pos;

                float distance = glm::distance(neighbour_pos, current_pos);
                float distanceSq = distance * distance;
                if (distanceSq > maxVelSq) {
                    vector = glm::normalize(vector) * params.MAX_VEL;
                    distanceSq = maxVelSq;
                }

                const float cosine = glm::dot(glm::normalize(current_vel), glm::normalize(vector));

                // If the fish can see the neighbour
                if (distance < params.R && cosine > params.COS_PHI) {
                    // Secure the division by zero
                    if (distance == 0) {
                        distance = 1.0E-16f;
                    }

                    // If the neighbour is from the same shoal
                    if (neighbour_shoal == current_shoal) {
                        separation -= 0.1f * vector / distanceSq;
                        alignment += neighbour_vel;
                        center += neighbour_pos;

                    // If the neighbour is from the different shoal
                    } else {
                        separation -= vector / distanceSq;
                    }
                    ++count;
                }
            }
        }
    }

    // Calculate the new velocity for the fish 
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

    // If user pressed the button, include the blackhole pulling effect
    if (params.BLACKHOLE.PULL) {
        glm::vec2 bh = glm::vec2(params.BLACKHOLE.X, params.BLACKHOLE.Y) - current_pos;
        const float R2 = glm::dot(bh, bh);
        bh = glm::normalize(bh) * params.BLACKHOLE.VEL / R2;
        newVelocity += bh;
        newVelocity /= 2;
    }

    // Make sure the value of velocity lies between MIN_VEL and MAX_VEL
    const float norm = glm::dot(newVelocity, newVelocity);
    if (norm > maxVelSq) {
        newVelocity = glm::normalize(newVelocity) * params.MAX_VEL;
    } else if (norm < params.MIN_VEL * params.MIN_VEL) {
        newVelocity = glm::normalize(newVelocity) * params.MIN_VEL;
    }

    // If simulation is not in wrapping mode, reflect the velocity, so the fish will bounce of the border (if necessary)
    if (!params.WRAP) {
        const glm::vec2 prediction = current_pos + newVelocity * params.DT;
        newVelocity = d_reflectVelocity(newVelocity, prediction.x, params.BOUNDS.MIN_X, params.BOUNDS.MAX_X, true);
        newVelocity = d_reflectVelocity(newVelocity, prediction.y, params.BOUNDS.MIN_Y, params.BOUNDS.MAX_Y, false);
    }

    // Copy the new calculate velocity
    newVelocities[index] = newVelocity;
}

void GPU::stepSimulation(const Parameters& params, Tables& tabs) {
    const int& N = params.FISH_NUM;
    const int& M = params.CELL_N * params.CELL_N;
    const dim3 blocksPerGrid_N((N + 1) / blockSize);
    const dim3 blocksPerGrid_M((M + 1) / blockSize);
    
    k_AssignIds <<< blocksPerGrid_N, threadsPerBlock >>> (N, params.CELL_N, params.CELL_LEN_INV, tabs.d_pos, tabs.d_fishId, tabs.d_cellId);
    cudaDeviceSynchronize();

    // Sort the cellId array, moving fish ids accordingly
    auto t_fishId = thrust::device_pointer_cast(tabs.d_fishId);
    auto t_cellId = thrust::device_pointer_cast(tabs.d_cellId);
    thrust::sort_by_key(t_cellId, t_cellId + N, t_fishId);

    k_resetTable <<< blocksPerGrid_M, threadsPerBlock >>> (M, tabs.d_cellStart);
    cudaDeviceSynchronize();

    k_resetTable <<< blocksPerGrid_M, threadsPerBlock >>> (M, tabs.d_cellEnd);
    cudaDeviceSynchronize();

    k_AssignCellStartEnd <<< blocksPerGrid_N, threadsPerBlock >>> (N, tabs.d_cellId, tabs.d_cellStart, tabs.d_cellEnd);
    cudaDeviceSynchronize();

    auto t_pos = thrust::device_pointer_cast(tabs.d_pos);
    auto t_vel = thrust::device_pointer_cast(tabs.d_vel);
    auto t_shoalId = thrust::device_pointer_cast(tabs.d_shoalId);
    auto t_pos_g = thrust::device_pointer_cast(tabs.d_pos_g);
    auto t_vel_g = thrust::device_pointer_cast(tabs.d_vel_g);
    auto t_shoalId_g = thrust::device_pointer_cast(tabs.d_shoalId_g);

    // Line up the position, velocity and shoal with the fish identifier
    thrust::gather(t_fishId, t_fishId + N, t_pos, t_pos_g);
    thrust::gather(t_fishId, t_fishId + N, t_vel, t_vel_g);
    thrust::gather(t_fishId, t_fishId + N, t_shoalId, t_shoalId_g);

    k_updateVelocity <<< blocksPerGrid_N, threadsPerBlock >>> (params, tabs.d_cellStart, tabs.d_cellEnd, tabs.d_pos_g, tabs.d_vel_g, tabs.d_shoalId_g, tabs.d_newVel);
    cudaDeviceSynchronize();

    const float& dt = params.DT;
    const bool& wrap = params.WRAP;
    const Parameters::Bounds& bounds = params.BOUNDS;

    auto t_newVel = thrust::device_pointer_cast(tabs.d_newVel);
    // Move fish to new positions based on calculated velocity
    thrust::transform(t_pos_g, t_pos_g + N, t_newVel, t_pos,
        [wrap, dt, bounds] __device__(const glm::vec2& pos, const glm::vec2& vel) {
            if (wrap) {
                return d_wrapPosition(pos + vel * dt, bounds);
            }
            return pos + vel * dt;
        });

    // Move newly calculated velocities
    thrust::copy(t_newVel, t_newVel + N, t_vel);

    // Move reshuffled shoal identifiers
    thrust::copy(t_shoalId_g, t_shoalId_g + N, t_shoalId);
}

/**************
 * copyToVBO *
 **************/

// Creates and copies vertices of the triangle, based on fish's position and velocity
__global__ void k_copyTriangleToVBO(int N, glm::vec2* pos, glm::vec2* vel, float* vbo) {
    const int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= N) return;

    // Rescale from [0,1] x [0,1] space to [-1,1] x [-1,1] space
    glm::vec2 posScaled = 2.0f * pos[index] - 1.0f;

    glm::vec2 velNormed = 0.01f * glm::normalize(vel[index]);
    glm::vec2 p1 = posScaled + 0.5f * glm::vec2(-velNormed.y, velNormed.x); // left
    glm::vec2 p2 = posScaled + 2.0f * velNormed; // top
    glm::vec2 p3 = posScaled + 0.5f * glm::vec2(velNormed.y, -velNormed.x); // right

    // Copy the data to buffer
    vbo[6 * index + 0] = p1.x;
    vbo[6 * index + 1] = p1.y;
    vbo[6 * index + 2] = p2.x;
    vbo[6 * index + 3] = p2.y;
    vbo[6 * index + 4] = p3.x;
    vbo[6 * index + 5] = p3.y;
}

// Copies shoal identifiers to given buffer
__global__ void k_copyShoalToVBO(int N, uint* shoal, uint* vbo) {
    const int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= N) return;

    // Copy the data to buffer
    vbo[3 * index + 0] = shoal[index];
    vbo[3 * index + 1] = shoal[index];
    vbo[3 * index + 2] = shoal[index];
}

void GPU::copyToVBO(const Parameters& params, Tables& tabs, float* vboTriangles, uint* vboShoals) {
    const dim3 blocksPerGrid((params.FISH_NUM + 1) / blockSize);

    k_copyTriangleToVBO << <blocksPerGrid, threadsPerBlock >> > (params.FISH_NUM, tabs.d_pos, tabs.d_vel, vboTriangles);
    checkCUDAError("k_copyTriangleToVBO failed!", __LINE__);
    cudaDeviceSynchronize();
    checkCUDAError("synchronize failed!", __LINE__);

    k_copyShoalToVBO << <blocksPerGrid, threadsPerBlock >> > (params.FISH_NUM, tabs.d_shoalId, vboShoals);
    checkCUDAError("k_copyShoalToVBO failed!", __LINE__);
    cudaDeviceSynchronize();
    checkCUDAError("synchronize failed!", __LINE__);
}

/******************
* endSimulation *
******************/

void GPU::endSimulation(Tables& tabs) {
  cudaFree(tabs.d_fishId);
  cudaFree(tabs.d_cellId);
  cudaFree(tabs.d_cellStart);
  cudaFree(tabs.d_cellEnd);
  cudaFree(tabs.d_pos);
  cudaFree(tabs.d_pos_g);
  cudaFree(tabs.d_vel);
  cudaFree(tabs.d_vel_g);
  cudaFree(tabs.d_shoalId);
  cudaFree(tabs.d_shoalId_g);
  cudaFree(tabs.d_newVel);
}
