#define GLM_FORCE_CUDA
#include "kernel.h"

/******************
* Error handling *
******************/

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

__host__ __device__ glm::vec2 d_generateRandomVec2(const int& time, const int& index, const float& a, const float& b) {
    thrust::minstd_rand rng(index * time);
    thrust::uniform_real_distribution<float> U(a, b);
    return glm::vec2(U(rng), U(rng));
}

__global__ void k_generateRandomFloatArray(long time, int N, float a, float b, glm::vec2* array) {
    const int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= N) return;
    glm::vec2 rand = d_generateRandomVec2(time, index, a, b);
    array[index].x = rand.x;
    array[index].y = rand.y;
}

__host__ __device__ uint d_generateRandomUint(const int& time, const int& index, const int& scale) {
    thrust::minstd_rand rng(index * time);
    thrust::uniform_real_distribution<uint> U(0, scale);
    return U(rng);
}

__global__ void k_generateRandomUintArray(long time, int N, int scale, uint* array) {
    const int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= N) return;
    array[index] = d_generateRandomUint(time, index, scale);
}

void Global::initSimulation(const Parameters& params, Tables& tabs) {
    const hrClock::time_point time = hrClock::now();
    const int N = params.FISH_NUM;
    const int M = params.CELL_N * params.CELL_N;
    const dim3 blocksPerGrid((N + 1) / blockSize);

    cudaMalloc(reinterpret_cast<void**>(&tabs.d_fishId), N * sizeof(uint));
    checkCUDAError("cudaMalloc d_fishId failed!", __LINE__);

    cudaMalloc(reinterpret_cast<void**>(&tabs.d_cellId), N * sizeof(uint));
    checkCUDAError("cudaMalloc d_cellId failed!", __LINE__);

    cudaMalloc(reinterpret_cast<void**>(&tabs.d_cellStart), M * sizeof(uint));
    checkCUDAError("cudaMalloc d_cellStart failed!", __LINE__);

    cudaMalloc(reinterpret_cast<void**>(&tabs.d_cellEnd), M * sizeof(uint));
    checkCUDAError("cudaMalloc d_cellEnd failed!", __LINE__);

    cudaMalloc(reinterpret_cast<void**>(&tabs.d_pos), N * sizeof(glm::vec2));
    checkCUDAError("cudaMalloc d_pos failed!", __LINE__);

    cudaMalloc(reinterpret_cast<void**>(&tabs.d_pos_g), N * sizeof(glm::vec2));
    checkCUDAError("cudaMalloc d_pos_g failed!", __LINE__);

    cudaMalloc(reinterpret_cast<void**>(&tabs.d_vel), N * sizeof(glm::vec2));
    checkCUDAError("cudaMalloc d_vel failed!", __LINE__);

    cudaMalloc(reinterpret_cast<void**>(&tabs.d_vel_g), N * sizeof(glm::vec2));
    checkCUDAError("cudaMalloc d_vel_g failed!", __LINE__);

    cudaMalloc(reinterpret_cast<void**>(&tabs.d_shoalId), N * sizeof(glm::vec2));
    checkCUDAError("cudaMalloc d_shoalId failed!", __LINE__);

    cudaMalloc(reinterpret_cast<void**>(&tabs.d_shoalId_g), N * sizeof(glm::vec2));
    checkCUDAError("cudaMalloc d_shoalId_g failed!", __LINE__);

    cudaMalloc(reinterpret_cast<void**>(&tabs.d_newVel), N * sizeof(glm::vec2));
    checkCUDAError("cudaMalloc d_newVel failed!", __LINE__);

    long duration = (hrClock::now() - time).count();

    k_generateRandomFloatArray <<< blocksPerGrid, threadsPerBlock >>> (duration, N, params.BOUNDS.MIN_X, params.BOUNDS.MAX_X, tabs.d_pos);
    k_generateRandomFloatArray <<< blocksPerGrid, threadsPerBlock >>> (duration, N, -0.1, 0.1, tabs.d_vel);
    checkCUDAError("k_generateRandomFloatArray failed!", __LINE__);

    k_generateRandomUintArray <<<blocksPerGrid, threadsPerBlock >>> (duration, N, params.SHOAL_NUM, tabs.d_shoalId);
    checkCUDAError("k_generateRandomUintArray failed!", __LINE__);

    cudaDeviceSynchronize();
}

/******************
* stepSimulation *
******************/

__host__ __device__ thrust::pair<int, int> d_getCellIndices(const float& x, const float& y, const float& L, const int& N) {
    int i = static_cast<int>(x * L);
    int j = static_cast<int>(y * L);
    // skrajny przypadek, w którym x lub y le¿¹ na górnej granicy
    if (i >= N) {
        i = N - 1;
    }
    if (j >= N) {
        j = N - 1;
    }
    return thrust::make_pair(i, j);
}

__host__ __device__ int d_getCellId(const int& i, const int& j, const int& N) {
    return i * N + j;
}

__global__ void k_AssignIds(int N, int cellLineNumber, float inverseCellWidth, glm::vec2* positions, uint* fishIndices, uint* cellIndices) {
    const int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= N) return;
    const auto pair = d_getCellIndices(positions[index].x, positions[index].y, inverseCellWidth, cellLineNumber);
    int id = d_getCellId(pair.first, pair.second, cellLineNumber);
    cellIndices[index] = id;
    fishIndices[index] = index;  
}

__global__ void k_resetTable(int N, int* array, int value = -1) {
    const int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= N) return;
    array[index] = value;
}

__global__ void k_AssignCellStartEnd(int N, uint* cellIndices, int* cellStarts, int* cellEnds) {
    const int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= N) return;

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

__host__ __device__ glm::vec2 d_reflectVelocity(const glm::vec2& vel, const float& val, const float& min, const float& max, const bool& side) {
    if (val < min || val > max) {
        if (side) {
            return glm::vec2(-vel.x, vel.y);
        } else {
            return glm::vec2(vel.x, -vel.y);
        }
    }
    return vel;
}

__global__ void k_updateVelocity(Global::Parameters params, int* cellStarts, int* cellEnds,
                                 glm::vec2* positions, glm::vec2* velocities, uint* shoals, glm::vec2* newVelocities) {
    const int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= params.FISH_NUM) return;

    const auto pair = d_getCellIndices(positions[index].x, positions[index].y, params.CELL_LEN_INV, params.CELL_N);
    const int iStart = thrust::max(pair.first - 1, 0);
    const int iEnd = thrust::min(pair.first + 1, params.CELL_N - 1);
    const int jStart = thrust:: max(pair.second - 1, 0);
    const int jEnd = thrust::min(pair.second + 1, params.CELL_N - 1);

    const glm::vec2 current_pos = positions[index];
    const glm::vec2 current_vel = velocities[index];
    const uint current_shoal = shoals[index];
    glm::vec2 separation(0, 0), alignment(0, 0), coherence(0, 0), center(0, 0);
    int count = 0;

    for (int i = iStart; i < iEnd; i++) {
        for (int j = jStart; j < jEnd; j++) {
            const auto id = d_getCellId(i, j, params.CELL_N);

            const int start = cellStarts[id];
            const int end = cellEnds[id];

            if (start == -1) continue;

            for (int k = start; k <= end; k++) {
                if (k == index) continue;
                const glm::vec2 neighbour_pos = positions[k];
                const glm::vec2 neighbour_vel = velocities[k];
                const uint neighbour_shoal = shoals[k];

                const float distance = glm::distance(neighbour_pos, current_pos);
                const glm::vec2 vector = neighbour_pos - current_pos;
                const float cosine = glm::dot(glm::normalize(current_vel), glm::normalize(vector));

                if (distance < params.R && cosine > params.COS_PHI) {
                    if (neighbour_shoal == current_shoal) {
                        separation -= vector / (distance * distance);
                        alignment += neighbour_vel;
                        center += neighbour_pos;
                        ++count;
                    } else
                    {
                        separation -= vector / (2 * distance * distance);
                    }
                }
            }
        }
    }

    // na ten moment zwyk³a œrednia wa¿ona
    glm::vec2 newVelocity = current_vel;
    newVelocity += separation * params.W_SEP;
    if (count != 0) {
        alignment /= count;
        newVelocity += alignment * params.W_ALI;

        center /= count;
        coherence = (center - current_pos);
        newVelocity += coherence * params.W_COH;

        newVelocity /= 4;
    } else {
        newVelocity /= 2;
    }

    const float norm = dot(newVelocity, newVelocity);
    if (norm > params.MAX_VEL * params.MAX_VEL) {
        newVelocity = glm::normalize(newVelocity) * params.MAX_VEL;
    } else if (norm < params.MIN_VEL * params.MIN_VEL) {
        newVelocity = glm::normalize(newVelocity) * params.MIN_VEL;
    }

    // odbij prêdkoœæ jeœli to konieczne
    const glm::vec2 prediction = current_pos + newVelocity * params.DT;
    newVelocity = d_reflectVelocity(newVelocity, prediction.x, params.BOUNDS.MIN_X, params.BOUNDS.MAX_X, true);
    newVelocity = d_reflectVelocity(newVelocity, prediction.y, params.BOUNDS.MIN_Y, params.BOUNDS.MAX_Y, false);
    newVelocities[index] = newVelocity;
}

__host__ __device__ float d_reflect(float val, const float& min, const float& max) {
    if (val < min) {
        float newVal = min + (min - val);
        val = newVal > max ? min : newVal;
    } else if (val > max) {
        float newVal = max - (val - max);
        val = newVal < min ? max : newVal;
    }
    return val;
}

__host__ __device__ glm::vec2 d_reflectPosition(glm::vec2 position, const Global::Parameters::Bounds& bounds) {
    position.x = d_reflect(position.x, bounds.MIN_X, bounds.MAX_X);
    position.y = d_reflect(position.y, bounds.MIN_Y, bounds.MAX_Y);
    return position;
}

void Global::stepSimulation(const Parameters& params, Tables& tabs) {
    const int& N = params.FISH_NUM;
    const int& M = params.CELL_N * params.CELL_N;
    const dim3 blocksPerGrid_N((N + 1) / blockSize);
    const dim3 blocksPerGrid_M((M + 1) / blockSize);
    
    k_AssignIds <<< blocksPerGrid_N, threadsPerBlock >>> (N, params.CELL_N, params.CELL_LEN_INV, tabs.d_pos, tabs.d_fishId, tabs.d_cellId);

    auto t_fishId = thrust::device_pointer_cast(tabs.d_fishId);
    auto t_cellId = thrust::device_pointer_cast(tabs.d_cellId);
    thrust::sort_by_key(t_cellId, t_cellId + N, t_fishId);

    k_resetTable <<< blocksPerGrid_M, threadsPerBlock >>> (M, tabs.d_cellStart);
    k_resetTable <<< blocksPerGrid_M, threadsPerBlock >>> (M, tabs.d_cellEnd);
    k_AssignCellStartEnd <<< blocksPerGrid_N, threadsPerBlock >>> (N, tabs.d_cellId, tabs.d_cellStart, tabs.d_cellEnd);

    auto t_pos = thrust::device_pointer_cast(tabs.d_pos);
    auto t_vel = thrust::device_pointer_cast(tabs.d_vel);
    auto t_shoalId = thrust::device_pointer_cast(tabs.d_shoalId);
    auto t_pos_g = thrust::device_pointer_cast(tabs.d_pos_g);
    auto t_vel_g = thrust::device_pointer_cast(tabs.d_vel_g);
    auto t_shoalId_g = thrust::device_pointer_cast(tabs.d_shoalId_g);

    thrust::gather(t_fishId, t_fishId + N, t_pos, t_pos_g);
    thrust::gather(t_fishId, t_fishId + N, t_vel, t_vel_g);
    thrust::gather(t_fishId, t_fishId + N, t_shoalId, t_shoalId_g);

    k_updateVelocity <<< blocksPerGrid_N, threadsPerBlock >>> (params, tabs.d_cellStart, tabs.d_cellEnd, tabs.d_pos_g, tabs.d_vel_g, tabs.d_shoalId_g, tabs.d_newVel);

    const float& dt = params.DT;
    const Parameters::Bounds& bounds = params.BOUNDS;

    auto t_newVel = thrust::device_pointer_cast(tabs.d_newVel);
    thrust::transform(t_pos_g, t_pos_g + N, t_newVel, t_pos, [dt, bounds] __device__(const glm::vec2& pos, const glm::vec2& vel) {
        return d_reflectPosition(pos + vel * dt, bounds);
    });

    thrust::swap(t_vel, t_newVel);
    thrust::swap(t_shoalId, t_shoalId_g);
}

/**************
 * copyToVBO *
 **************/

__global__ void k_copyTabToVBO(int N, glm::vec2* tab, float* vbo) {
    const int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= N) return;
    vbo[2 * index + 0] = 2 * tab[index].x - 1.0f;
    vbo[2 * index + 1] = 2 * tab[index].y - 1.0f;
}

void Global::copyToVBO(const Parameters& params, Tables& tabs, float* d_vboPositions, float* d_vboVelocities) {
    const dim3 blocksPerGrid((params.FISH_NUM + 1) / blockSize);

    k_copyTabToVBO << <blocksPerGrid, threadsPerBlock >> > (params.FISH_NUM, tabs.d_pos, d_vboPositions);
    k_copyTabToVBO << <blocksPerGrid, threadsPerBlock >> > (params.FISH_NUM, tabs.d_vel, d_vboVelocities);
    // and later also shoalId
    checkCUDAError("copyToVBO failed!", __LINE__);
    cudaDeviceSynchronize();
}

/******************
* endSimulation *
******************/

void Global::endSimulation(Tables& tabs) {
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
