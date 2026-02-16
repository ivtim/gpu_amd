#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <iostream>
#include <chrono>
#include <vector>


#define HIP_CHECK(cmd) \
  do { \
    hipError_t e = cmd; \
    if (e != hipSuccess) { \
      std::cerr << "HIP error: " << hipGetErrorString(e) \
                << " (" << e << ") at " << __FILE__ << ":" << __LINE__ << std::endl; \
      std::exit(EXIT_FAILURE); \
    } \
  } while(0)


__global__ void vecAdd(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

__global__ void vecAdd2(const float* A, const float* B, float* C, int N) {
    int pair_id = blockIdx.x * blockDim.x + threadIdx.x;
    int idx0 = pair_id * 2;
    int idx1 = idx0 + 1;

    if (idx0 >= N) return;

    C[idx0] = A[idx0] + B[idx0];
    C[idx1] = A[idx1] + B[idx1];
}

int main() {
    const int blockSize = 256;

    int devices_count = 0;
    HIP_CHECK(hipGetDeviceCount(&devices_count));

    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));

    hipDeviceProp_t prop = {};
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    std::cout << "GPU device name: " << prop.name << std::endl;
    std::cout << "CU: " << prop.multiProcessorCount << std::endl;
    std::cout << "Max threads per CU: " << prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "Max threads per Block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Warp size: " << prop.warpSize << std::endl;
    std::cout << "Max grid size: [" << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << "]" << std::endl;

    size_t maxElemntsByGrid = static_cast<size_t>(prop.maxGridSize[0]) * blockSize;
    std::cout << "Max count per single launch: " << maxElemntsByGrid << std::endl;

    size_t availableMemoryTotal;
    size_t availableMemoryFree;
    HIP_CHECK(hipMemGetInfo(&availableMemoryFree, &availableMemoryTotal));
    std::cout << "available Memory: " << availableMemoryFree << std::endl;
    size_t maxElementsByMemory = availableMemoryFree / (3 * sizeof(float));
    std::cout << "max Elements By Memory: " << maxElementsByMemory << std::endl;
    maxElementsByMemory = maxElementsByMemory * 0.7;
    std::cout << "max Elements By Memory: " << maxElementsByMemory << std::endl;

    size_t optimalN = std::min(maxElemntsByGrid, maxElementsByMemory);
    std::cout << "optimalN: " << optimalN << std::endl;
    optimalN = (optimalN / (blockSize * prop.warpSize)) * (blockSize * prop.warpSize);
    std::cout << "optimalN: " << optimalN << std::endl;

    size_t Nmin = 1 << 10;
    size_t Nmax = maxElementsByMemory;
    std::vector<size_t> Nvals;
    for (size_t n = Nmin; n <= Nmax; n <<= 1) Nvals.push_back(n);

    //const int N = 1000 << 20; 
    const int N = optimalN;
    const size_t bytes = N * sizeof(float);

    float* hA = new float[N];
    float* hB = new float[N];
    float* hC = new float[N];

    for (int i = 0; i < N; ++i) {
        hA[i] = static_cast<float>(i);
        hB[i] = static_cast<float>(2 * i);
    }

    float* dA = nullptr, * dB = nullptr, * dC = nullptr;
    HIP_CHECK(hipMalloc(&dA, bytes));
    HIP_CHECK(hipMalloc(&dB, bytes));
    HIP_CHECK(hipMalloc(&dC, bytes));

    HIP_CHECK(hipMemcpy(dA, hA, bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hB, bytes, hipMemcpyHostToDevice));

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    struct Result { int ILP;  size_t N; int block; double bandwidthMBs; double ms; };
    std::vector<Result> results;

    std::vector<int> blockCandidates = { 64, 128, 256, 512 };

    for (int block : blockCandidates) {
        for (size_t N2 : Nvals) {
            const int gridSize = (N2 + block - 1) / block;

            for (int w = 0; w < 3; ++w) {
                hipLaunchKernelGGL(vecAdd, dim3(gridSize), dim3(block), 0, 0, dA, dB, dC, (int)N2);
            }
            HIP_CHECK(hipDeviceSynchronize());

            const int repeats = 20;
            HIP_CHECK(hipEventRecord(start, 0));
            for (int r = 0; r < repeats; ++r) {
                hipLaunchKernelGGL(vecAdd, dim3(gridSize), dim3(block), 0, 0, dA, dB, dC, (int)N2);
            }
            HIP_CHECK(hipEventRecord(stop, 0));
            HIP_CHECK(hipEventSynchronize(stop));
            float ms = 0.0f;
            HIP_CHECK(hipEventElapsedTime(&ms, start, stop));

            double avg_ms = ms / repeats;
            double bytesProcessed = double(3) * double(N2) * double(sizeof(float));
            double bandwidthMBs = (bytesProcessed / (avg_ms / 1000.0)) / 1048576.0;

            results.push_back({ 1, N2, block, bandwidthMBs, avg_ms });
            //std::cout << "block=" << block << " N=" << N2 << " ms=" << avg_ms << " bandwidth(MB/s)=" << bandwidthMBs << std::endl;
        }
    }

    for (int block : blockCandidates) {
        for (size_t N2 : Nvals) {
            size_t N2origin = N2;
            N2 = N2 / 2;
            const int gridSize = (N2 + block - 1) / block;

            for (int w = 0; w < 3; ++w) {
                hipLaunchKernelGGL(vecAdd, dim3(gridSize), dim3(block), 0, 0, dA, dB, dC, (int)N2);
            }
            HIP_CHECK(hipDeviceSynchronize());

            const int repeats = 20;
            HIP_CHECK(hipEventRecord(start, 0));
            for (int r = 0; r < repeats; ++r) {
                hipLaunchKernelGGL(vecAdd, dim3(gridSize), dim3(block), 0, 0, dA, dB, dC, (int)N2);
            }
            HIP_CHECK(hipEventRecord(stop, 0));
            HIP_CHECK(hipEventSynchronize(stop));
            float ms = 0.0f;
            HIP_CHECK(hipEventElapsedTime(&ms, start, stop));

            double avg_ms = ms / repeats;
            double bytesProcessed = double(3) * double(N2) * double(sizeof(float));
            double bandwidthMBs = (bytesProcessed / (avg_ms / 1000.0)) / 1048576.0;

            results.push_back({ 2, N2origin, block, bandwidthMBs, avg_ms });
            //std::cout << "block=" << block << " N=" << N2 << " ms=" << avg_ms << " bandwidth(MB/s)=" << bandwidthMBs << std::endl;
        }
    }

    for (auto& res : results) {
        std::cout << res.ILP << " " << res.block << " " << res.N << " " << res.ms << " " << res.bandwidthMBs << std::endl;
    }

    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(hC, dC, bytes, hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(dA));
    HIP_CHECK(hipFree(dB));
    HIP_CHECK(hipFree(dC));
    delete[] hA;
    delete[] hB;
    delete[] hC;

    return 0;
}
