#include "option_pricing.hpp"

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace option_pricing {
namespace {

constexpr int kThreadsPerBlock = 256;

void check_cuda(cudaError_t result, const char* context) {
    if (result != cudaSuccess) {
        std::ostringstream message;
        message << context << ": " << cudaGetErrorString(result);
        throw std::runtime_error(message.str());
    }
}

int choose_block_count(std::uint64_t paths) {
    int device = 0;
    check_cuda(cudaGetDevice(&device), "get active CUDA device");

    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, device), "get CUDA device properties");

    const std::uint64_t useful_blocks =
        (paths + kThreadsPerBlock - 1) / kThreadsPerBlock;
    const std::uint64_t capped_useful_blocks =
        std::min<std::uint64_t>(useful_blocks, std::numeric_limits<int>::max());

    const int occupancy_blocks = std::max(1, properties.multiProcessorCount * 8);
    return std::max(1, std::min(static_cast<int>(capped_useful_blocks), occupancy_blocks));
}

__global__ void monte_carlo_kernel(
    OptionParams params,
    std::uint64_t paths,
    std::uint64_t seed,
    double* block_sums) {
    extern __shared__ double shared_payoffs[];

    const unsigned int thread_id = threadIdx.x;
    const std::uint64_t global_thread_id =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t stride =
        static_cast<std::uint64_t>(blockDim.x) * gridDim.x;

    curandStatePhilox4_32_10_t rng_state;
    curand_init(static_cast<unsigned long long>(seed),
                static_cast<unsigned long long>(global_thread_id),
                0,
                &rng_state);

    const double drift = (params.r - 0.5 * params.sigma * params.sigma) * params.T;
    const double diffusion = params.sigma * sqrt(params.T);

    double payoff_sum = 0.0;
    for (std::uint64_t path = global_thread_id; path < paths; path += stride) {
        const double z = curand_normal_double(&rng_state);
        const double terminal_price = params.S0 * exp(drift + diffusion * z);

        const double payoff = (params.type == OptionType::Call)
            ? fmax(terminal_price - params.K, 0.0)
            : fmax(params.K - terminal_price, 0.0);

        payoff_sum += payoff;
    }

    shared_payoffs[thread_id] = payoff_sum;
    __syncthreads();

    for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (thread_id < offset) {
            shared_payoffs[thread_id] += shared_payoffs[thread_id + offset];
        }
        __syncthreads();
    }

    if (thread_id == 0) {
        block_sums[blockIdx.x] = shared_payoffs[0];
    }
}

}  // namespace

PricingResult price_option_cuda(
    const OptionParams& params,
    std::uint64_t paths,
    std::uint64_t seed) {
    if (paths == 0) {
        throw std::invalid_argument("paths must be greater than zero");
    }

    check_cuda(cudaFree(nullptr), "initialize CUDA context");

    const int block_count = choose_block_count(paths);
    double* device_block_sums = nullptr;
    check_cuda(
        cudaMalloc(&device_block_sums, sizeof(double) * block_count),
        "allocate block sums");

    cudaEvent_t start{};
    cudaEvent_t stop{};
    check_cuda(cudaEventCreate(&start), "create CUDA start event");
    check_cuda(cudaEventCreate(&stop), "create CUDA stop event");

    check_cuda(cudaEventRecord(start), "record CUDA start event");
    monte_carlo_kernel<<<block_count, kThreadsPerBlock, kThreadsPerBlock * sizeof(double)>>>(
        params,
        paths,
        seed,
        device_block_sums);
    check_cuda(cudaGetLastError(), "launch Monte Carlo CUDA kernel");
    check_cuda(cudaEventRecord(stop), "record CUDA stop event");
    check_cuda(cudaEventSynchronize(stop), "synchronize CUDA stop event");

    float runtime_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&runtime_ms, start, stop), "measure CUDA elapsed time");

    std::vector<double> host_block_sums(static_cast<std::size_t>(block_count));
    check_cuda(
        cudaMemcpy(
            host_block_sums.data(),
            device_block_sums,
            sizeof(double) * block_count,
            cudaMemcpyDeviceToHost),
        "copy block sums to host");

    check_cuda(cudaEventDestroy(start), "destroy CUDA start event");
    check_cuda(cudaEventDestroy(stop), "destroy CUDA stop event");
    check_cuda(cudaFree(device_block_sums), "free block sums");

    long double payoff_sum = 0.0L;
    for (const double block_sum : host_block_sums) {
        payoff_sum += block_sum;
    }

    const double average_payoff = static_cast<double>(payoff_sum / paths);
    const double discount = std::exp(-params.r * params.T);
    return PricingResult{discount * average_payoff, static_cast<double>(runtime_ms)};
}

}  // namespace option_pricing
