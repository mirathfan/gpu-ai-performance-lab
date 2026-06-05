#include "option_pricing.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>

namespace option_pricing {

PricingResult price_option_cpu(
    const OptionParams& params,
    std::uint64_t paths,
    std::uint64_t seed) {
    if (paths == 0) {
        throw std::invalid_argument("paths must be greater than zero");
    }

    std::seed_seq seed_sequence{
        static_cast<std::uint32_t>(seed & 0xffffffffULL),
        static_cast<std::uint32_t>((seed >> 32) & 0xffffffffULL)
    };
    std::mt19937 rng(seed_sequence);
    std::normal_distribution<double> normal(0.0, 1.0);

    const double drift = (params.r - 0.5 * params.sigma * params.sigma) * params.T;
    const double diffusion = params.sigma * std::sqrt(params.T);
    const double discount = std::exp(-params.r * params.T);

    const auto start = std::chrono::high_resolution_clock::now();

    long double payoff_sum = 0.0L;
    for (std::uint64_t i = 0; i < paths; ++i) {
        const double z = normal(rng);
        const double terminal_price = params.S0 * std::exp(drift + diffusion * z);

        const double payoff = (params.type == OptionType::Call)
            ? std::max(terminal_price - params.K, 0.0)
            : std::max(params.K - terminal_price, 0.0);

        payoff_sum += payoff;
    }

    const auto end = std::chrono::high_resolution_clock::now();
    const double runtime_ms =
        std::chrono::duration<double, std::milli>(end - start).count();

    const double average_payoff = static_cast<double>(payoff_sum / paths);
    return PricingResult{discount * average_payoff, runtime_ms};
}

}  // namespace option_pricing
