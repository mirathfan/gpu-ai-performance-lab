#pragma once

#include <cstdint>

namespace option_pricing {

enum class OptionType {
    Call,
    Put
};

struct OptionParams {
    double S0;
    double K;
    double r;
    double sigma;
    double T;
    OptionType type;
};

struct PricingResult {
    double price;
    double runtime_ms;
};

PricingResult price_option_cpu(
    const OptionParams& params,
    std::uint64_t paths,
    std::uint64_t seed);

PricingResult price_option_cuda(
    const OptionParams& params,
    std::uint64_t paths,
    std::uint64_t seed);

}  // namespace option_pricing
