#include "option_pricing.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using option_pricing::OptionParams;
using option_pricing::OptionType;
using option_pricing::PricingResult;

struct CliOptions {
    std::uint64_t paths = 1'000'000;
    std::uint64_t seed = 1234;
    std::string output = "results/monte_carlo_benchmark.csv";
    OptionParams params{
        100.0,
        100.0,
        0.05,
        0.2,
        1.0,
        OptionType::Call
    };
};

struct BackendRow {
    std::string backend;
    PricingResult result;
    double paths_per_sec;
    double speedup;
};

std::string to_lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

OptionType parse_option_type(const std::string& value) {
    const std::string lowered = to_lower(value);
    if (lowered == "call") {
        return OptionType::Call;
    }
    if (lowered == "put") {
        return OptionType::Put;
    }
    throw std::invalid_argument("--type must be call or put");
}

const char* option_type_name(OptionType type) {
    return type == OptionType::Call ? "call" : "put";
}

std::string require_value(int& index, int argc, char** argv) {
    if (index + 1 >= argc) {
        throw std::invalid_argument(std::string("missing value for ") + argv[index]);
    }
    ++index;
    return argv[index];
}

void print_help(const char* program) {
    std::cout
        << "Usage: " << program << " [options]\n\n"
        << "Options:\n"
        << "  --paths <n>       Number of Monte Carlo paths\n"
        << "  --S0 <value>      Initial stock price\n"
        << "  --K <value>       Strike price\n"
        << "  --r <value>       Risk-free rate\n"
        << "  --sigma <value>   Volatility\n"
        << "  --T <value>       Time to maturity\n"
        << "  --type <call|put> Option type\n"
        << "  --seed <n>        Random seed\n"
        << "  --output <path>   CSV output path\n"
        << "  --help            Show this help\n";
}

CliOptions parse_args(int argc, char** argv) {
    CliOptions options;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "--help") {
            print_help(argv[0]);
            std::exit(0);
        } else if (arg == "--paths") {
            options.paths = std::stoull(require_value(i, argc, argv));
        } else if (arg == "--S0") {
            options.params.S0 = std::stod(require_value(i, argc, argv));
        } else if (arg == "--K") {
            options.params.K = std::stod(require_value(i, argc, argv));
        } else if (arg == "--r") {
            options.params.r = std::stod(require_value(i, argc, argv));
        } else if (arg == "--sigma") {
            options.params.sigma = std::stod(require_value(i, argc, argv));
        } else if (arg == "--T") {
            options.params.T = std::stod(require_value(i, argc, argv));
        } else if (arg == "--type") {
            options.params.type = parse_option_type(require_value(i, argc, argv));
        } else if (arg == "--seed") {
            options.seed = std::stoull(require_value(i, argc, argv));
        } else if (arg == "--output") {
            options.output = require_value(i, argc, argv);
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }

    if (options.paths == 0) {
        throw std::invalid_argument("--paths must be greater than zero");
    }
    if (options.params.S0 <= 0.0) {
        throw std::invalid_argument("--S0 must be greater than zero");
    }
    if (options.params.K <= 0.0) {
        throw std::invalid_argument("--K must be greater than zero");
    }
    if (options.params.sigma < 0.0) {
        throw std::invalid_argument("--sigma must be non-negative");
    }
    if (options.params.T <= 0.0) {
        throw std::invalid_argument("--T must be greater than zero");
    }

    return options;
}

double paths_per_second(std::uint64_t paths, double runtime_ms) {
    if (runtime_ms <= 0.0) {
        return 0.0;
    }
    return static_cast<double>(paths) / (runtime_ms / 1000.0);
}

void print_table(const std::vector<BackendRow>& rows) {
    std::cout << "\n";
    std::cout << std::left << std::setw(10) << "backend"
              << std::right << std::setw(18) << "option price"
              << std::setw(16) << "runtime ms"
              << std::setw(18) << "paths/sec"
              << std::setw(12) << "speedup"
              << "\n";
    std::cout << std::string(76, '-') << "\n";

    for (const BackendRow& row : rows) {
        std::cout << std::left << std::setw(10) << row.backend
                  << std::right << std::fixed << std::setprecision(6)
                  << std::setw(18) << row.result.price
                  << std::setw(16) << row.result.runtime_ms
                  << std::setw(18) << std::setprecision(2) << row.paths_per_sec
                  << std::setw(11) << std::setprecision(3) << row.speedup << "x"
                  << "\n";
    }
}

void append_csv(
    const std::filesystem::path& output_path,
    const CliOptions& options,
    const std::vector<BackendRow>& rows) {
    const std::filesystem::path parent = output_path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }

    const bool write_header =
        !std::filesystem::exists(output_path) ||
        std::filesystem::file_size(output_path) == 0;

    std::ofstream output(output_path, std::ios::app);
    if (!output) {
        throw std::runtime_error("could not open CSV output: " + output_path.string());
    }

    if (write_header) {
        output
            << "backend,option_type,paths,S0,K,r,sigma,T,seed,"
            << "option_price,runtime_ms,paths_per_sec,speedup\n";
    }

    output << std::fixed << std::setprecision(10);
    for (const BackendRow& row : rows) {
        output
            << row.backend << ","
            << option_type_name(options.params.type) << ","
            << options.paths << ","
            << options.params.S0 << ","
            << options.params.K << ","
            << options.params.r << ","
            << options.params.sigma << ","
            << options.params.T << ","
            << options.seed << ","
            << row.result.price << ","
            << row.result.runtime_ms << ","
            << row.paths_per_sec << ","
            << row.speedup << "\n";
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const CliOptions options = parse_args(argc, argv);

        std::cout
            << "Monte Carlo European " << option_type_name(options.params.type)
            << " option benchmark\n"
            << "paths=" << options.paths
            << ", S0=" << options.params.S0
            << ", K=" << options.params.K
            << ", r=" << options.params.r
            << ", sigma=" << options.params.sigma
            << ", T=" << options.params.T
            << ", seed=" << options.seed
            << "\n";

        const PricingResult cpu =
            option_pricing::price_option_cpu(options.params, options.paths, options.seed);
        const PricingResult cuda =
            option_pricing::price_option_cuda(options.params, options.paths, options.seed);

        const double cpu_speed = paths_per_second(options.paths, cpu.runtime_ms);
        const double cuda_speed = paths_per_second(options.paths, cuda.runtime_ms);
        const double cuda_speedup =
            cuda.runtime_ms > 0.0 ? cpu.runtime_ms / cuda.runtime_ms : 0.0;

        const std::vector<BackendRow> rows{
            BackendRow{"CPU", cpu, cpu_speed, 1.0},
            BackendRow{"CUDA", cuda, cuda_speed, cuda_speedup}
        };

        print_table(rows);
        append_csv(options.output, options, rows);

        std::cout << "\nCSV output: " << options.output << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
}
