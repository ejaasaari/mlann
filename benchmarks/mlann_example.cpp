#include <H5Cpp.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "../cpp/rf-class-depth.h"

namespace {

constexpr int kNTrees = 10;
constexpr int kDepth = 14;
constexpr float kVotesRequired = 0.000005F;

struct BenchmarkOptions {
  std::string index_type = "RF";
  int n_trees = kNTrees;
  int depth = kDepth;
  int leaf_vote_threshold = 1;
  int query_repeats = 3;
  int max_queries = -1;
  int warmup_queries = 100;
  float density = -1.0F;
  std::vector<float> votes_required = {kVotesRequired};
};

struct DatasetConfig {
  const char *name;
  int k;
  Distance distance;
  bool normalize_corpus;
};

constexpr std::array<DatasetConfig, 4> kDatasets = {{
    {"coco-nomic-768-normalized", 100, IP, false},
    {"hotpotqa-harrier-640-normalized", 100, IP, false},
    {"llama-128-ip", 100, IP, false},
    {"yandex-200-cosine", 100, IP, true},
}};

template <typename Scalar>
using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

int checked_dimension(hsize_t dimension, const std::string &dataset_name) {
  if (dimension > static_cast<hsize_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("Dataset '" + dataset_name + "' is too large for MLANN");
  }
  return static_cast<int>(dimension);
}

template <typename Scalar>
Matrix<Scalar> read_matrix(H5::H5File &file, const std::string &dataset_name,
                           const H5::DataType &memory_type, int max_columns = -1) {
  H5::DataSet dataset = file.openDataSet(dataset_name);
  H5::DataSpace file_space = dataset.getSpace();
  if (file_space.getSimpleExtentNdims() != 2) {
    throw std::runtime_error("Dataset '" + dataset_name + "' must be two-dimensional");
  }

  std::array<hsize_t, 2> dimensions{};
  file_space.getSimpleExtentDims(dimensions.data());
  if (max_columns > 0) {
    if (dimensions[1] < static_cast<hsize_t>(max_columns)) {
      throw std::runtime_error("Dataset '" + dataset_name + "' has fewer than " +
                               std::to_string(max_columns) + " columns");
    }
    dimensions[1] = static_cast<hsize_t>(max_columns);
  }

  const int rows = checked_dimension(dimensions[0], dataset_name);
  const int columns = checked_dimension(dimensions[1], dataset_name);
  Matrix<Scalar> matrix(rows, columns);

  const std::array<hsize_t, 2> offset = {0, 0};
  file_space.selectHyperslab(H5S_SELECT_SET, dimensions.data(), offset.data());
  H5::DataSpace memory_space(2, dimensions.data());
  dataset.read(matrix.data(), memory_type, memory_space, file_space);
  return matrix;
}

const DatasetConfig *find_dataset(const std::string &name) {
  const auto dataset = std::find_if(kDatasets.begin(), kDatasets.end(),
                                    [&](const auto &config) { return name == config.name; });
  return dataset == kDatasets.end() ? nullptr : &*dataset;
}

void normalize_rows(RowMatrix &matrix, const std::string &dataset_name) {
  for (int row = 0; row < matrix.rows(); ++row) {
    const float norm = matrix.row(row).norm();
    if (!(norm > std::numeric_limits<float>::epsilon())) {
      throw std::runtime_error("Dataset '" + dataset_name + "' contains a zero vector");
    }
    matrix.row(row) /= norm;
  }
}

void print_usage(const char *program) {
  std::cerr << "Usage: " << program << " <dataset> [options]\n"
            << "Options: --trees=N --depth=N --density=F --leaf-votes=N "
               "--votes-required=F1,F2,... "
               "--query-repeats=N --queries=N --warmup=N\n"
            << "Available datasets: ";
  for (std::size_t i = 0; i < kDatasets.size(); ++i) {
    if (i != 0) std::cerr << ", ";
    std::cerr << kDatasets[i].name;
  }
  std::cerr << '\n';
}

std::string option_value(const std::string &argument, const std::string &name) {
  const std::string prefix = "--" + name + "=";
  if (argument.rfind(prefix, 0) != 0) return {};
  return argument.substr(prefix.size());
}

int parse_integer(const std::string &value, const std::string &name) {
  std::size_t consumed = 0;
  const int parsed = std::stoi(value, &consumed);
  if (consumed != value.size()) throw std::invalid_argument("Invalid --" + name);
  return parsed;
}

float parse_float(const std::string &value, const std::string &name) {
  std::size_t consumed = 0;
  const float parsed = std::stof(value, &consumed);
  if (consumed != value.size()) throw std::invalid_argument("Invalid --" + name);
  return parsed;
}

BenchmarkOptions parse_options(int argc, char **argv) {
  BenchmarkOptions options;
  int first_option = 2;

  for (int i = first_option; i < argc; ++i) {
    const std::string argument = argv[i];
    std::string value;
    if (!(value = option_value(argument, "trees")).empty()) {
      options.n_trees = parse_integer(value, "trees");
    } else if (!(value = option_value(argument, "depth")).empty()) {
      options.depth = parse_integer(value, "depth");
    } else if (!(value = option_value(argument, "density")).empty()) {
      options.density = parse_float(value, "density");
    } else if (!(value = option_value(argument, "leaf-votes")).empty()) {
      options.leaf_vote_threshold = parse_integer(value, "leaf-votes");
    } else if (!(value = option_value(argument, "query-repeats")).empty()) {
      options.query_repeats = parse_integer(value, "query-repeats");
    } else if (!(value = option_value(argument, "queries")).empty()) {
      options.max_queries = parse_integer(value, "queries");
    } else if (!(value = option_value(argument, "warmup")).empty()) {
      options.warmup_queries = parse_integer(value, "warmup");
    } else if (!(value = option_value(argument, "votes-required")).empty()) {
      options.votes_required.clear();
      std::stringstream stream(value);
      std::string item;
      while (std::getline(stream, item, ',')) {
        options.votes_required.push_back(parse_float(item, "votes-required"));
      }
      if (options.votes_required.empty()) {
        throw std::invalid_argument("At least one votes_required threshold is required");
      }
    } else {
      throw std::invalid_argument("Unknown option: " + argument);
    }
  }
  if (options.query_repeats <= 0) throw std::invalid_argument("query-repeats must be positive");
  if (options.max_queries == 0 || options.max_queries < -1) {
    throw std::invalid_argument("queries must be positive or -1 for all queries");
  }
  if (options.warmup_queries < 0) throw std::invalid_argument("warmup must be non-negative");
  if (options.leaf_vote_threshold <= 0) {
    throw std::invalid_argument("leaf-votes must be positive");
  }
  if (std::any_of(options.votes_required.begin(), options.votes_required.end(),
                  [](float threshold) { return !(threshold > 0.0F); })) {
    throw std::invalid_argument("votes-required thresholds must be positive");
  }
  if (options.density > 1.0F || options.density == 0.0F) {
    throw std::invalid_argument("density must be in (0, 1] or negative for auto");
  }
  return options;
}

void validate_shapes(const RowMatrix &train, const RowMatrix &training_queries,
                     const UIntRowMatrix &training_neighbors, const RowMatrix &test,
                     const UIntRowMatrix &ground_truth, int k) {
  if (train.rows() == 0 || train.cols() == 0) {
    throw std::runtime_error("The training dataset is empty");
  }
  if (training_queries.cols() != train.cols() || test.cols() != train.cols()) {
    throw std::runtime_error("Train, learn, and test vectors must have the same dimension");
  }
  if (training_neighbors.rows() != training_queries.rows()) {
    throw std::runtime_error("learn and learn_neighbors must have the same number of rows");
  }
  if (ground_truth.rows() != test.rows()) {
    throw std::runtime_error("test and neighbors must have the same number of rows");
  }
  if (training_neighbors.cols() != k || ground_truth.cols() != k) {
    throw std::runtime_error("Neighbor datasets do not contain the requested number of columns");
  }
}

double recall_at_k(const std::vector<int> &results, const UIntRowMatrix &ground_truth, int k,
                   int n_queries) {
  std::size_t hits = 0;
  for (int row = 0; row < n_queries; ++row) {
    std::unordered_set<uint32_t> expected;
    std::unordered_set<int> seen;
    expected.reserve(static_cast<std::size_t>(k));
    seen.reserve(static_cast<std::size_t>(k));

    for (int column = 0; column < k; ++column) {
      expected.insert(ground_truth(row, column));
    }
    for (int column = 0; column < k; ++column) {
      const int result = results[static_cast<std::size_t>(row) * k + column];
      if (result >= 0 && seen.insert(result).second &&
          expected.count(static_cast<uint32_t>(result)) != 0) {
        ++hits;
      }
    }
  }

  return static_cast<double>(hits) /
         (static_cast<double>(n_queries) * ground_truth.cols());
}

}  // namespace

int main(int argc, char **argv) {
  std::cout << std::unitbuf;
  if (argc < 2) {
    print_usage(argv[0]);
    return 1;
  }

  const DatasetConfig *config = find_dataset(argv[1]);
  if (config == nullptr) {
    print_usage(argv[0]);
    return 1;
  }

  H5::Exception::dontPrint();
  try {
    const BenchmarkOptions options = parse_options(argc, argv);
    const std::string hdf5_name = std::string(config->name) + ".hdf5";
    H5::H5File file(hdf5_name, H5F_ACC_RDONLY);

    RowMatrix train = read_matrix<float>(file, "train", H5::PredType::NATIVE_FLOAT);
    RowMatrix training_queries = read_matrix<float>(file, "learn", H5::PredType::NATIVE_FLOAT);
    UIntRowMatrix training_neighbors =
        read_matrix<uint32_t>(file, "learn_neighbors", H5::PredType::NATIVE_UINT32, config->k);
    RowMatrix test = read_matrix<float>(file, "test", H5::PredType::NATIVE_FLOAT);
    UIntRowMatrix ground_truth =
        read_matrix<uint32_t>(file, "neighbors", H5::PredType::NATIVE_UINT32, config->k);

    if (config->normalize_corpus) normalize_rows(train, "train");

    validate_shapes(train, training_queries, training_neighbors, test, ground_truth, config->k);

    std::cout << "Database vectors: " << train.rows() << ", dimension: " << train.cols() << '\n';
    std::cout << "MLANN training queries: " << training_queries.rows() << '\n';
    std::cout << "Building " << options.index_type << " MLANN index...\n";

    RFClass index(train.data(), train.rows(), train.cols());
    auto start = std::chrono::steady_clock::now();
    index.grow(options.n_trees, options.depth, training_neighbors, training_queries,
               options.density, options.leaf_vote_threshold);
    double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    std::cout << std::fixed << std::setprecision(2) << "Build time (s): " << elapsed << '\n';
    std::cout << std::defaultfloat << "Parameters: n_trees=" << options.n_trees
              << ", depth=" << options.depth << ", density=" << options.density
              << ", leaf_votes=" << options.leaf_vote_threshold;
    std::cout << '\n';

    const int test_rows = static_cast<int>(test.rows());
    const int n_queries =
        options.max_queries > 0 ? std::min(options.max_queries, test_rows) : test_rows;
    const int n_warmup = std::min(options.warmup_queries, n_queries);
    std::cout << "Benchmark queries: " << n_queries << ", warmup queries: " << n_warmup << '\n';

    const std::size_t result_count =
        static_cast<std::size_t>(n_queries) * static_cast<std::size_t>(config->k);
    std::vector<int> results(result_count);
    std::vector<float> distances(result_count);

    for (const float votes_required : options.votes_required) {
      // Warm the corpus pages and routing code before measuring this operating point.
      for (int row = 0; row < n_warmup; ++row) {
        const std::size_t offset = static_cast<std::size_t>(row) * config->k;
        index.query(test.row(row).data(), config->k, votes_required, results.data() + offset,
                    config->distance, distances.data() + offset);
      }

      std::vector<double> timings;
      timings.reserve(options.query_repeats);
      std::size_t elected_total = 0;
      for (int repetition = 0; repetition < options.query_repeats; ++repetition) {
        std::size_t repetition_elected = 0;
        start = std::chrono::steady_clock::now();
        for (int row = 0; row < n_queries; ++row) {
          const std::size_t offset = static_cast<std::size_t>(row) * config->k;
          int elected = 0;
          index.query(test.row(row).data(), config->k, votes_required, results.data() + offset,
                      config->distance, distances.data() + offset, &elected);
          repetition_elected += static_cast<std::size_t>(elected);
        }
        timings.push_back(
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
        elected_total = repetition_elected;
      }
      std::sort(timings.begin(), timings.end());
      elapsed = timings[timings.size() / 2];

      const double recall = recall_at_k(results, ground_truth, config->k, n_queries);
      const double average_ms = elapsed / n_queries * 1e3;
      const double average_elected = static_cast<double>(elected_total) / n_queries;
      std::cout << std::defaultfloat << std::setprecision(9)
                << "votes_required=" << votes_required << std::fixed << std::setprecision(5)
                << " recall=" << recall << std::setprecision(3) << " query_ms=" << average_ms
                << std::setprecision(2) << " elected=" << average_elected << '\n';
    }
  } catch (const H5::Exception &error) {
    std::cerr << "HDF5 error: " << error.getDetailMsg() << '\n';
    return 1;
  } catch (const std::exception &error) {
    std::cerr << "Error: " << error.what() << '\n';
    return 1;
  }

  return 0;
}
