// SPDX-FileCopyrightText: 2026 Stefano Cacciatore
// SPDX-License-Identifier: MIT

#include "kodama/kodama.hpp"
#include "metal_backend.hpp"

#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <complex>
#include <condition_variable>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <exception>
#include <functional>
#include <limits>
#include <mutex>
#include <numeric>
#include <random>
#include <string>
#include <stdexcept>
#include <thread>
#include <utility>

namespace kodama {
namespace {

using Clock = std::chrono::steady_clock;

std::pair<double, double> find_ab_params(const double spread, const double min_dist) {
  if (std::abs(spread - 1.0) < 1e-12 && std::abs(min_dist - 0.01) < 1e-12) {
    return {1.895605865596314, 0.8006377738365004};
  }
  if (std::abs(spread - 1.0) < 1e-12 && std::abs(min_dist - 0.1) < 1e-12) {
    return {1.5769434601962196, 0.8950608781227859};
  }
  if (std::abs(spread - 1.0) < 1e-12 && std::abs(min_dist - 0.5) < 1e-12) {
    return {0.5830300199018228, 1.3341669931033755};
  }

  std::vector<double> xs;
  std::vector<double> ys;
  xs.reserve(300);
  ys.reserve(300);
  for (int i = 0; i < 300; ++i) {
    const double x = (spread * 3.0) * static_cast<double>(i) / 299.0;
    xs.push_back(x);
    ys.push_back(x < min_dist ? 1.0 : std::exp(-(x - min_dist) / spread));
  }

  double best_a = 1.5769434601962196;
  double best_b = 0.8950608781227859;
  double best_loss = std::numeric_limits<double>::infinity();

  for (double loga = -4.0; loga <= 4.0001; loga += 0.2) {
    for (double b = 0.25; b <= 2.0001; b += 0.05) {
      const double a = std::exp(loga);
      double loss = 0.0;
      for (std::size_t i = 0; i < xs.size(); ++i) {
        const double x2b = std::pow(xs[i], 2.0 * b);
        const double yhat = 1.0 / (1.0 + a * x2b);
        const double e = yhat - ys[i];
        loss += e * e;
      }
      if (loss < best_loss) {
        best_loss = loss;
        best_a = a;
        best_b = b;
      }
    }
  }

  for (int iter = 0; iter < 80; ++iter) {
    double ga = 0.0;
    double gb = 0.0;
    for (std::size_t i = 0; i < xs.size(); ++i) {
      const double x = std::max(xs[i], 1e-6);
      const double x2b = std::pow(x, 2.0 * best_b);
      const double denom = 1.0 + best_a * x2b;
      const double yhat = 1.0 / denom;
      const double e = yhat - ys[i];
      ga += e * (-x2b / (denom * denom));
      gb += e * (-(best_a * x2b * 2.0 * std::log(x)) / (denom * denom));
    }
    best_a = std::max(1e-4, best_a - 0.01 * ga);
    best_b = std::max(0.1, best_b - 0.01 * gb);
  }

  return {best_a, best_b};
}

class ReusableBarrier {
 public:
  explicit ReusableBarrier(const int participants)
      : threshold_(participants), count_(participants), generation_(0) {}

  void wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    const int generation = generation_;
    if (--count_ == 0) {
      ++generation_;
      count_ = threshold_;
      cv_.notify_all();
      return;
    }
    cv_.wait(lock, [&]() { return generation != generation_; });
  }

 private:
  const int threshold_;
  int count_;
  int generation_;
  std::mutex mutex_;
  std::condition_variable cv_;
};

int effective_cpu_threads(const int n_threads, const int n_items) {
  if (n_items <= 1) return 1;
  return std::max(1, std::min(std::min(n_threads, 4), n_items));
}

class ParallelExecutor {
 public:
  explicit ParallelExecutor(const int max_threads)
      : max_threads_(std::max(1, max_threads)) {
    workers_.reserve(static_cast<std::size_t>(max_threads_ - 1));
    for (int thread_id = 1; thread_id < max_threads_; ++thread_id) {
      workers_.emplace_back([this, thread_id]() { worker_loop(thread_id); });
    }
  }

  ParallelExecutor(const ParallelExecutor&) = delete;
  ParallelExecutor& operator=(const ParallelExecutor&) = delete;

  ~ParallelExecutor() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
      ++generation_;
    }
    start_cv_.notify_all();
    for (std::thread& worker : workers_) worker.join();
  }

  int max_threads() const {
    return max_threads_;
  }

  template <typename Function>
  void run(const int n, const int requested_threads, Function& fn) {
    const int active_threads = std::max(
      1,
      std::min(std::min(requested_threads, max_threads_), std::max(1, n))
    );
    if (active_threads <= 1 || n < 2) {
      fn(0, n, 0);
      return;
    }

    const int chunk = (n + active_threads - 1) / active_threads;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      task_ = [&fn](const int begin, const int end, const int thread_id) {
        fn(begin, end, thread_id);
      };
      n_ = n;
      active_threads_ = active_threads;
      chunk_ = chunk;
      completed_workers_ = 0;
      worker_error_ = nullptr;
      ++generation_;
    }
    start_cv_.notify_all();

    std::exception_ptr main_error;
    try {
      fn(0, std::min(n, chunk), 0);
    } catch (...) {
      main_error = std::current_exception();
    }

    std::exception_ptr worker_error;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      done_cv_.wait(lock, [this]() {
        return completed_workers_ == static_cast<int>(workers_.size());
      });
      worker_error = worker_error_;
      task_ = nullptr;
    }
    if (main_error != nullptr) std::rethrow_exception(main_error);
    if (worker_error != nullptr) std::rethrow_exception(worker_error);
  }

 private:
  void worker_loop(const int thread_id) {
    std::uint64_t observed_generation = 0;
    while (true) {
      std::function<void(int, int, int)> task;
      int begin = 0;
      int end = 0;
      bool active = false;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        start_cv_.wait(lock, [this, observed_generation]() {
          return stop_ || generation_ != observed_generation;
        });
        if (stop_) return;
        observed_generation = generation_;
        task = task_;
        active = thread_id < active_threads_;
        if (active) {
          begin = thread_id * chunk_;
          end = std::min(n_, begin + chunk_);
          active = begin < end;
        }
      }

      if (active) {
        try {
          task(begin, end, thread_id);
        } catch (...) {
          std::lock_guard<std::mutex> lock(mutex_);
          if (worker_error_ == nullptr) worker_error_ = std::current_exception();
        }
      }

      {
        std::lock_guard<std::mutex> lock(mutex_);
        ++completed_workers_;
        if (completed_workers_ == static_cast<int>(workers_.size())) {
          done_cv_.notify_one();
        }
      }
    }
  }

  int max_threads_ = 1;
  std::vector<std::thread> workers_;
  std::mutex mutex_;
  std::condition_variable start_cv_;
  std::condition_variable done_cv_;
  std::function<void(int, int, int)> task_;
  std::uint64_t generation_ = 0;
  int n_ = 0;
  int active_threads_ = 1;
  int chunk_ = 0;
  int completed_workers_ = 0;
  std::exception_ptr worker_error_;
  bool stop_ = false;
};

thread_local ParallelExecutor* active_parallel_executor = nullptr;

class ParallelExecutorScope {
 public:
  explicit ParallelExecutorScope(ParallelExecutor* executor)
      : previous_(active_parallel_executor) {
    active_parallel_executor = executor;
  }

  ~ParallelExecutorScope() {
    active_parallel_executor = previous_;
  }

 private:
  ParallelExecutor* previous_;
};

template <typename Worker>
void parallel_for_chunks(const int n_items, const int n_threads, Worker worker) {
  const int threads = effective_cpu_threads(n_threads, n_items);
  if (threads == 1) {
    worker(0, n_items, 0);
    return;
  }
  if (active_parallel_executor != nullptr &&
      threads <= active_parallel_executor->max_threads()) {
    active_parallel_executor->run(n_items, threads, worker);
    return;
  }
  std::vector<std::thread> workers;
  workers.reserve(static_cast<std::size_t>(threads - 1));
  for (int t = 1; t < threads; ++t) {
    const int begin = static_cast<int>(
      static_cast<std::int64_t>(n_items) * static_cast<std::int64_t>(t) / threads
    );
    const int end = static_cast<int>(
      static_cast<std::int64_t>(n_items) * static_cast<std::int64_t>(t + 1) / threads
    );
    workers.emplace_back([&, begin, end, t]() { worker(begin, end, t); });
  }
  const int first_end = static_cast<int>(static_cast<std::int64_t>(n_items) / threads);
  worker(0, first_end, 0);
  for (auto& thread : workers) thread.join();
}

double clip_value(const double x, const double lo, const double hi) {
  return std::max(lo, std::min(hi, x));
}

float clip4f(const float x) {
  return x < -4.0f ? -4.0f : (x > 4.0f ? 4.0f : x);
}

void atomic_add_float(std::atomic<float>& target, const float value) {
  float expected = target.load(std::memory_order_relaxed);
  while (!target.compare_exchange_weak(
    expected,
    expected + value,
    std::memory_order_relaxed,
    std::memory_order_relaxed
  )) {}
}

// Copied/adapted from tkcaccia/fastEmbedR (MIT) for standalone kodama-cpp.
double umap_pow(const double x, const double b) {
  if (x <= 0.0) return 0.0;
  std::uint64_t x_bits = 0;
  std::memcpy(&x_bits, &x, sizeof(double));
  constexpr double exponent_bias_word = 1072632447.0;
  const int whole = static_cast<int>(b);
  const double fractional = b - static_cast<double>(whole);
  const double high_word = static_cast<double>(static_cast<std::uint32_t>(x_bits >> 32));
  const std::uint64_t interp_high = static_cast<std::uint64_t>(
    fractional * (high_word - exponent_bias_word) + exponent_bias_word
  );
  const std::uint64_t interp_bits = (interp_high & 0xffffffffull) << 32;
  double fractional_pow = 0.0;
  std::memcpy(&fractional_pow, &interp_bits, sizeof(double));
  double integer_pow = 1.0;
  double base = x;
  int exponent = whole;
  while (exponent > 0) {
    if ((exponent & 1) != 0) integer_pow *= base;
    base *= base;
    exponent >>= 1;
  }
  return integer_pow * fractional_pow;
}

float umap_powf_fast(const float x, const float b) {
  if (x <= 0.0f) return 0.0f;
  std::uint32_t x_bits = 0;
  std::memcpy(&x_bits, &x, sizeof(float));
  constexpr float exponent_bias_word = 1064866805.0f;
  const int whole = static_cast<int>(b);
  const float fractional = b - static_cast<float>(whole);
  const std::uint32_t interp_bits = static_cast<std::uint32_t>(
    fractional * (static_cast<float>(x_bits) - exponent_bias_word) + exponent_bias_word
  );
  float fractional_pow = 0.0f;
  std::memcpy(&fractional_pow, &interp_bits, sizeof(float));
  float integer_pow = 1.0f;
  float base = x;
  int exponent = whole;
  while (exponent > 0) {
    if ((exponent & 1) != 0) integer_pow *= base;
    base *= base;
    exponent >>= 1;
  }
  return integer_pow * fractional_pow;
}

std::uint32_t mix_seed(std::uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

int deterministic_vertex(
  const int n,
  const int seed,
  const int epoch,
  const std::size_t edge,
  const int sample
) {
  std::uint32_t x = static_cast<std::uint32_t>(seed);
  x ^= static_cast<std::uint32_t>(epoch * 0x9e3779b9u);
  x ^= static_cast<std::uint32_t>((edge + 1u) * 0x85ebca6bu);
  x ^= static_cast<std::uint32_t>((sample + 1) * 0xc2b2ae35u);
  return static_cast<int>(mix_seed(x) % static_cast<std::uint32_t>(n));
}

struct TauPrng {
  std::uint32_t state;

  TauPrng(const std::uint64_t s0, const std::uint64_t s1, const std::uint64_t s2)
      : state(mix_seed(
          static_cast<std::uint32_t>(s0) ^
          static_cast<std::uint32_t>(s1 << 7) ^
          static_cast<std::uint32_t>(s2 << 17)
        )) {
    if (state == 0u) state = 0x9e3779b9u;
  }

  std::uint32_t next() {
    std::uint32_t x = state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    state = x == 0u ? 0x9e3779b9u : x;
    return state;
  }

  int vertex(const int n) {
    return static_cast<int>(next() % static_cast<std::uint32_t>(n));
  }
};

TauPrng make_tau_prng(
  const int seed,
  const int epoch,
  const std::size_t window_end,
  const int thread_id
) {
  const std::uint32_t base = mix_seed(
    static_cast<std::uint32_t>(seed) ^
    static_cast<std::uint32_t>((epoch + 1) * 0x9e3779b9u) ^
    static_cast<std::uint32_t>((thread_id + 1) * 0x85ebca6bu) ^
    static_cast<std::uint32_t>((window_end + 1u) * 0xc2b2ae35u)
  );
  const std::uint32_t s0 = mix_seed(base ^ 0xa341316cu);
  const std::uint32_t s1 = mix_seed(base ^ 0xc8013ea4u);
  const std::uint32_t s2 = mix_seed(base ^ 0xad90777du);
  return TauPrng(s0, s1, s2);
}

std::vector<float> make_opentsne_random_init(const int n, const int n_components, const int seed) {
  std::vector<float> init(static_cast<std::size_t>(n) * static_cast<std::size_t>(n_components), 0.0f);
  std::mt19937 rng(static_cast<std::uint32_t>(seed));
  std::normal_distribution<float> normal(0.0f, 1.0e-4f);
  for (int c = 0; c < n_components; ++c) {
    double mean = 0.0;
    for (int i = 0; i < n; ++i) {
      const float value = normal(rng);
      init[static_cast<std::size_t>(i) * static_cast<std::size_t>(n_components) +
           static_cast<std::size_t>(c)] = value;
      mean += static_cast<double>(value);
    }
    mean /= static_cast<double>(n);
    for (int i = 0; i < n; ++i) {
      init[static_cast<std::size_t>(i) * static_cast<std::size_t>(n_components) +
           static_cast<std::size_t>(c)] -= static_cast<float>(mean);
    }
  }
  return init;
}

struct PreparedGraph {
  std::vector<int> indices;
  std::vector<float> distances;
  int samples = 0;
  int neighbors = 0;
  int index_offset = 0;
};

struct CsrGraph {
  std::vector<int> offsets;
  std::vector<int> neighbors;
  std::vector<float> weights;
  std::vector<float> epochs_per_sample;
  float max_weight = 1.0f;
};

PreparedGraph prepare_graph(const NeighborGraph& graph, int requested_neighbors) {
  if (graph.neighbors <= 0) {
    throw std::invalid_argument("NeighborGraph.neighbors must be positive.");
  }
  if (graph.indices.size() != graph.distances.size() ||
      graph.indices.size() % static_cast<std::size_t>(graph.neighbors) != 0) {
    throw std::invalid_argument("NeighborGraph indices/distances have inconsistent sizes.");
  }
  PreparedGraph out;
  out.samples = static_cast<int>(graph.indices.size() / static_cast<std::size_t>(graph.neighbors));
  out.neighbors = std::min(graph.neighbors, requested_neighbors > 0 ? requested_neighbors : graph.neighbors);
  if (out.samples < 2 || out.neighbors < 1) {
    throw std::invalid_argument("NeighborGraph must contain at least two samples and one neighbor.");
  }
  out.indices.assign(static_cast<std::size_t>(out.samples) * out.neighbors, 0);
  out.distances.assign(static_cast<std::size_t>(out.samples) * out.neighbors, 0.0f);

  int min_index = std::numeric_limits<int>::max();
  int max_index = std::numeric_limits<int>::min();
  for (int value : graph.indices) {
    if (value < 0) continue;
    min_index = std::min(min_index, value);
    max_index = std::max(max_index, value);
  }
  const bool one_based = graph.index_base == GraphIndexBase::One ||
    (graph.index_base == GraphIndexBase::Auto && min_index >= 1 && max_index <= out.samples);

  for (int i = 0; i < out.samples; ++i) {
    float row_max = 1.0f;
    for (int j = 0; j < graph.neighbors; ++j) {
      const float d = graph.distances[static_cast<std::size_t>(i) * graph.neighbors + j];
      if (std::isfinite(d) && d > row_max) row_max = d;
    }
    for (int j = 0; j < out.neighbors; ++j) {
      const std::size_t src = static_cast<std::size_t>(i) * graph.neighbors + j;
      const std::size_t dst = static_cast<std::size_t>(j) * out.samples + i;
      int index = graph.indices[src];
      if (one_based) --index;
      index = std::max(0, std::min(out.samples - 1, index));
      float distance = graph.distances[src];
      if (!std::isfinite(distance) || distance < 0.0f) distance = row_max;
      out.indices[dst] = index;
      out.distances[dst] = distance;
    }
  }
  return out;
}

void smooth_knn_dist(
  const PreparedGraph& graph,
  std::vector<float>& sigmas,
  std::vector<float>& rhos,
  const int n_threads
) {
  const int n = graph.samples;
  const int k = graph.neighbors;
  const double target = std::log2(static_cast<double>(k));
  const double tol = 1.0e-5;
  const double min_k_dist_scale = 1.0e-3;
  sigmas.assign(static_cast<std::size_t>(n), 1.0f);
  rhos.assign(static_cast<std::size_t>(n), 0.0f);

  const int threads = effective_cpu_threads(n_threads, n);
  std::vector<long double> thread_sums(static_cast<std::size_t>(threads), 0.0L);
  std::vector<std::size_t> thread_counts(static_cast<std::size_t>(threads), 0u);
  auto global_worker = [&](const int begin, const int end, const int thread_id) {
    long double local_sum = 0.0L;
    std::size_t local_count = 0u;
    for (int row = begin; row < end; ++row) {
      for (int col = 0; col < k; ++col) {
        const float distance =
          graph.distances[static_cast<std::size_t>(col) * n + row];
        if (std::isfinite(distance) && distance >= 0.0f) {
          local_sum += static_cast<long double>(distance);
          ++local_count;
        }
      }
    }
    thread_sums[static_cast<std::size_t>(thread_id)] = local_sum;
    thread_counts[static_cast<std::size_t>(thread_id)] = local_count;
  };
  if (threads == 1 || n < 2048) {
    global_worker(0, n, 0);
  } else {
    parallel_for_chunks(n, threads, global_worker);
  }

  long double global_sum = 0.0L;
  std::size_t global_count = 0u;
  for (int thread_id = 0; thread_id < threads; ++thread_id) {
    global_sum += thread_sums[static_cast<std::size_t>(thread_id)];
    global_count += thread_counts[static_cast<std::size_t>(thread_id)];
  }
  const double global_mean = global_count > 0u ?
    static_cast<double>(global_sum / static_cast<long double>(global_count)) :
    1.0;

  auto membership_sum = [&](const int row, const double rho, const double sigma) {
    double sum = 0.0;
    const double safe_sigma = std::max(sigma, 1.0e-12);
    for (int col = 0; col < k; ++col) {
      const float raw = graph.distances[static_cast<std::size_t>(col) * n + row];
      if (!std::isfinite(raw)) continue;
      const double distance = static_cast<double>(raw) - rho;
      sum += distance <= 0.0 ? 1.0 : std::exp(-distance / safe_sigma);
    }
    return sum;
  };

  auto worker = [&](const int begin, const int end, const int) {
    for (int row = begin; row < end; ++row) {
      double rho = std::numeric_limits<double>::infinity();
      double row_sum = 0.0;
      int row_count = 0;
      for (int col = 0; col < k; ++col) {
        const float distance =
          graph.distances[static_cast<std::size_t>(col) * n + row];
        if (!std::isfinite(distance)) continue;
        if (distance >= 0.0f) {
          row_sum += static_cast<double>(distance);
          ++row_count;
        }
        if (distance > 0.0f && static_cast<double>(distance) < rho) {
          rho = static_cast<double>(distance);
        }
      }
      if (!std::isfinite(rho)) rho = 0.0;
      rhos[static_cast<std::size_t>(row)] = static_cast<float>(rho);

      constexpr double sigma_max = (std::numeric_limits<double>::max)();
      double sigma = 1.0;
      double best_sigma = sigma;
      double best_diff = sigma_max;
      double lo = 0.0;
      double hi = sigma_max;
      for (int iter = 0; iter < 64; ++iter) {
        const double sum = membership_sum(row, rho, sigma);
        const double diff = std::abs(sum - target);
        if (diff < best_diff) {
          best_diff = diff;
          best_sigma = sigma;
        }
        if (sum > target) {
          hi = sigma;
          sigma = 0.5 * (lo + hi);
        } else {
          lo = sigma;
          sigma = hi == sigma_max ? sigma * 2.0 : 0.5 * (lo + hi);
        }
        if (diff < tol) break;
      }

      const double row_mean = row_count > 0 ?
        row_sum / static_cast<double>(row_count) :
        global_mean;
      const double sigma_floor =
        min_k_dist_scale * (rho > 0.0 ? row_mean : global_mean);
      best_sigma = std::max(best_sigma, sigma_floor);
      sigmas[static_cast<std::size_t>(row)] =
        static_cast<float>(std::max(best_sigma, 1.0e-12));
    }
  };
  if (threads == 1 || n < 2048) {
    worker(0, n, 0);
  } else {
    parallel_for_chunks(n, threads, worker);
  }
}

void attach_umap_epoch_schedule(CsrGraph& graph) {
  graph.max_weight = 0.0f;
  for (const float weight : graph.weights) {
    if (std::isfinite(weight) && weight > graph.max_weight) {
      graph.max_weight = weight;
    }
  }
  graph.epochs_per_sample.resize(graph.weights.size());
  if (graph.max_weight <= 0.0f) {
    std::fill(
      graph.epochs_per_sample.begin(),
      graph.epochs_per_sample.end(),
      std::numeric_limits<float>::infinity()
    );
    return;
  }
  for (std::size_t i = 0; i < graph.weights.size(); ++i) {
    graph.epochs_per_sample[i] =
      graph.max_weight / std::max(graph.weights[i], 1.0e-6f);
  }
}

int compact_umap_directed_rows(
  const std::vector<int>& raw_offsets,
  const std::vector<int>& row_ends,
  std::vector<int>& neighbors,
  std::vector<float>* weights,
  std::vector<int>& offsets,
  const int n_threads
) {
  const int n = static_cast<int>(raw_offsets.size()) - 1;
  offsets.assign(static_cast<std::size_t>(n) + 1u, 0);
  std::vector<int> counts(static_cast<std::size_t>(n), 0);

  parallel_for_chunks(n, n_threads, [&](const int begin_row, const int end_row, const int) {
    for (int row = begin_row; row < end_row; ++row) {
      const int begin = raw_offsets[static_cast<std::size_t>(row)];
      const int end = row_ends[static_cast<std::size_t>(row)];
      for (int pos = begin + 1; pos < end; ++pos) {
        const int neighbor = neighbors[static_cast<std::size_t>(pos)];
        const float weight = weights == nullptr ? 1.0f : (*weights)[static_cast<std::size_t>(pos)];
        int cursor = pos;
        while (cursor > begin &&
               neighbors[static_cast<std::size_t>(cursor - 1)] > neighbor) {
          neighbors[static_cast<std::size_t>(cursor)] =
            neighbors[static_cast<std::size_t>(cursor - 1)];
          if (weights != nullptr) {
            (*weights)[static_cast<std::size_t>(cursor)] =
              (*weights)[static_cast<std::size_t>(cursor - 1)];
          }
          --cursor;
        }
        neighbors[static_cast<std::size_t>(cursor)] = neighbor;
        if (weights != nullptr) (*weights)[static_cast<std::size_t>(cursor)] = weight;
      }

      int count = 0;
      int read = begin;
      while (read < end) {
        const int neighbor = neighbors[static_cast<std::size_t>(read++)];
        while (read < end && neighbors[static_cast<std::size_t>(read)] == neighbor) {
          ++read;
        }
        ++count;
      }
      counts[static_cast<std::size_t>(row)] = count;
    }
  });

  for (int row = 0; row < n; ++row) {
    offsets[static_cast<std::size_t>(row + 1)] =
      offsets[static_cast<std::size_t>(row)] + counts[static_cast<std::size_t>(row)];
  }
  const int compact_size = offsets[static_cast<std::size_t>(n)];
  std::vector<int> compact_neighbors(static_cast<std::size_t>(compact_size));
  std::vector<float> compact_weights;
  if (weights != nullptr) compact_weights.resize(static_cast<std::size_t>(compact_size));

  parallel_for_chunks(n, n_threads, [&](const int begin_row, const int end_row, const int) {
    for (int row = begin_row; row < end_row; ++row) {
      int write = offsets[static_cast<std::size_t>(row)];
      int read = raw_offsets[static_cast<std::size_t>(row)];
      const int end = row_ends[static_cast<std::size_t>(row)];
      while (read < end) {
        const int neighbor = neighbors[static_cast<std::size_t>(read)];
        float weight = weights == nullptr ? 1.0f : (*weights)[static_cast<std::size_t>(read)];
        ++read;
        while (read < end && neighbors[static_cast<std::size_t>(read)] == neighbor) {
          if (weights != nullptr) {
            weight = std::max(weight, (*weights)[static_cast<std::size_t>(read)]);
          }
          ++read;
        }
        compact_neighbors[static_cast<std::size_t>(write)] = neighbor;
        if (weights != nullptr) compact_weights[static_cast<std::size_t>(write)] = weight;
        ++write;
      }
    }
  });

  neighbors.swap(compact_neighbors);
  if (weights != nullptr) weights->swap(compact_weights);
  return compact_size;
}

CsrGraph build_umap_csr_graph(const PreparedGraph& graph, const int n_threads = 1) {
  const int n = graph.samples;
  const int k = graph.neighbors;
  std::vector<float> sigmas;
  std::vector<float> rhos;
  smooth_knn_dist(graph, sigmas, rhos, n_threads);

  std::vector<int> raw_counts(static_cast<std::size_t>(n), 0);
  parallel_for_chunks(n, n_threads, [&](const int begin_row, const int end_row, const int) {
    for (int row = begin_row; row < end_row; ++row) {
      int count = 0;
      for (int col = 0; col < k; ++col) {
        const int neighbor = graph.indices[static_cast<std::size_t>(col) * n + row];
        const float distance = graph.distances[static_cast<std::size_t>(col) * n + row];
        if (neighbor >= 0 && neighbor < n && neighbor != row && std::isfinite(distance)) {
          ++count;
        }
      }
      raw_counts[static_cast<std::size_t>(row)] = count;
    }
  });

  std::vector<int> raw_offsets(static_cast<std::size_t>(n) + 1u, 0);
  for (int row = 0; row < n; ++row) {
    raw_offsets[static_cast<std::size_t>(row + 1)] =
      raw_offsets[static_cast<std::size_t>(row)] + raw_counts[static_cast<std::size_t>(row)];
  }
  if (raw_offsets[static_cast<std::size_t>(n)] == 0) {
    throw std::runtime_error("The fuzzy UMAP graph has no usable edges.");
  }

  std::vector<int> directed_neighbors(
    static_cast<std::size_t>(raw_offsets[static_cast<std::size_t>(n)])
  );
  std::vector<float> directed_weights(directed_neighbors.size());
  std::vector<int> row_ends = raw_offsets;
  parallel_for_chunks(n, n_threads, [&](const int begin_row, const int end_row, const int) {
    for (int row = begin_row; row < end_row; ++row) {
      int write = raw_offsets[static_cast<std::size_t>(row)];
      const float rho = rhos[static_cast<std::size_t>(row)];
      const float sigma = sigmas[static_cast<std::size_t>(row)];
      for (int col = 0; col < k; ++col) {
        const int neighbor = graph.indices[static_cast<std::size_t>(col) * n + row];
        const float distance = graph.distances[static_cast<std::size_t>(col) * n + row];
        if (neighbor < 0 || neighbor >= n || neighbor == row || !std::isfinite(distance)) {
          continue;
        }
        const float weight = distance <= rho ? 1.0f : std::exp(-(distance - rho) / sigma);
        if (!std::isfinite(weight) || weight <= 0.0f) continue;
        directed_neighbors[static_cast<std::size_t>(write)] = neighbor;
        directed_weights[static_cast<std::size_t>(write)] = weight;
        ++write;
      }
      row_ends[static_cast<std::size_t>(row)] = write;
    }
  });

  std::vector<int> directed_offsets;
  const int directed_nnz = compact_umap_directed_rows(
    raw_offsets,
    row_ends,
    directed_neighbors,
    &directed_weights,
    directed_offsets,
    n_threads
  );
  sigmas.clear();
  sigmas.shrink_to_fit();
  rhos.clear();
  rhos.shrink_to_fit();

  const int graph_threads = effective_cpu_threads(n_threads, n);
  std::vector<std::vector<int>> incoming_cursor(
    static_cast<std::size_t>(graph_threads),
    std::vector<int>(static_cast<std::size_t>(n), 0)
  );
  parallel_for_chunks(n, graph_threads, [&](const int begin_row, const int end_row, const int thread_id) {
    auto& counts = incoming_cursor[static_cast<std::size_t>(thread_id)];
    for (int row = begin_row; row < end_row; ++row) {
      for (int pos = directed_offsets[static_cast<std::size_t>(row)];
           pos < directed_offsets[static_cast<std::size_t>(row + 1)]; ++pos) {
        ++counts[static_cast<std::size_t>(directed_neighbors[static_cast<std::size_t>(pos)])];
      }
    }
  });

  std::vector<int> incoming_offsets(static_cast<std::size_t>(n) + 1u, 0);
  for (int row = 0; row < n; ++row) {
    int count = 0;
    for (int thread_id = 0; thread_id < graph_threads; ++thread_id) {
      count += incoming_cursor[static_cast<std::size_t>(thread_id)][static_cast<std::size_t>(row)];
    }
    incoming_offsets[static_cast<std::size_t>(row + 1)] =
      incoming_offsets[static_cast<std::size_t>(row)] + count;
  }
  for (int row = 0; row < n; ++row) {
    int start = incoming_offsets[static_cast<std::size_t>(row)];
    for (int thread_id = 0; thread_id < graph_threads; ++thread_id) {
      int& cursor = incoming_cursor[static_cast<std::size_t>(thread_id)][static_cast<std::size_t>(row)];
      const int count = cursor;
      cursor = start;
      start += count;
    }
  }

  std::vector<int> incoming_neighbors(static_cast<std::size_t>(directed_nnz));
  std::vector<float> incoming_weights(static_cast<std::size_t>(directed_nnz));
  parallel_for_chunks(n, graph_threads, [&](const int begin_row, const int end_row, const int thread_id) {
    auto& cursor = incoming_cursor[static_cast<std::size_t>(thread_id)];
    for (int row = begin_row; row < end_row; ++row) {
      for (int pos = directed_offsets[static_cast<std::size_t>(row)];
           pos < directed_offsets[static_cast<std::size_t>(row + 1)]; ++pos) {
        const int target = directed_neighbors[static_cast<std::size_t>(pos)];
        const int output = cursor[static_cast<std::size_t>(target)]++;
        incoming_neighbors[static_cast<std::size_t>(output)] = row;
        incoming_weights[static_cast<std::size_t>(output)] =
          directed_weights[static_cast<std::size_t>(pos)];
      }
    }
  });
  incoming_cursor.clear();
  incoming_cursor.shrink_to_fit();

  auto merge_row = [&](const int row, const bool write_output, int& output, CsrGraph& result) {
    int forward_pos = directed_offsets[static_cast<std::size_t>(row)];
    const int forward_end = directed_offsets[static_cast<std::size_t>(row + 1)];
    int reverse_pos = incoming_offsets[static_cast<std::size_t>(row)];
    const int reverse_end = incoming_offsets[static_cast<std::size_t>(row + 1)];
    int count = 0;
    while (forward_pos < forward_end || reverse_pos < reverse_end) {
      const int forward_neighbor = forward_pos < forward_end ?
        directed_neighbors[static_cast<std::size_t>(forward_pos)] :
        std::numeric_limits<int>::max();
      const int reverse_neighbor = reverse_pos < reverse_end ?
        incoming_neighbors[static_cast<std::size_t>(reverse_pos)] :
        std::numeric_limits<int>::max();
      int neighbor = 0;
      float forward_weight = 0.0f;
      float reverse_weight = 0.0f;
      if (forward_neighbor == reverse_neighbor) {
        neighbor = forward_neighbor;
        forward_weight = directed_weights[static_cast<std::size_t>(forward_pos++)];
        reverse_weight = incoming_weights[static_cast<std::size_t>(reverse_pos++)];
      } else if (forward_neighbor < reverse_neighbor) {
        neighbor = forward_neighbor;
        forward_weight = directed_weights[static_cast<std::size_t>(forward_pos++)];
      } else {
        neighbor = reverse_neighbor;
        reverse_weight = incoming_weights[static_cast<std::size_t>(reverse_pos++)];
      }
      if (neighbor == row) continue;
      const float weight =
        forward_weight + reverse_weight - forward_weight * reverse_weight;
      if (!std::isfinite(weight) || weight <= 1.0e-6f) continue;
      if (write_output) {
        result.neighbors[static_cast<std::size_t>(output)] = neighbor;
        result.weights[static_cast<std::size_t>(output)] = weight;
        ++output;
      }
      ++count;
    }
    return count;
  };

  std::fill(raw_counts.begin(), raw_counts.end(), 0);
  CsrGraph unused;
  parallel_for_chunks(n, n_threads, [&](const int begin_row, const int end_row, const int) {
    for (int row = begin_row; row < end_row; ++row) {
      int output = 0;
      raw_counts[static_cast<std::size_t>(row)] = merge_row(row, false, output, unused);
    }
  });

  CsrGraph result;
  result.offsets.resize(static_cast<std::size_t>(n) + 1u, 0);
  for (int row = 0; row < n; ++row) {
    result.offsets[static_cast<std::size_t>(row + 1)] =
      result.offsets[static_cast<std::size_t>(row)] + raw_counts[static_cast<std::size_t>(row)];
  }
  const int nnz = result.offsets[static_cast<std::size_t>(n)];
  if (nnz == 0) throw std::runtime_error("The fuzzy UMAP graph has no usable edges.");
  result.neighbors.resize(static_cast<std::size_t>(nnz));
  result.weights.resize(static_cast<std::size_t>(nnz));
  parallel_for_chunks(n, n_threads, [&](const int begin_row, const int end_row, const int) {
    for (int row = begin_row; row < end_row; ++row) {
      int output = result.offsets[static_cast<std::size_t>(row)];
      merge_row(row, true, output, result);
    }
  });
  attach_umap_epoch_schedule(result);
  return result;
}

CsrGraph build_umap_binary_csr_graph(
  const PreparedGraph& graph,
  const int n_threads
) {
  const int n = graph.samples;
  const int k = graph.neighbors;
  std::vector<int> raw_counts(static_cast<std::size_t>(n), 0);
  parallel_for_chunks(n, n_threads, [&](const int begin_row, const int end_row, const int) {
    for (int row = begin_row; row < end_row; ++row) {
      int count = 0;
      for (int col = 0; col < k; ++col) {
        const int neighbor = graph.indices[static_cast<std::size_t>(col) * n + row];
        if (neighbor >= 0 && neighbor < n && neighbor != row) ++count;
      }
      raw_counts[static_cast<std::size_t>(row)] = count;
    }
  });

  std::vector<int> raw_offsets(static_cast<std::size_t>(n) + 1u, 0);
  for (int row = 0; row < n; ++row) {
    raw_offsets[static_cast<std::size_t>(row + 1)] =
      raw_offsets[static_cast<std::size_t>(row)] + raw_counts[static_cast<std::size_t>(row)];
  }
  if (raw_offsets[static_cast<std::size_t>(n)] == 0) {
    throw std::runtime_error("The binary UMAP graph has no usable edges.");
  }

  std::vector<int> directed_neighbors(
    static_cast<std::size_t>(raw_offsets[static_cast<std::size_t>(n)])
  );
  std::vector<int> row_ends = raw_offsets;
  parallel_for_chunks(n, n_threads, [&](const int begin_row, const int end_row, const int) {
    for (int row = begin_row; row < end_row; ++row) {
      int write = raw_offsets[static_cast<std::size_t>(row)];
      for (int col = 0; col < k; ++col) {
        const int neighbor = graph.indices[static_cast<std::size_t>(col) * n + row];
        if (neighbor < 0 || neighbor >= n || neighbor == row) continue;
        directed_neighbors[static_cast<std::size_t>(write++)] = neighbor;
      }
      row_ends[static_cast<std::size_t>(row)] = write;
    }
  });

  std::vector<int> directed_offsets;
  const int directed_nnz = compact_umap_directed_rows(
    raw_offsets,
    row_ends,
    directed_neighbors,
    nullptr,
    directed_offsets,
    n_threads
  );

  const int graph_threads = effective_cpu_threads(n_threads, n);
  std::vector<std::vector<int>> incoming_cursor(
    static_cast<std::size_t>(graph_threads),
    std::vector<int>(static_cast<std::size_t>(n), 0)
  );
  parallel_for_chunks(n, graph_threads, [&](const int begin_row, const int end_row, const int thread_id) {
    auto& counts = incoming_cursor[static_cast<std::size_t>(thread_id)];
    for (int row = begin_row; row < end_row; ++row) {
      for (int pos = directed_offsets[static_cast<std::size_t>(row)];
           pos < directed_offsets[static_cast<std::size_t>(row + 1)]; ++pos) {
        ++counts[static_cast<std::size_t>(directed_neighbors[static_cast<std::size_t>(pos)])];
      }
    }
  });

  std::vector<int> incoming_offsets(static_cast<std::size_t>(n) + 1u, 0);
  for (int row = 0; row < n; ++row) {
    int count = 0;
    for (int thread_id = 0; thread_id < graph_threads; ++thread_id) {
      count += incoming_cursor[static_cast<std::size_t>(thread_id)][static_cast<std::size_t>(row)];
    }
    incoming_offsets[static_cast<std::size_t>(row + 1)] =
      incoming_offsets[static_cast<std::size_t>(row)] + count;
  }
  for (int row = 0; row < n; ++row) {
    int start = incoming_offsets[static_cast<std::size_t>(row)];
    for (int thread_id = 0; thread_id < graph_threads; ++thread_id) {
      int& cursor = incoming_cursor[static_cast<std::size_t>(thread_id)][static_cast<std::size_t>(row)];
      const int count = cursor;
      cursor = start;
      start += count;
    }
  }

  std::vector<int> incoming_neighbors(static_cast<std::size_t>(directed_nnz));
  parallel_for_chunks(n, graph_threads, [&](const int begin_row, const int end_row, const int thread_id) {
    auto& cursor = incoming_cursor[static_cast<std::size_t>(thread_id)];
    for (int row = begin_row; row < end_row; ++row) {
      for (int pos = directed_offsets[static_cast<std::size_t>(row)];
           pos < directed_offsets[static_cast<std::size_t>(row + 1)]; ++pos) {
        const int target = directed_neighbors[static_cast<std::size_t>(pos)];
        incoming_neighbors[static_cast<std::size_t>(cursor[static_cast<std::size_t>(target)]++)] = row;
      }
    }
  });

  auto merge_count = [&](const int row) {
    int forward_pos = directed_offsets[static_cast<std::size_t>(row)];
    const int forward_end = directed_offsets[static_cast<std::size_t>(row + 1)];
    int reverse_pos = incoming_offsets[static_cast<std::size_t>(row)];
    const int reverse_end = incoming_offsets[static_cast<std::size_t>(row + 1)];
    int count = 0;
    while (forward_pos < forward_end || reverse_pos < reverse_end) {
      const int forward_neighbor = forward_pos < forward_end ?
        directed_neighbors[static_cast<std::size_t>(forward_pos)] :
        std::numeric_limits<int>::max();
      const int reverse_neighbor = reverse_pos < reverse_end ?
        incoming_neighbors[static_cast<std::size_t>(reverse_pos)] :
        std::numeric_limits<int>::max();
      if (forward_neighbor == reverse_neighbor) {
        ++forward_pos;
        ++reverse_pos;
      } else if (forward_neighbor < reverse_neighbor) {
        ++forward_pos;
      } else {
        ++reverse_pos;
      }
      ++count;
    }
    return count;
  };

  std::fill(raw_counts.begin(), raw_counts.end(), 0);
  parallel_for_chunks(n, n_threads, [&](const int begin_row, const int end_row, const int) {
    for (int row = begin_row; row < end_row; ++row) {
      raw_counts[static_cast<std::size_t>(row)] = merge_count(row);
    }
  });

  CsrGraph result;
  result.offsets.resize(static_cast<std::size_t>(n) + 1u, 0);
  for (int row = 0; row < n; ++row) {
    result.offsets[static_cast<std::size_t>(row + 1)] =
      result.offsets[static_cast<std::size_t>(row)] + raw_counts[static_cast<std::size_t>(row)];
  }
  const int nnz = result.offsets[static_cast<std::size_t>(n)];
  result.neighbors.resize(static_cast<std::size_t>(nnz));
  parallel_for_chunks(n, n_threads, [&](const int begin_row, const int end_row, const int) {
    for (int row = begin_row; row < end_row; ++row) {
      int forward_pos = directed_offsets[static_cast<std::size_t>(row)];
      const int forward_end = directed_offsets[static_cast<std::size_t>(row + 1)];
      int reverse_pos = incoming_offsets[static_cast<std::size_t>(row)];
      const int reverse_end = incoming_offsets[static_cast<std::size_t>(row + 1)];
      int write = result.offsets[static_cast<std::size_t>(row)];
      while (forward_pos < forward_end || reverse_pos < reverse_end) {
        const int forward_neighbor = forward_pos < forward_end ?
          directed_neighbors[static_cast<std::size_t>(forward_pos)] :
          std::numeric_limits<int>::max();
        const int reverse_neighbor = reverse_pos < reverse_end ?
          incoming_neighbors[static_cast<std::size_t>(reverse_pos)] :
          std::numeric_limits<int>::max();
        if (forward_neighbor == reverse_neighbor) {
          result.neighbors[static_cast<std::size_t>(write++)] = forward_neighbor;
          ++forward_pos;
          ++reverse_pos;
        } else if (forward_neighbor < reverse_neighbor) {
          result.neighbors[static_cast<std::size_t>(write++)] = forward_neighbor;
          ++forward_pos;
        } else {
          result.neighbors[static_cast<std::size_t>(write++)] = reverse_neighbor;
          ++reverse_pos;
        }
      }
    }
  });
  result.weights.assign(static_cast<std::size_t>(nnz), 1.0f);
  result.epochs_per_sample.assign(static_cast<std::size_t>(nnz), 1.0f);
  result.max_weight = 1.0f;
  return result;
}

CsrGraph build_selected_umap_graph(
  const PreparedGraph& graph,
  const UMAPOptions& options
) {
  return options.graph_mode == UMAPGraphMode::Binary ?
    build_umap_binary_csr_graph(graph, options.n_threads) :
    build_umap_csr_graph(graph, options.n_threads);
}

void scale_embedding_max_abs_and_jitter(std::vector<double>& embedding, double scale, double jitter, std::mt19937& rng) {
  double max_abs = 0.0;
  for (double v : embedding) max_abs = std::max(max_abs, std::abs(v));
  const double factor = max_abs > 0.0 ? scale / max_abs : 1.0;
  std::normal_distribution<double> noise(0.0, jitter);
  for (double& v : embedding) v = v * factor + noise(rng);
}

std::vector<float> initialize_layout_csr_spectral(
  const CsrGraph& graph,
  int n,
  int n_components,
  int spectral_n_iter,
  int seed
) {
  const int nnz = graph.offsets[static_cast<std::size_t>(n)];
  std::mt19937 rng(static_cast<std::uint32_t>(seed));
  std::normal_distribution<double> normal(0.0, 0.0001);
  std::uniform_real_distribution<double> uniform(-10.0, 10.0);
  std::vector<double> embedding(static_cast<std::size_t>(n) * n_components, 0.0);
  if (nnz == 0) {
    for (double& v : embedding) v = uniform(rng);
    std::vector<float> out(embedding.size());
    for (std::size_t i = 0; i < out.size(); ++i) out[i] = static_cast<float>(embedding[i]);
    return out;
  }

  std::vector<int> spectral_rows;
  std::vector<int> spectral_pos;
  spectral_rows.reserve(static_cast<std::size_t>(nnz / 2));
  spectral_pos.reserve(static_cast<std::size_t>(nnz / 2));
  for (int row = 0; row < n; ++row) {
    for (int pos = graph.offsets[static_cast<std::size_t>(row)];
         pos < graph.offsets[static_cast<std::size_t>(row + 1)]; ++pos) {
      const int nb = graph.neighbors[static_cast<std::size_t>(pos)];
      if (row < nb) {
        spectral_rows.push_back(row);
        spectral_pos.push_back(pos);
      }
    }
  }
  if (spectral_pos.empty()) {
    for (int row = 0; row < n; ++row) {
      for (int pos = graph.offsets[static_cast<std::size_t>(row)];
           pos < graph.offsets[static_cast<std::size_t>(row + 1)]; ++pos) {
        spectral_rows.push_back(row);
        spectral_pos.push_back(pos);
      }
    }
  }

  std::vector<float> degree(static_cast<std::size_t>(n), 0.0f);
  for (int row = 0; row < n; ++row) {
    double sum = 0.0;
    for (int pos = graph.offsets[static_cast<std::size_t>(row)];
         pos < graph.offsets[static_cast<std::size_t>(row + 1)]; ++pos) {
      sum += graph.weights[static_cast<std::size_t>(pos)];
    }
    degree[static_cast<std::size_t>(row)] = static_cast<float>(sum);
  }

  std::vector<std::vector<std::pair<int, float>>> spectral(static_cast<std::size_t>(n));
  for (std::size_t e = 0; e < spectral_pos.size(); ++e) {
    const int head = spectral_rows[e];
    const int pos = spectral_pos[e];
    const int tail = graph.neighbors[static_cast<std::size_t>(pos)];
    if (head < 0 || head >= n || tail < 0 || tail >= n || head == tail) continue;
    const double denom = std::sqrt(std::max(
      static_cast<double>(degree[static_cast<std::size_t>(head)]) *
      static_cast<double>(degree[static_cast<std::size_t>(tail)]),
      1e-24
    ));
    const float w = static_cast<float>(graph.weights[static_cast<std::size_t>(pos)] / denom);
    spectral[static_cast<std::size_t>(head)].push_back({tail, w});
    spectral[static_cast<std::size_t>(tail)].push_back({head, w});
  }

  std::vector<double> trivial(static_cast<std::size_t>(n), 0.0);
  double trivial_norm = 0.0;
  for (int i = 0; i < n; ++i) {
    trivial[static_cast<std::size_t>(i)] = std::sqrt(std::max(static_cast<double>(degree[static_cast<std::size_t>(i)]), 0.0));
    trivial_norm += trivial[static_cast<std::size_t>(i)] * trivial[static_cast<std::size_t>(i)];
  }
  trivial_norm = std::sqrt(std::max(trivial_norm, 1e-12));
  for (double& v : trivial) v /= trivial_norm;

  const int block_size = std::min(n - 1, std::max(n_components + 6, 8));
  if (block_size < n_components) {
    for (double& v : embedding) v = uniform(rng);
    std::vector<float> out(embedding.size());
    for (std::size_t i = 0; i < out.size(); ++i) out[i] = static_cast<float>(embedding[i]);
    return out;
  }

  auto at = [n](std::vector<double>& x, int row, int col) -> double& {
    return x[static_cast<std::size_t>(col) * n + row];
  };
  auto cat = [n](const std::vector<double>& x, int row, int col) -> double {
    return x[static_cast<std::size_t>(col) * n + row];
  };
  auto apply = [&](const std::vector<double>& x, std::vector<double>& y, int cols) {
    std::fill(y.begin(), y.end(), 0.0);
    for (int row = 0; row < n; ++row) {
      for (const auto& edge : spectral[static_cast<std::size_t>(row)]) {
        const int tail = edge.first;
        const double w = edge.second;
        for (int c = 0; c < cols; ++c) at(y, row, c) += w * cat(x, tail, c);
      }
    }
  };
  auto orthonormalize = [&](std::vector<double>& q, int cols) {
    for (int c = 0; c < cols; ++c) {
      double dot0 = 0.0;
      for (int i = 0; i < n; ++i) dot0 += cat(q, i, c) * trivial[static_cast<std::size_t>(i)];
      for (int i = 0; i < n; ++i) at(q, i, c) -= dot0 * trivial[static_cast<std::size_t>(i)];
      for (int prev = 0; prev < c; ++prev) {
        double dot = 0.0;
        for (int i = 0; i < n; ++i) dot += cat(q, i, c) * cat(q, i, prev);
        for (int i = 0; i < n; ++i) at(q, i, c) -= dot * cat(q, i, prev);
      }
      double norm = 0.0;
      for (int i = 0; i < n; ++i) norm += cat(q, i, c) * cat(q, i, c);
      norm = std::sqrt(norm);
      if (norm < 1e-10) {
        for (int i = 0; i < n; ++i) at(q, i, c) = normal(rng) * 10000.0;
        --c;
      } else {
        for (int i = 0; i < n; ++i) at(q, i, c) /= norm;
      }
    }
  };

  std::vector<double> q(static_cast<std::size_t>(n) * block_size);
  std::vector<double> z(static_cast<std::size_t>(n) * block_size);
  for (double& v : q) v = normal(rng) * 10000.0;
  orthonormalize(q, block_size);
  for (int iter = 0; iter < spectral_n_iter; ++iter) {
    apply(q, z, block_size);
    orthonormalize(z, block_size);
    q.swap(z);
  }

  apply(q, z, block_size);
  std::vector<double> projected(static_cast<std::size_t>(block_size) * block_size, 0.0);
  for (int row = 0; row < n; ++row) {
    for (int a = 0; a < block_size; ++a) {
      const double qv = cat(q, row, a);
      for (int b = a; b < block_size; ++b) {
        projected[static_cast<std::size_t>(a) * block_size + b] += qv * cat(z, row, b);
      }
    }
  }
  for (int a = 0; a < block_size; ++a) {
    for (int b = a + 1; b < block_size; ++b) projected[static_cast<std::size_t>(b) * block_size + a] = projected[static_cast<std::size_t>(a) * block_size + b];
  }

  std::vector<double> eigvec(static_cast<std::size_t>(block_size) * block_size, 0.0);
  for (int i = 0; i < block_size; ++i) eigvec[static_cast<std::size_t>(i) * block_size + i] = 1.0;
  for (int sweep = 0; sweep < 80; ++sweep) {
    int p = 0;
    int qidx = 1;
    double max_offdiag = 0.0;
    for (int i = 0; i < block_size; ++i) {
      for (int j = i + 1; j < block_size; ++j) {
        const double v = std::abs(projected[static_cast<std::size_t>(i) * block_size + j]);
        if (v > max_offdiag) {
          max_offdiag = v;
          p = i;
          qidx = j;
        }
      }
    }
    if (max_offdiag < 1e-10) break;
    const double app = projected[static_cast<std::size_t>(p) * block_size + p];
    const double aqq = projected[static_cast<std::size_t>(qidx) * block_size + qidx];
    const double apq = projected[static_cast<std::size_t>(p) * block_size + qidx];
    const double tau = (aqq - app) / (2.0 * apq);
    const double t = (tau >= 0.0 ? 1.0 : -1.0) / (std::abs(tau) + std::sqrt(1.0 + tau * tau));
    const double cs = 1.0 / std::sqrt(1.0 + t * t);
    const double sn = t * cs;
    for (int k = 0; k < block_size; ++k) {
      if (k == p || k == qidx) continue;
      const double akp = projected[static_cast<std::size_t>(k) * block_size + p];
      const double akq = projected[static_cast<std::size_t>(k) * block_size + qidx];
      projected[static_cast<std::size_t>(k) * block_size + p] = cs * akp - sn * akq;
      projected[static_cast<std::size_t>(p) * block_size + k] = projected[static_cast<std::size_t>(k) * block_size + p];
      projected[static_cast<std::size_t>(k) * block_size + qidx] = sn * akp + cs * akq;
      projected[static_cast<std::size_t>(qidx) * block_size + k] = projected[static_cast<std::size_t>(k) * block_size + qidx];
    }
    projected[static_cast<std::size_t>(p) * block_size + p] = cs * cs * app - 2.0 * sn * cs * apq + sn * sn * aqq;
    projected[static_cast<std::size_t>(qidx) * block_size + qidx] = sn * sn * app + 2.0 * sn * cs * apq + cs * cs * aqq;
    projected[static_cast<std::size_t>(p) * block_size + qidx] = 0.0;
    projected[static_cast<std::size_t>(qidx) * block_size + p] = 0.0;
    for (int k = 0; k < block_size; ++k) {
      const double vip = eigvec[static_cast<std::size_t>(k) * block_size + p];
      const double viq = eigvec[static_cast<std::size_t>(k) * block_size + qidx];
      eigvec[static_cast<std::size_t>(k) * block_size + p] = cs * vip - sn * viq;
      eigvec[static_cast<std::size_t>(k) * block_size + qidx] = sn * vip + cs * viq;
    }
  }

  std::vector<int> order(block_size);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    return projected[static_cast<std::size_t>(a) * block_size + a] >
           projected[static_cast<std::size_t>(b) * block_size + b];
  });
  for (int c = 0; c < n_components; ++c) {
    const int ritz_col = order[static_cast<std::size_t>(c)];
    std::vector<double> values(static_cast<std::size_t>(n), 0.0);
    for (int i = 0; i < n; ++i) {
      double value = 0.0;
      for (int basis = 0; basis < block_size; ++basis) {
        value += cat(q, i, basis) * eigvec[static_cast<std::size_t>(basis) * block_size + ritz_col];
      }
      values[static_cast<std::size_t>(i)] = value;
    }
    const double mean = std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(n);
    for (int i = 0; i < n; ++i) embedding[static_cast<std::size_t>(i) * n_components + c] = values[static_cast<std::size_t>(i)] - mean;
  }
  scale_embedding_max_abs_and_jitter(embedding, 10.0, 1.0e-4, rng);

  std::vector<float> out(embedding.size());
  for (std::size_t i = 0; i < out.size(); ++i) out[i] = static_cast<float>(embedding[i]);
  return out;
}

void add_delta_2d(
  std::vector<float>& delta,
  const int i,
  const double dx,
  const double dy,
  const double scale,
  const bool update_other,
  const int other
) {
  const double gx = clip_value(scale * dx, -4.0, 4.0);
  const double gy = clip_value(scale * dy, -4.0, 4.0);
  delta[static_cast<std::size_t>(i) * 2u] += static_cast<float>(gx);
  delta[static_cast<std::size_t>(i) * 2u + 1u] += static_cast<float>(gy);
  if (update_other) {
    delta[static_cast<std::size_t>(other) * 2u] -= static_cast<float>(gx);
    delta[static_cast<std::size_t>(other) * 2u + 1u] -= static_cast<float>(gy);
  }
}

std::vector<float> optimize_umap_csr_cpu(
  const CsrGraph& graph,
  const UMAPOptions& options,
  const int n
) {
  if (options.n_components != 2) {
    throw std::invalid_argument("CPU UMAP currently supports n_components=2.");
  }
  const int nnz = graph.offsets[static_cast<std::size_t>(n)];
  if (nnz == 0) {
    throw std::runtime_error("CPU UMAP requires at least one graph edge.");
  }

  const double max_weight = static_cast<double>(graph.max_weight);
  if (!std::isfinite(max_weight) || max_weight <= 0.0) {
    throw std::runtime_error("CPU UMAP graph has no usable edge weight.");
  }
  const double min_sample_weight = options.n_epochs > 0 ?
    max_weight / static_cast<double>(options.n_epochs) :
    0.0;

  std::vector<int> active_rows;
  std::vector<int> active_pos;
  active_rows.reserve(static_cast<std::size_t>(nnz));
  active_pos.reserve(static_cast<std::size_t>(nnz));
  for (int row = 0; row < n; ++row) {
    for (int pos = graph.offsets[static_cast<std::size_t>(row)];
         pos < graph.offsets[static_cast<std::size_t>(row + 1)]; ++pos) {
      const int nb = graph.neighbors[static_cast<std::size_t>(pos)];
      const float w = graph.weights[static_cast<std::size_t>(pos)];
      if (nb >= 0 && nb < n && nb != row && std::isfinite(w) && w >= min_sample_weight) {
        active_rows.push_back(row);
        active_pos.push_back(pos);
      }
    }
  }
  if (active_pos.empty()) {
    throw std::runtime_error("CPU UMAP graph has no edges sampled by n_epochs.");
  }

  std::vector<int> active_tails(active_pos.size());
  std::vector<float> epochs_per_sample(active_pos.size());
  std::vector<float> epoch_of_next_sample(active_pos.size());
  std::vector<float> epochs_per_negative_sample(active_pos.size());
  std::vector<float> epoch_of_next_negative_sample(active_pos.size());
  for (std::size_t i = 0; i < active_pos.size(); ++i) {
    const int pos = active_pos[i];
    active_tails[i] = graph.neighbors[static_cast<std::size_t>(pos)];
    epochs_per_sample[i] = graph.epochs_per_sample[static_cast<std::size_t>(pos)];
    epoch_of_next_sample[i] = epochs_per_sample[i];
    if (options.negative_sample_rate > 0) {
      epochs_per_negative_sample[i] =
        epochs_per_sample[i] / static_cast<float>(options.negative_sample_rate);
      epoch_of_next_negative_sample[i] = epochs_per_negative_sample[i];
    } else {
      epochs_per_negative_sample[i] = std::numeric_limits<float>::infinity();
      epoch_of_next_negative_sample[i] = std::numeric_limits<float>::infinity();
    }
  }

  std::vector<float> embedding = options.init;
  if (embedding.size() != static_cast<std::size_t>(n) * 2u) {
    embedding = initialize_layout_csr_spectral(
      graph,
      n,
      2,
      options.spectral_n_iter,
      options.seed
    );
  }
  if (options.n_epochs == 0) return embedding;

  const auto ab = find_ab_params(1.0, options.min_dist);
  const float af = static_cast<float>(ab.first);
  const float bf = static_cast<float>(ab.second);
  const float attraction_const = static_cast<float>(-2.0 * ab.first * ab.second);
  const float repulsion_const =
    static_cast<float>(2.0 * options.repulsion_strength * ab.second);
  const float eps = std::numeric_limits<float>::epsilon();
  const int threads = effective_cpu_threads(options.n_threads, static_cast<int>(active_pos.size()));

  if (threads > 1 && n >= 10000) {
    std::vector<std::atomic<float>> x(static_cast<std::size_t>(n));
    std::vector<std::atomic<float>> y(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
      x[static_cast<std::size_t>(i)].store(
        embedding[static_cast<std::size_t>(i) * 2u], std::memory_order_relaxed
      );
      y[static_cast<std::size_t>(i)].store(
        embedding[static_cast<std::size_t>(i) * 2u + 1u], std::memory_order_relaxed
      );
    }

    ReusableBarrier barrier(threads);
    std::vector<std::size_t> worker_begin(static_cast<std::size_t>(threads) + 1u, 0u);
    worker_begin[0] = 0u;
    worker_begin[static_cast<std::size_t>(threads)] = active_pos.size();
    for (int t = 1; t < threads; ++t) {
      std::size_t boundary = active_pos.size() * static_cast<std::size_t>(t) /
                             static_cast<std::size_t>(threads);
      while (boundary < active_pos.size() &&
             boundary > 0 &&
             active_rows[boundary] == active_rows[boundary - 1u]) {
        ++boundary;
      }
      worker_begin[static_cast<std::size_t>(t)] = boundary;
    }

    const int* active_rows_ptr = active_rows.data();
    const int* active_tails_ptr = active_tails.data();
    const float* epochs_per_sample_ptr = epochs_per_sample.data();
    const float* epochs_per_negative_sample_ptr = epochs_per_negative_sample.data();
    const float alpha0 = static_cast<float>(options.learning_rate);
    const float alpha_step = alpha0 / static_cast<float>(options.n_epochs);

    auto run_worker = [&](const int t) {
      const std::size_t begin = worker_begin[static_cast<std::size_t>(t)];
      const std::size_t end = worker_begin[static_cast<std::size_t>(t + 1)];
      float alpha = alpha0;
      for (int epoch = 0; epoch < options.n_epochs; ++epoch) {
        TauPrng prng = make_tau_prng(options.seed, epoch, end, t);
        for (std::size_t edge = begin; edge < end; ++edge) {
          if (epoch_of_next_sample[edge] > epoch) continue;
          const int head = active_rows_ptr[edge];
          const int tail = active_tails_ptr[edge];
          const float head_x = x[static_cast<std::size_t>(head)].load(std::memory_order_relaxed);
          const float head_y = y[static_cast<std::size_t>(head)].load(std::memory_order_relaxed);
          const float tail_x = x[static_cast<std::size_t>(tail)].load(std::memory_order_relaxed);
          const float tail_y = y[static_cast<std::size_t>(tail)].load(std::memory_order_relaxed);
          const float dx = head_x - tail_x;
          const float dy = head_y - tail_y;
          const float dist_sq = std::max(eps, dx * dx + dy * dy);
          const float dist_pow = umap_powf_fast(dist_sq, bf);
          const float grad_coeff =
            attraction_const * dist_pow / (dist_sq * (af * dist_pow + 1.0f));
          const float gx = clip4f(grad_coeff * dx) * alpha;
          const float gy = clip4f(grad_coeff * dy) * alpha;
          atomic_add_float(x[static_cast<std::size_t>(head)], gx);
          atomic_add_float(y[static_cast<std::size_t>(head)], gy);
          atomic_add_float(x[static_cast<std::size_t>(tail)], -gx);
          atomic_add_float(y[static_cast<std::size_t>(tail)], -gy);
          epoch_of_next_sample[edge] += epochs_per_sample_ptr[edge];

          int n_neg_samples = 0;
          if (options.negative_sample_rate > 0 &&
              epoch >= epoch_of_next_negative_sample[edge]) {
            n_neg_samples = static_cast<int>(std::floor(
              (epoch - epoch_of_next_negative_sample[edge]) /
              epochs_per_negative_sample_ptr[edge]
            ));
            n_neg_samples = std::max(0, n_neg_samples);
          }
          for (int sample = 0; sample < n_neg_samples; ++sample) {
            const int neg = prng.vertex(n);
            if (neg == head) continue;
            const float current_head_x =
              x[static_cast<std::size_t>(head)].load(std::memory_order_relaxed);
            const float current_head_y =
              y[static_cast<std::size_t>(head)].load(std::memory_order_relaxed);
            const float neg_x = x[static_cast<std::size_t>(neg)].load(std::memory_order_relaxed);
            const float neg_y = y[static_cast<std::size_t>(neg)].load(std::memory_order_relaxed);
            const float ndx = current_head_x - neg_x;
            const float ndy = current_head_y - neg_y;
            const float neg_dist_sq = std::max(eps, ndx * ndx + ndy * ndy);
            const float neg_pow = umap_powf_fast(neg_dist_sq, bf);
            const float repulse =
              repulsion_const / ((0.001f + neg_dist_sq) * (af * neg_pow + 1.0f));
            atomic_add_float(
              x[static_cast<std::size_t>(head)], clip4f(repulse * ndx) * alpha
            );
            atomic_add_float(
              y[static_cast<std::size_t>(head)], clip4f(repulse * ndy) * alpha
            );
          }
          if (n_neg_samples > 0) {
            epoch_of_next_negative_sample[edge] +=
              n_neg_samples * epochs_per_negative_sample_ptr[edge];
          }
        }
        if ((epoch + 1) % 50 == 0 || epoch == options.n_epochs - 1) barrier.wait();
        alpha -= alpha_step;
      }
    };

    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(threads - 1));
    for (int t = 1; t < threads; ++t) workers.emplace_back(run_worker, t);
    run_worker(0);
    for (auto& worker : workers) worker.join();
    for (int i = 0; i < n; ++i) {
      embedding[static_cast<std::size_t>(i) * 2u] =
        x[static_cast<std::size_t>(i)].load(std::memory_order_relaxed);
      embedding[static_cast<std::size_t>(i) * 2u + 1u] =
        y[static_cast<std::size_t>(i)].load(std::memory_order_relaxed);
    }
    return embedding;
  }

  if (threads > 1) {
    const int sync_batches = n >= 3000 ?
      std::min(8, std::max(2, 2 * threads)) :
      std::min(4, threads);
    std::vector<std::vector<float>> deltas(
      static_cast<std::size_t>(threads),
      std::vector<float>(static_cast<std::size_t>(n) * 2u, 0.0f)
    );
    ReusableBarrier barrier(threads);
    auto run_worker = [&](const int t) {
      auto& delta = deltas[static_cast<std::size_t>(t)];
      for (int epoch = 0; epoch < options.n_epochs; ++epoch) {
        const double alpha =
          options.learning_rate * (1.0 - static_cast<double>(epoch) / options.n_epochs);
        for (int batch = 0; batch < sync_batches; ++batch) {
          std::fill(delta.begin(), delta.end(), 0.0f);
          const std::size_t batch_begin =
            active_pos.size() * static_cast<std::size_t>(batch) /
            static_cast<std::size_t>(sync_batches);
          const std::size_t batch_end =
            active_pos.size() * static_cast<std::size_t>(batch + 1) /
            static_cast<std::size_t>(sync_batches);
          const std::size_t batch_size = batch_end - batch_begin;
          const std::size_t begin =
            batch_begin + batch_size * static_cast<std::size_t>(t) /
            static_cast<std::size_t>(threads);
          const std::size_t end =
            batch_begin + batch_size * static_cast<std::size_t>(t + 1) /
            static_cast<std::size_t>(threads);
          for (std::size_t edge = begin; edge < end; ++edge) {
            if (epoch_of_next_sample[edge] > epoch) continue;
            const int head = active_rows[edge];
            const int tail = active_tails[edge];
            const std::size_t hb = static_cast<std::size_t>(head) * 2u;
            const std::size_t tb = static_cast<std::size_t>(tail) * 2u;
            const double dx = embedding[hb] - embedding[tb];
            const double dy = embedding[hb + 1u] - embedding[tb + 1u];
            const double dist_sq = dx * dx + dy * dy;
            double grad_coeff = 0.0;
            if (dist_sq > 0.0) {
              const double dist_pow = umap_pow(dist_sq, ab.second);
              grad_coeff = -2.0 * ab.first * ab.second * (dist_pow / dist_sq) /
                           (ab.first * dist_pow + 1.0);
            }
            add_delta_2d(delta, head, dx, dy, grad_coeff, true, tail);
            epoch_of_next_sample[edge] += epochs_per_sample[edge];

            int n_neg_samples = 0;
            if (options.negative_sample_rate > 0 &&
                epoch >= epoch_of_next_negative_sample[edge]) {
              n_neg_samples = static_cast<int>(std::floor(
                (epoch - epoch_of_next_negative_sample[edge]) /
                epochs_per_negative_sample[edge]
              ));
              n_neg_samples = std::max(0, n_neg_samples);
            }
            for (int sample = 0; sample < n_neg_samples; ++sample) {
              const int neg = deterministic_vertex(n, options.seed, epoch, edge, sample);
              if (neg == head) continue;
              const std::size_t nb = static_cast<std::size_t>(neg) * 2u;
              const double ndx = embedding[hb] - embedding[nb];
              const double ndy = embedding[hb + 1u] - embedding[nb + 1u];
              const double neg_dist_sq = ndx * ndx + ndy * ndy;
              double repulse = 0.0;
              if (neg_dist_sq > 0.0) {
                repulse = 2.0 * options.repulsion_strength * ab.second /
                          ((0.001 + neg_dist_sq) *
                           (ab.first * umap_pow(neg_dist_sq, ab.second) + 1.0));
              }
              add_delta_2d(delta, head, ndx, ndy, repulse, false, neg);
            }
            if (n_neg_samples > 0) {
              epoch_of_next_negative_sample[edge] +=
                n_neg_samples * epochs_per_negative_sample[edge];
            }
          }
          barrier.wait();
          if (t == 0) {
            for (int worker_id = 0; worker_id < threads; ++worker_id) {
              const auto& worker_delta = deltas[static_cast<std::size_t>(worker_id)];
              for (std::size_t i = 0; i < embedding.size(); ++i) {
                embedding[i] += static_cast<float>(alpha * worker_delta[i]);
              }
            }
          }
          barrier.wait();
        }
      }
    };
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(threads - 1));
    for (int t = 1; t < threads; ++t) workers.emplace_back(run_worker, t);
    run_worker(0);
    for (auto& worker : workers) worker.join();
    return embedding;
  }

  for (int epoch = 0; epoch < options.n_epochs; ++epoch) {
    const double alpha =
      options.learning_rate * (1.0 - static_cast<double>(epoch) / options.n_epochs);
    for (std::size_t edge = 0; edge < active_pos.size(); ++edge) {
      if (epoch_of_next_sample[edge] > epoch) continue;
      const int head = active_rows[edge];
      const int tail = active_tails[edge];
      const std::size_t hb = static_cast<std::size_t>(head) * 2u;
      const std::size_t tb = static_cast<std::size_t>(tail) * 2u;
      const double dx = embedding[hb] - embedding[tb];
      const double dy = embedding[hb + 1u] - embedding[tb + 1u];
      const double dist_sq = dx * dx + dy * dy;
      double grad_coeff = 0.0;
      if (dist_sq > 0.0) {
        const double dist_pow = umap_pow(dist_sq, ab.second);
        grad_coeff = -2.0 * ab.first * ab.second * (dist_pow / dist_sq) /
                     (ab.first * dist_pow + 1.0);
      }
      const double gx = clip_value(grad_coeff * dx, -4.0, 4.0);
      const double gy = clip_value(grad_coeff * dy, -4.0, 4.0);
      embedding[hb] += static_cast<float>(gx * alpha);
      embedding[hb + 1u] += static_cast<float>(gy * alpha);
      embedding[tb] -= static_cast<float>(gx * alpha);
      embedding[tb + 1u] -= static_cast<float>(gy * alpha);
      epoch_of_next_sample[edge] += epochs_per_sample[edge];

      int n_neg_samples = 0;
      if (options.negative_sample_rate > 0 && epoch >= epoch_of_next_negative_sample[edge]) {
        n_neg_samples = static_cast<int>(std::floor(
          (epoch - epoch_of_next_negative_sample[edge]) / epochs_per_negative_sample[edge]
        ));
        n_neg_samples = std::max(0, n_neg_samples);
      }
      for (int sample = 0; sample < n_neg_samples; ++sample) {
        const int neg = deterministic_vertex(n, options.seed, epoch, edge, sample);
        if (neg == head) continue;
        const std::size_t nb = static_cast<std::size_t>(neg) * 2u;
        const double ndx = embedding[hb] - embedding[nb];
        const double ndy = embedding[hb + 1u] - embedding[nb + 1u];
        const double neg_dist_sq = ndx * ndx + ndy * ndy;
        double repulse = 0.0;
        if (neg_dist_sq > 0.0) {
          repulse = 2.0 * options.repulsion_strength * ab.second /
                    ((0.001 + neg_dist_sq) *
                     (ab.first * umap_pow(neg_dist_sq, ab.second) + 1.0));
        }
        embedding[hb] += static_cast<float>(
          clip_value(repulse * ndx, -4.0, 4.0) * alpha
        );
        embedding[hb + 1u] += static_cast<float>(
          clip_value(repulse * ndy, -4.0, 4.0) * alpha
        );
      }
      if (n_neg_samples > 0) {
        epoch_of_next_negative_sample[edge] +=
          n_neg_samples * epochs_per_negative_sample[edge];
      }
    }
  }
  return embedding;
}

struct SparseProbabilitiesF {
  std::vector<int> row_ptr;
  std::vector<int> col;
  std::vector<float> val;
};

struct PackedEdgeF {
  std::uint64_t key;
  float value;
};

std::uint64_t pair_key(const int a, const int b) {
  return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(a)) << 32u) |
         static_cast<std::uint32_t>(b);
}

int key_first(const std::uint64_t key) {
  return static_cast<int>(key >> 32u);
}

int key_second(const std::uint64_t key) {
  return static_cast<int>(key & 0xffffffffu);
}

void compute_row_probabilities_float(
  const PreparedGraph& graph,
  const int row,
  const double perplexity,
  float* row_p
) {
  const int n = graph.samples;
  const int k = graph.neighbors;
  std::fill(row_p, row_p + k, 0.0f);
  bool found = false;
  float beta = 1.0f;
  float min_beta = -FLT_MAX;
  float max_beta = FLT_MAX;
  const float tol = 1.0e-5f;
  float sum_p = FLT_MIN;

  for (int iter = 0; !found && iter < 200; ++iter) {
    sum_p = FLT_MIN;
    for (int j = 0; j < k; ++j) {
      const float d = graph.distances[static_cast<std::size_t>(j) * n + row];
      const float d2 = d * d;
      const float p = std::exp(-beta * d2);
      row_p[j] = p;
      sum_p += p;
    }
    float entropy = 0.0f;
    for (int j = 0; j < k; ++j) {
      const float d = graph.distances[static_cast<std::size_t>(j) * n + row];
      entropy += beta * (d * d * row_p[j]);
    }
    entropy = entropy / sum_p + std::log(sum_p);
    const float diff = entropy - static_cast<float>(std::log(perplexity));
    if (std::abs(diff) < tol) {
      found = true;
    } else if (diff > 0.0f) {
      min_beta = beta;
      beta = (max_beta == FLT_MAX || max_beta == -FLT_MAX) ?
        beta * 2.0f :
        (beta + max_beta) * 0.5f;
    } else {
      max_beta = beta;
      beta = (min_beta == -FLT_MAX || min_beta == FLT_MAX) ?
        beta * 0.5f :
        (beta + min_beta) * 0.5f;
    }
  }

  const float inv_sum_p = 1.0f / sum_p;
  for (int j = 0; j < k; ++j) row_p[j] *= inv_sum_p;
}

SparseProbabilitiesF build_tsne_probabilities_float(
  const PreparedGraph& graph,
  const double perplexity,
  const int n_threads
) {
  const int n = graph.samples;
  const int k = graph.neighbors;
  for (int row = 0; row < n; ++row) {
    for (int j = 0; j < k; ++j) {
      const int nb = graph.indices[static_cast<std::size_t>(j) * n + row];
      const float d = graph.distances[static_cast<std::size_t>(j) * n + row];
      if (nb < 0 || nb >= n) throw std::invalid_argument("openTSNE KNN indices are out of range.");
      if (!std::isfinite(d) || d < 0.0f) {
        throw std::invalid_argument("openTSNE KNN distances must be finite and non-negative.");
      }
    }
  }

  const int threads = effective_cpu_threads(n_threads, n);
  std::vector<std::vector<PackedEdgeF>> local_edges(static_cast<std::size_t>(threads));
  parallel_for_chunks(n, threads, [&](const int begin, const int end, const int thread_id) {
    std::vector<float> row_p(static_cast<std::size_t>(k), 0.0f);
    auto& edges = local_edges[static_cast<std::size_t>(thread_id)];
    edges.reserve(edges.size() + static_cast<std::size_t>(std::max(0, end - begin)) * k);
    for (int row = begin; row < end; ++row) {
      compute_row_probabilities_float(graph, row, perplexity, row_p.data());
      for (int j = 0; j < k; ++j) {
        const int nb = graph.indices[static_cast<std::size_t>(j) * n + row];
        if (nb == row) continue;
        const int a = std::min(row, nb);
        const int b = std::max(row, nb);
        edges.push_back({pair_key(a, b), row_p[static_cast<std::size_t>(j)]});
      }
    }
  });

  std::size_t edge_count = 0;
  for (const auto& edges : local_edges) edge_count += edges.size();
  std::vector<PackedEdgeF> edges;
  edges.reserve(edge_count);
  for (auto& local : local_edges) {
    edges.insert(edges.end(), local.begin(), local.end());
    std::vector<PackedEdgeF>().swap(local);
  }
  std::sort(edges.begin(), edges.end(), [](const PackedEdgeF& a, const PackedEdgeF& b) {
    return a.key < b.key;
  });
  if (edges.empty()) {
    throw std::runtime_error("openTSNE KNN graph produced no non-self edges.");
  }

  SparseProbabilitiesF p;
  p.row_ptr.assign(static_cast<std::size_t>(n) + 1u, 0);
  std::size_t write = 0;
  double total_directed_mass = 0.0;
  for (std::size_t read = 0; read < edges.size();) {
    const std::uint64_t key = edges[read].key;
    double sum = 0.0;
    while (read < edges.size() && edges[read].key == key) {
      sum += static_cast<double>(edges[read].value);
      ++read;
    }
    edges[write++] = {key, static_cast<float>(sum)};
    total_directed_mass += sum;
    const int a = key_first(key);
    const int b = key_second(key);
    ++p.row_ptr[static_cast<std::size_t>(a + 1)];
    ++p.row_ptr[static_cast<std::size_t>(b + 1)];
  }
  edges.resize(write);

  if (!std::isfinite(total_directed_mass) || total_directed_mass <= 0.0) {
    throw std::runtime_error("openTSNE probability normalization failed.");
  }
  for (int i = 0; i < n; ++i) {
    p.row_ptr[static_cast<std::size_t>(i + 1)] += p.row_ptr[static_cast<std::size_t>(i)];
  }
  p.col.assign(static_cast<std::size_t>(p.row_ptr[static_cast<std::size_t>(n)]), 0);
  p.val.assign(p.col.size(), 0.0f);
  std::vector<int> fill = p.row_ptr;
  const float norm = static_cast<float>(0.5 / total_directed_mass);
  for (const PackedEdgeF& edge : edges) {
    const int a = key_first(edge.key);
    const int b = key_second(edge.key);
    const float value = edge.value * norm;
    int pos = fill[static_cast<std::size_t>(a)]++;
    p.col[static_cast<std::size_t>(pos)] = b;
    p.val[static_cast<std::size_t>(pos)] = value;
    pos = fill[static_cast<std::size_t>(b)]++;
    p.col[static_cast<std::size_t>(pos)] = a;
    p.val[static_cast<std::size_t>(pos)] = value;
  }
  return p;
}

void add_sparse_attractive_gradient_f(
  const SparseProbabilitiesF& p,
  const std::vector<float>& y,
  const int n,
  const int dims,
  const float exaggeration,
  const int n_threads,
  std::vector<float>& grad
) {
  parallel_for_chunks(n, n_threads, [&](const int begin, const int end, const int) {
    for (int i = begin; i < end; ++i) {
      const std::size_t ib = static_cast<std::size_t>(i) * dims;
      const int row_begin = p.row_ptr[static_cast<std::size_t>(i)];
      const int row_end = p.row_ptr[static_cast<std::size_t>(i + 1)];
      for (int pos = row_begin; pos < row_end; ++pos) {
        const int j = p.col[static_cast<std::size_t>(pos)];
        const std::size_t jb = static_cast<std::size_t>(j) * dims;
        float diff[3] = {0.0f, 0.0f, 0.0f};
        float d2 = 0.0f;
        for (int d = 0; d < dims; ++d) {
          diff[d] = y[ib + d] - y[jb + d];
          d2 += diff[d] * diff[d];
        }
        const float q = 1.0f / (1.0f + d2);
        const float coeff = exaggeration * p.val[static_cast<std::size_t>(pos)] * q;
        for (int d = 0; d < dims; ++d) grad[ib + d] += coeff * diff[d];
      }
    }
  });
}

void compute_gradient_pair_symmetric_f(
  const SparseProbabilitiesF& p,
  const std::vector<float>& y,
  const int n,
  const int dims,
  const float exaggeration,
  const int n_threads,
  std::vector<float>& grad
) {
  std::fill(grad.begin(), grad.end(), 0.0f);
  const int threads = effective_cpu_threads(n_threads, n);
  std::vector<std::vector<float>> local_grad(
    static_cast<std::size_t>(threads),
    std::vector<float>(grad.size(), 0.0f)
  );
  std::vector<double> partial_sum_q(static_cast<std::size_t>(threads), 0.0);
  auto repulsive_worker = [&](const int thread_id) {
    auto& g = local_grad[static_cast<std::size_t>(thread_id)];
    double local_sum_q = 0.0;
    for (int i = thread_id; i < n - 1; i += threads) {
      const std::size_t ib = static_cast<std::size_t>(i) * dims;
      for (int j = i + 1; j < n; ++j) {
        const std::size_t jb = static_cast<std::size_t>(j) * dims;
        float d2 = 0.0f;
        for (int d = 0; d < dims; ++d) {
          const float diff = y[ib + d] - y[jb + d];
          d2 += diff * diff;
        }
        const float q = 1.0f / (1.0f + d2);
        local_sum_q += 2.0 * static_cast<double>(q);
        const float coeff = -(q * q);
        for (int d = 0; d < dims; ++d) {
          const float step = coeff * (y[ib + d] - y[jb + d]);
          g[ib + d] += step;
          g[jb + d] -= step;
        }
      }
    }
    partial_sum_q[static_cast<std::size_t>(thread_id)] = local_sum_q;
  };
  if (threads <= 1) {
    repulsive_worker(0);
  } else {
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(threads - 1));
    for (int t = 1; t < threads; ++t) workers.emplace_back(repulsive_worker, t);
    repulsive_worker(0);
    for (auto& thread : workers) thread.join();
  }
  const float inv_sum_q = static_cast<float>(1.0 / std::max(
    std::accumulate(partial_sum_q.begin(), partial_sum_q.end(), 0.0),
    static_cast<double>(FLT_MIN)
  ));
  parallel_for_chunks(static_cast<int>(grad.size()), threads, [&](const int begin, const int end, const int) {
    for (int index = begin; index < end; ++index) {
      float value = 0.0f;
      for (int t = 0; t < threads; ++t) {
        value += local_grad[static_cast<std::size_t>(t)][static_cast<std::size_t>(index)];
      }
      grad[static_cast<std::size_t>(index)] = value * inv_sum_q;
    }
  });
  add_sparse_attractive_gradient_f(p, y, n, dims, exaggeration, threads, grad);
}

int env_positive_int(const char* name, const int fallback) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') return fallback;
  char* end = nullptr;
  const long parsed = std::strtol(raw, &end, 10);
  if (end == raw || parsed <= 0L || parsed > static_cast<long>(std::numeric_limits<int>::max())) {
    return fallback;
  }
  return static_cast<int>(parsed);
}

int tsne_fft_grid_size(const int n) {
  const int fallback = n >= 50000 ? 256 : (n >= 10000 ? 128 : 64);
  const int requested = env_positive_int("KODAMA_TSNE_FFT_GRID", fallback);
  int grid = 32;
  while (grid < requested && grid < 512) grid <<= 1;
  return std::max(32, std::min(512, grid));
}

template <typename T>
struct FftPlanT {
  int size = 0;
  int thread_capacity = 0;
  std::vector<int> bit_reverse;
  std::vector<std::complex<T>> forward_roots;
  std::vector<std::complex<T>> inverse_roots;
  std::vector<std::complex<T>> column_scratch;

  void ensure(const int requested_size, const int requested_threads) {
    const int threads = std::max(1, requested_threads);
    if (size == requested_size) {
      if (thread_capacity < threads) {
        thread_capacity = threads;
        column_scratch.resize(static_cast<std::size_t>(size) * thread_capacity);
      }
      return;
    }
    size = requested_size;
    thread_capacity = threads;
    bit_reverse.resize(static_cast<std::size_t>(size));
    int bits = 0;
    for (int value = size; value > 1; value >>= 1) ++bits;
    for (int i = 0; i < size; ++i) {
      unsigned int value = static_cast<unsigned int>(i);
      unsigned int reversed = 0u;
      for (int bit = 0; bit < bits; ++bit) {
        reversed = (reversed << 1u) | (value & 1u);
        value >>= 1u;
      }
      bit_reverse[static_cast<std::size_t>(i)] = static_cast<int>(reversed);
    }
    forward_roots.clear();
    inverse_roots.clear();
    forward_roots.reserve(static_cast<std::size_t>(bits));
    inverse_roots.reserve(static_cast<std::size_t>(bits));
    for (int len = 2; len <= size; len <<= 1) {
      const T angle = static_cast<T>(6.283185307179586476925286766559) /
        static_cast<T>(len);
      forward_roots.emplace_back(std::cos(angle), std::sin(angle));
      inverse_roots.emplace_back(std::cos(-angle), std::sin(-angle));
    }
    column_scratch.resize(static_cast<std::size_t>(size) * thread_capacity);
  }
};

template <typename T>
void fft_1d_t(
  std::complex<T>* a,
  const int n,
  const bool inverse,
  const FftPlanT<T>& plan
) {
  for (int i = 1; i < n; ++i) {
    const int j = plan.bit_reverse[static_cast<std::size_t>(i)];
    if (i < j) std::swap(a[i], a[j]);
  }

  const std::vector<std::complex<T>>& roots = inverse ?
    plan.inverse_roots :
    plan.forward_roots;
  int stage = 0;
  for (int len = 2; len <= n; len <<= 1, ++stage) {
    const std::complex<T> root = roots[static_cast<std::size_t>(stage)];
    for (int i = 0; i < n; i += len) {
      std::complex<T> w(static_cast<T>(1.0), static_cast<T>(0.0));
      const int half = len >> 1;
      for (int j = 0; j < half; ++j) {
        const std::complex<T> u = a[i + j];
        const std::complex<T> v = a[i + j + half] * w;
        a[i + j] = u + v;
        a[i + j + half] = u - v;
        w *= root;
      }
    }
  }
  if (inverse) {
    const T scale = static_cast<T>(1.0) / static_cast<T>(n);
    for (int i = 0; i < n; ++i) a[i] *= scale;
  }
}

template <typename T>
void fft_2d_t(
  std::vector<std::complex<T>>& values,
  const int size,
  const bool inverse,
  const int n_threads,
  FftPlanT<T>& plan
) {
  parallel_for_chunks(size, n_threads, [&](const int begin, const int end, const int) {
    for (int row = begin; row < end; ++row) {
      fft_1d_t<T>(
        values.data() + static_cast<std::size_t>(row) * size,
        size,
        inverse,
        plan
      );
    }
  });
  parallel_for_chunks(size, n_threads, [&](const int begin, const int end, const int thread_id) {
    std::complex<T>* column = plan.column_scratch.data() +
      static_cast<std::size_t>(thread_id) * size;
    for (int col = begin; col < end; ++col) {
      for (int row = 0; row < size; ++row) {
        column[row] =
          values[static_cast<std::size_t>(row) * size + col];
      }
      fft_1d_t<T>(column, size, inverse, plan);
      for (int row = 0; row < size; ++row) {
        values[static_cast<std::size_t>(row) * size + col] =
          column[row];
      }
    }
  });
}

template <typename T>
T bilinear_grid_value_t(const std::vector<T>& grid, const int grid_size, const T gx, const T gy) {
  const T cx = std::max(static_cast<T>(0.0), std::min(static_cast<T>(grid_size - 1), gx));
  const T cy = std::max(static_cast<T>(0.0), std::min(static_cast<T>(grid_size - 1), gy));
  const int x0 = std::max(0, std::min(grid_size - 2, static_cast<int>(std::floor(cx))));
  const int y0 = std::max(0, std::min(grid_size - 2, static_cast<int>(std::floor(cy))));
  const T tx = cx - static_cast<T>(x0);
  const T ty = cy - static_cast<T>(y0);
  const int x1 = x0 + 1;
  const int y1 = y0 + 1;
  const T v00 = grid[static_cast<std::size_t>(y0) * grid_size + x0];
  const T v10 = grid[static_cast<std::size_t>(y0) * grid_size + x1];
  const T v01 = grid[static_cast<std::size_t>(y1) * grid_size + x0];
  const T v11 = grid[static_cast<std::size_t>(y1) * grid_size + x1];
  return (static_cast<T>(1.0) - tx) * (static_cast<T>(1.0) - ty) * v00 +
         tx * (static_cast<T>(1.0) - ty) * v10 +
         (static_cast<T>(1.0) - tx) * ty * v01 +
         tx * ty * v11;
}

template <typename T>
struct FftGridWorkspaceT {
  int grid_size = 0;
  int fft_size = 0;
  int n = 0;
  int n_threads = 0;
  std::vector<T> mass;
  std::vector<T> mass_x;
  std::vector<T> mass_y;
  std::vector<T> gx;
  std::vector<T> gy;
  std::vector<T> q_grid;
  std::vector<T> q2_grid;
  std::vector<T> xq2_grid;
  std::vector<T> yq2_grid;
  std::vector<double> partial_sum_q;
  std::vector<std::complex<T>> mass_fft;
  std::vector<std::complex<T>> mass_x_fft;
  std::vector<std::complex<T>> mass_y_fft;
  std::vector<std::complex<T>> kernel_q;
  std::vector<std::complex<T>> kernel_q2;
  std::vector<std::complex<T>> work;
  FftPlanT<T> fft_plan;

  void ensure(const int requested_grid_size, const int requested_n, const int requested_threads) {
    grid_size = requested_grid_size;
    fft_size = grid_size << 1;
    n = requested_n;
    n_threads = std::max(1, requested_threads);
    const std::size_t grid_total = static_cast<std::size_t>(grid_size) * grid_size;
    const std::size_t fft_total = static_cast<std::size_t>(fft_size) * fft_size;
    mass.resize(grid_total);
    mass_x.resize(grid_total);
    mass_y.resize(grid_total);
    gx.resize(static_cast<std::size_t>(n));
    gy.resize(static_cast<std::size_t>(n));
    q_grid.resize(grid_total);
    q2_grid.resize(grid_total);
    xq2_grid.resize(grid_total);
    yq2_grid.resize(grid_total);
    partial_sum_q.resize(static_cast<std::size_t>(n_threads));
    mass_fft.resize(fft_total);
    mass_x_fft.resize(fft_total);
    mass_y_fft.resize(fft_total);
    kernel_q.resize(fft_total);
    kernel_q2.resize(fft_total);
    work.resize(fft_total);
    fft_plan.ensure(fft_size, n_threads);
  }

  void clear_grid_mass() {
    std::fill(mass.begin(), mass.end(), static_cast<T>(0.0));
    std::fill(mass_x.begin(), mass_x.end(), static_cast<T>(0.0));
    std::fill(mass_y.begin(), mass_y.end(), static_cast<T>(0.0));
  }
};

template <typename T>
void copy_grid_to_fft_t(
  const std::vector<T>& grid,
  const int grid_size,
  const int fft_size,
  std::vector<std::complex<T>>& out,
  const int n_threads
) {
  parallel_for_chunks(fft_size, n_threads, [&](const int begin, const int end, const int) {
    const std::complex<T> zero(static_cast<T>(0.0), static_cast<T>(0.0));
    for (int y_cell = begin; y_cell < end; ++y_cell) {
      std::complex<T>* output_row =
        out.data() + static_cast<std::size_t>(y_cell) * fft_size;
      std::fill(output_row, output_row + fft_size, zero);
      if (y_cell < grid_size) {
        const T* input_row =
          grid.data() + static_cast<std::size_t>(y_cell) * grid_size;
        for (int x_cell = 0; x_cell < grid_size; ++x_cell) {
          output_row[x_cell] = input_row[x_cell];
        }
      }
    }
  });
}

template <typename T>
void copy_fft_to_grid_t(
  const std::vector<std::complex<T>>& values,
  const int grid_size,
  const int fft_size,
  std::vector<T>& out,
  const int n_threads
) {
  out.resize(static_cast<std::size_t>(grid_size) * grid_size);
  parallel_for_chunks(grid_size, n_threads, [&](const int begin, const int end, const int) {
    for (int y_cell = begin; y_cell < end; ++y_cell) {
      for (int x_cell = 0; x_cell < grid_size; ++x_cell) {
        out[static_cast<std::size_t>(y_cell) * grid_size + x_cell] =
          values[static_cast<std::size_t>(y_cell) * fft_size + x_cell].real();
      }
    }
  });
}

template <typename T>
void compute_fft_grid_convolution_workspace_t(
  const std::vector<T>& mass,
  const std::vector<T>& mass_x,
  const std::vector<T>& mass_y,
  const int grid_size,
  const T spacing,
  const int n_threads,
  FftGridWorkspaceT<T>& ws
) {
  const int fft_size = grid_size << 1;
  const std::size_t fft_total = static_cast<std::size_t>(fft_size) * fft_size;
  copy_grid_to_fft_t<T>(mass, grid_size, fft_size, ws.mass_fft, n_threads);
  copy_grid_to_fft_t<T>(mass_x, grid_size, fft_size, ws.mass_x_fft, n_threads);
  copy_grid_to_fft_t<T>(mass_y, grid_size, fft_size, ws.mass_y_fft, n_threads);
  parallel_for_chunks(static_cast<int>(fft_total), n_threads, [&](const int begin, const int end, const int) {
    const std::complex<T> zero(static_cast<T>(0.0), static_cast<T>(0.0));
    std::fill(ws.kernel_q.begin() + begin, ws.kernel_q.begin() + end, zero);
    std::fill(ws.kernel_q2.begin() + begin, ws.kernel_q2.begin() + end, zero);
  });
  const int kernel_rows = 2 * grid_size - 1;
  parallel_for_chunks(kernel_rows, n_threads, [&](const int begin, const int end, const int) {
    for (int row = begin; row < end; ++row) {
      const int dy = row - (grid_size - 1);
      const int yy = dy < 0 ? dy + fft_size : dy;
      const T y_offset = static_cast<T>(dy) * spacing;
      for (int dx = -(grid_size - 1); dx <= grid_size - 1; ++dx) {
        const int xx = dx < 0 ? dx + fft_size : dx;
        const T x_offset = static_cast<T>(dx) * spacing;
        const T d2 = x_offset * x_offset + y_offset * y_offset;
        const T q = static_cast<T>(1.0) / (static_cast<T>(1.0) + d2);
        const T q2 = q * q;
        const std::size_t pos = static_cast<std::size_t>(yy) * fft_size + xx;
        ws.kernel_q[pos] = q;
        ws.kernel_q2[pos] = q2;
      }
    }
  });
  fft_2d_t<T>(ws.mass_fft, fft_size, false, n_threads, ws.fft_plan);
  fft_2d_t<T>(ws.mass_x_fft, fft_size, false, n_threads, ws.fft_plan);
  fft_2d_t<T>(ws.mass_y_fft, fft_size, false, n_threads, ws.fft_plan);
  fft_2d_t<T>(ws.kernel_q, fft_size, false, n_threads, ws.fft_plan);
  fft_2d_t<T>(ws.kernel_q2, fft_size, false, n_threads, ws.fft_plan);
  auto convolve = [&](const std::vector<std::complex<T>>& mass_values,
                      const std::vector<std::complex<T>>& kernel_values,
                      std::vector<T>& out) {
    parallel_for_chunks(static_cast<int>(fft_total), n_threads, [&](const int begin, const int end, const int) {
      for (int index = begin; index < end; ++index) {
        ws.work[static_cast<std::size_t>(index)] =
          mass_values[static_cast<std::size_t>(index)] *
          kernel_values[static_cast<std::size_t>(index)];
      }
    });
    fft_2d_t<T>(ws.work, fft_size, true, n_threads, ws.fft_plan);
    copy_fft_to_grid_t<T>(ws.work, grid_size, fft_size, out, n_threads);
  };
  convolve(ws.mass_fft, ws.kernel_q, ws.q_grid);
  convolve(ws.mass_fft, ws.kernel_q2, ws.q2_grid);
  convolve(ws.mass_x_fft, ws.kernel_q2, ws.xq2_grid);
  convolve(ws.mass_y_fft, ws.kernel_q2, ws.yq2_grid);
}

void compute_gradient_fft_grid_f(
  const SparseProbabilitiesF& p,
  const std::vector<float>& y,
  const int n,
  const int dims,
  const float exaggeration,
  const int n_threads,
  FftGridWorkspaceT<float>* workspace,
  std::vector<float>& grad
) {
  if (dims != 2) {
    compute_gradient_pair_symmetric_f(p, y, n, dims, exaggeration, n_threads, grad);
    return;
  }
  std::fill(grad.begin(), grad.end(), 0.0f);
  const int grid_size = tsne_fft_grid_size(n);
  float min_x = y[0], max_x = y[0], min_y = y[1], max_y = y[1];
  for (int i = 1; i < n; ++i) {
    const std::size_t base = static_cast<std::size_t>(i) * 2u;
    min_x = std::min(min_x, y[base]);
    max_x = std::max(max_x, y[base]);
    min_y = std::min(min_y, y[base + 1u]);
    max_y = std::max(max_y, y[base + 1u]);
  }
  const float cx = 0.5f * (min_x + max_x);
  const float cy = 0.5f * (min_y + max_y);
  float span = std::max(max_x - min_x, max_y - min_y);
  if (!std::isfinite(span) || span <= 0.0f) span = 1.0f;
  const float half = 0.55f * span + 1.0e-3f;
  const float lower_x = cx - half;
  const float lower_y = cy - half;
  const float spacing = (2.0f * half) / static_cast<float>(grid_size - 1);
  const float inv_spacing = 1.0f / spacing;

  FftGridWorkspaceT<float> local_workspace;
  FftGridWorkspaceT<float>& ws = workspace == nullptr ? local_workspace : *workspace;
  ws.ensure(grid_size, n, n_threads);
  ws.clear_grid_mass();
  for (int i = 0; i < n; ++i) {
    const std::size_t base = static_cast<std::size_t>(i) * 2u;
    const float x_coord = y[base];
    const float y_coord = y[base + 1u];
    const float raw_x = (x_coord - lower_x) * inv_spacing;
    const float raw_y = (y_coord - lower_y) * inv_spacing;
    const float clamped_x = std::max(0.0f, std::min(static_cast<float>(grid_size - 1), raw_x));
    const float clamped_y = std::max(0.0f, std::min(static_cast<float>(grid_size - 1), raw_y));
    const int x0 = std::max(0, std::min(grid_size - 2, static_cast<int>(std::floor(clamped_x))));
    const int y0 = std::max(0, std::min(grid_size - 2, static_cast<int>(std::floor(clamped_y))));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const float tx = clamped_x - static_cast<float>(x0);
    const float ty = clamped_y - static_cast<float>(y0);
    ws.gx[static_cast<std::size_t>(i)] = clamped_x;
    ws.gy[static_cast<std::size_t>(i)] = clamped_y;
    const float w00 = (1.0f - tx) * (1.0f - ty);
    const float w10 = tx * (1.0f - ty);
    const float w01 = (1.0f - tx) * ty;
    const float w11 = tx * ty;
    const std::size_t p00 = static_cast<std::size_t>(y0) * grid_size + x0;
    const std::size_t p10 = static_cast<std::size_t>(y0) * grid_size + x1;
    const std::size_t p01 = static_cast<std::size_t>(y1) * grid_size + x0;
    const std::size_t p11 = static_cast<std::size_t>(y1) * grid_size + x1;
    ws.mass[p00] += w00;
    ws.mass[p10] += w10;
    ws.mass[p01] += w01;
    ws.mass[p11] += w11;
    ws.mass_x[p00] += w00 * x_coord;
    ws.mass_x[p10] += w10 * x_coord;
    ws.mass_x[p01] += w01 * x_coord;
    ws.mass_x[p11] += w11 * x_coord;
    ws.mass_y[p00] += w00 * y_coord;
    ws.mass_y[p10] += w10 * y_coord;
    ws.mass_y[p01] += w01 * y_coord;
    ws.mass_y[p11] += w11 * y_coord;
  }

  compute_fft_grid_convolution_workspace_t<float>(
    ws.mass, ws.mass_x, ws.mass_y, grid_size, spacing, n_threads, ws
  );

  std::fill(ws.partial_sum_q.begin(), ws.partial_sum_q.end(), 0.0);
  parallel_for_chunks(n, n_threads, [&](const int begin, const int end, const int thread_id) {
    double local_sum_q = 0.0;
    for (int i = begin; i < end; ++i) {
      local_sum_q += static_cast<double>(bilinear_grid_value_t<float>(
        ws.q_grid, grid_size, ws.gx[static_cast<std::size_t>(i)], ws.gy[static_cast<std::size_t>(i)]
      ));
    }
    ws.partial_sum_q[static_cast<std::size_t>(thread_id)] = local_sum_q;
  });
  const float inv_sum_q = static_cast<float>(1.0 / std::max(
    std::accumulate(ws.partial_sum_q.begin(), ws.partial_sum_q.end(), 0.0) -
      static_cast<double>(n),
    static_cast<double>(FLT_MIN)
  ));
  parallel_for_chunks(n, n_threads, [&](const int begin, const int end, const int) {
    for (int i = begin; i < end; ++i) {
      const std::size_t base = static_cast<std::size_t>(i) * 2u;
      const float px = ws.gx[static_cast<std::size_t>(i)];
      const float py = ws.gy[static_cast<std::size_t>(i)];
      const float q2_value = bilinear_grid_value_t<float>(ws.q2_grid, grid_size, px, py);
      const float xq2_value = bilinear_grid_value_t<float>(ws.xq2_grid, grid_size, px, py);
      const float yq2_value = bilinear_grid_value_t<float>(ws.yq2_grid, grid_size, px, py);
      grad[base] = -(y[base] * q2_value - xq2_value) * inv_sum_q;
      grad[base + 1u] = -(y[base + 1u] * q2_value - yq2_value) * inv_sum_q;
    }
  });
  add_sparse_attractive_gradient_f(p, y, n, dims, exaggeration, n_threads, grad);
}

void compute_gradient_f(
  const SparseProbabilitiesF& p,
  const std::vector<float>& y,
  const int n,
  const int dims,
  const float exaggeration,
  const int n_threads,
  const bool use_fft,
  FftGridWorkspaceT<float>* fft_workspace,
  std::vector<float>& grad
) {
  if (use_fft) {
    compute_gradient_fft_grid_f(p, y, n, dims, exaggeration, n_threads, fft_workspace, grad);
  } else {
    compute_gradient_pair_symmetric_f(p, y, n, dims, exaggeration, n_threads, grad);
  }
}

void zero_mean_f(std::vector<float>& y, const int n, const int dims) {
  std::array<double, 3> mean{{0.0, 0.0, 0.0}};
  for (int i = 0; i < n; ++i) {
    const std::size_t base = static_cast<std::size_t>(i) * dims;
    for (int d = 0; d < dims; ++d) mean[static_cast<std::size_t>(d)] += y[base + d];
  }
  for (int d = 0; d < dims; ++d) mean[static_cast<std::size_t>(d)] /= static_cast<double>(n);
  for (int i = 0; i < n; ++i) {
    const std::size_t base = static_cast<std::size_t>(i) * dims;
    for (int d = 0; d < dims; ++d) y[base + d] -= static_cast<float>(mean[static_cast<std::size_t>(d)]);
  }
}

float sign_tsne_f(float x) {
  return x == 0.0f ? 0.0f : (x < 0.0f ? -1.0f : 1.0f);
}

void apply_open_tsne_update_f(
  std::vector<float>& y,
  std::vector<float>& update,
  std::vector<float>& gains,
  const std::vector<float>& grad,
  const int n,
  const int dims,
  const float learning_rate,
  const float momentum,
  const float min_gain,
  const float max_step_norm,
  const int n_threads
) {
  const bool clip_steps = std::isfinite(max_step_norm) && max_step_norm > 0.0f;
  const float max_step_norm_sq = max_step_norm * max_step_norm;
  parallel_for_chunks(n, n_threads, [&](const int begin, const int end, const int) {
    for (int i = begin; i < end; ++i) {
      const std::size_t base = static_cast<std::size_t>(i) * dims;
      float step_norm_sq = 0.0f;
      for (int d = 0; d < dims; ++d) {
        const std::size_t index = base + static_cast<std::size_t>(d);
        if (sign_tsne_f(update[index]) != sign_tsne_f(grad[index])) {
          gains[index] += 0.2f;
        } else {
          gains[index] = gains[index] * 0.8f + min_gain;
        }
        if (gains[index] < min_gain) gains[index] = min_gain;
        update[index] = momentum * update[index] - learning_rate * gains[index] * grad[index];
        step_norm_sq += update[index] * update[index];
      }
      float scale = 1.0f;
      if (clip_steps && step_norm_sq > max_step_norm_sq) {
        scale = max_step_norm / std::sqrt(std::max(step_norm_sq, FLT_MIN));
      }
      for (int d = 0; d < dims; ++d) {
        const std::size_t index = base + static_cast<std::size_t>(d);
        update[index] *= scale;
        y[index] += update[index];
      }
    }
  });
}

void validate_opentsne_options(
  const OpenTSNEOptions& options,
  const int samples
) {
  if (!std::isfinite(options.perplexity) || options.perplexity <= 0.0) {
    throw std::invalid_argument("openTSNE perplexity must be positive and finite.");
  }
  if (samples - 1 < 3.0 * options.perplexity) {
    throw std::invalid_argument("openTSNE perplexity is too large for the number of samples.");
  }
  if (options.theta < 0.0 || options.theta > 1.0) {
    throw std::invalid_argument("openTSNE theta must lie in [0, 1].");
  }
  if (options.early_exaggeration_iter < 0 || options.n_iter < 0 ||
      options.early_exaggeration_iter + options.n_iter < 1) {
    throw std::invalid_argument("openTSNE iteration counts must be non-negative and sum to at least one.");
  }
  if (!std::isfinite(options.early_exaggeration) || options.early_exaggeration <= 0.0 ||
      !std::isfinite(options.exaggeration) || options.exaggeration <= 0.0) {
    throw std::invalid_argument("openTSNE exaggeration values must be positive and finite.");
  }
  if (options.learning_rate <= 0.0 && !options.learning_rate_auto) {
    throw std::invalid_argument("openTSNE learning_rate must be positive or automatic.");
  }
  if (!std::isfinite(options.initial_momentum) || options.initial_momentum < 0.0 ||
      !std::isfinite(options.final_momentum) || options.final_momentum < 0.0) {
    throw std::invalid_argument("openTSNE momentum values must be non-negative and finite.");
  }
  if (!std::isfinite(options.min_gain) || options.min_gain <= 0.0) {
    throw std::invalid_argument("openTSNE min_gain must be positive and finite.");
  }
}

std::vector<float> optimize_opentsne_cpu(
  const PreparedGraph& graph,
  const OpenTSNEOptions& options
) {
  if (options.n_components < 1 || options.n_components > 3) {
    throw std::invalid_argument("CPU openTSNE supports n_components in 1..3.");
  }
  const int n = graph.samples;
  validate_opentsne_options(options, n);
  const int threads = effective_cpu_threads(options.n_threads, n);
  ParallelExecutor parallel_executor(threads);
  ParallelExecutorScope parallel_scope(&parallel_executor);
  SparseProbabilitiesF p = build_tsne_probabilities_float(graph, options.perplexity, threads);

  std::vector<float> y = options.init;
  if (y.size() != static_cast<std::size_t>(n) * static_cast<std::size_t>(options.n_components)) {
    y = make_opentsne_random_init(n, options.n_components, options.seed);
  }
  zero_mean_f(y, n, options.n_components);

  std::vector<float> grad(y.size(), 0.0f);
  std::vector<float> update(y.size(), 0.0f);
  std::vector<float> gains(y.size(), 1.0f);
  FftGridWorkspaceT<float> fft_workspace;
  // Match fastEmbedR's validated openTSNE dispatch: exact repulsion is both
  // faster and more faithful for small layouts; FFT is used above this size.
  const bool use_fft = n > 3000 && options.theta > 0.0 && options.n_components == 2;

  auto run_phase = [&](const int phase_iter, const double phase_exaggeration, const double phase_momentum) {
    if (phase_iter <= 0) return;
    const float phase_lr = static_cast<float>(options.learning_rate_auto ?
      static_cast<double>(n) / std::max(phase_exaggeration, DBL_MIN) :
      options.learning_rate);
    const float phase_exag_f = static_cast<float>(phase_exaggeration);
    const float phase_momentum_f = static_cast<float>(phase_momentum);
    const float min_gain_f = static_cast<float>(options.min_gain);
    const float max_step_norm_f = std::isfinite(options.max_step_norm) ?
      static_cast<float>(options.max_step_norm) :
      std::numeric_limits<float>::quiet_NaN();
    for (int iter = 0; iter < phase_iter; ++iter) {
      compute_gradient_f(
        p, y, n, options.n_components, phase_exag_f, threads, use_fft, &fft_workspace, grad
      );
      apply_open_tsne_update_f(
        y, update, gains, grad, n, options.n_components, phase_lr, phase_momentum_f,
        min_gain_f, max_step_norm_f, threads
      );
      zero_mean_f(y, n, options.n_components);
    }
  };

  run_phase(options.early_exaggeration_iter, options.early_exaggeration, options.initial_momentum);
  run_phase(options.n_iter, options.exaggeration, options.final_momentum);
  return y;
}

const char* cuda_embedding_error();

}  // namespace

#if defined(KODAMA_ENABLE_CUDA)
extern "C" {
bool kodama_cuda_available();
const char* kodama_cuda_embedding_last_error();
int kodama_cuda_umap_from_knn_spectral_float(
  const int* indices,
  const float* distances,
  int n,
  int k,
  int n_epochs,
  int negative_sample_rate,
  float learning_rate,
  float a,
  float b,
  float repulsion_strength,
  int spectral_n_iter,
  unsigned int seed,
  int index_offset,
  int optimizer_mode,
  float* out
);
int kodama_cuda_umap_optimize_coo(
  const int* heads,
  const int* tails,
  const float* weights,
  const float* epochs_per_sample,
  const float* init,
  int n,
  int n_edges_capacity,
  int n_epochs,
  int negative_sample_rate,
  float learning_rate,
  float a,
  float b,
  float repulsion_strength,
  unsigned int seed,
  int optimizer_mode,
  float* out
);
int kodama_cuda_opentsne_fft_from_knn_float(
  const int* indices,
  const float* distances,
  const float* init,
  int has_init,
  int n,
  int k,
  int n_components,
  float perplexity,
  int early_exaggeration_iter,
  int n_iter,
  float early_exaggeration,
  float exaggeration,
  float learning_rate,
  int learning_rate_auto,
  float initial_momentum,
  float final_momentum,
  float min_gain,
  float max_step_norm,
  unsigned int seed,
  int index_offset,
  float* out
);
}

namespace {
const char* cuda_embedding_error() {
  const char* message = kodama_cuda_embedding_last_error();
  return message == nullptr || message[0] == '\0' ? "unknown CUDA embedding error" : message;
}
}  // namespace
#else
namespace {
const char* cuda_embedding_error() {
  return "kodama-cpp was built without CUDA visualization support";
}
}  // namespace
#endif

namespace {
void validate_umap_options(const UMAPOptions& options) {
  if (options.n_epochs < 0) {
    throw std::invalid_argument("UMAP n_epochs must be non-negative.");
  }
  if (!std::isfinite(options.min_dist) || options.min_dist < 0.0) {
    throw std::invalid_argument("UMAP min_dist must be non-negative and finite.");
  }
  if (options.negative_sample_rate < 0) {
    throw std::invalid_argument("UMAP negative_sample_rate must be non-negative.");
  }
  if (!std::isfinite(options.learning_rate) || options.learning_rate <= 0.0) {
    throw std::invalid_argument("UMAP learning_rate must be positive and finite.");
  }
  if (!std::isfinite(options.repulsion_strength) || options.repulsion_strength <= 0.0) {
    throw std::invalid_argument("UMAP repulsion_strength must be positive and finite.");
  }
  if (options.spectral_n_iter < 1) {
    throw std::invalid_argument("UMAP spectral_n_iter must be positive.");
  }
  if (options.n_threads < 1) {
    throw std::invalid_argument("UMAP n_threads must be positive.");
  }
}

std::vector<float> centered_pca_scores(
  const PCAResult& pca,
  const int requested_components
) {
  std::vector<float> out(
    static_cast<std::size_t>(pca.samples) * requested_components,
    0.0f
  );
  const int copied_components = std::min(pca.components, requested_components);
  for (int row = 0; row < pca.samples; ++row) {
    for (int component = 0; component < copied_components; ++component) {
      out[static_cast<std::size_t>(row) * requested_components + component] =
        pca.scores[static_cast<std::size_t>(row) * pca.components + component];
    }
  }
  for (int component = 0; component < requested_components; ++component) {
    float mean = 0.0f;
    for (int row = 0; row < pca.samples; ++row) {
      mean += out[static_cast<std::size_t>(row) * requested_components + component];
    }
    mean /= static_cast<float>(pca.samples);
    for (int row = 0; row < pca.samples; ++row) {
      out[static_cast<std::size_t>(row) * requested_components + component] -= mean;
    }
  }
  return out;
}

void scale_opentsne_pca_init(
  std::vector<float>& init,
  const int samples,
  const int components
) {
  float max_sd = 0.0f;
  for (int component = 0; component < components; ++component) {
    float sum_squares = 0.0f;
    for (int row = 0; row < samples; ++row) {
      const float value = init[static_cast<std::size_t>(row) * components + component];
      sum_squares += value * value;
    }
    const float variance = samples > 1 ?
      sum_squares / static_cast<float>(samples - 1) :
      0.0f;
    max_sd = std::max(max_sd, std::sqrt(std::max(0.0f, variance)));
  }
  if (std::isfinite(max_sd) && max_sd > 0.0f) {
    const float multiplier = 1.0e-4f / max_sd;
    for (float& value : init) value *= multiplier;
  }
}

void scale_umap_pca_init(
  std::vector<float>& init,
  const int samples,
  const int components,
  const std::uint64_t seed
) {
  float max_abs = 0.0f;
  for (const float value : init) max_abs = std::max(max_abs, std::abs(value));
  if (std::isfinite(max_abs) && max_abs > 0.0f) {
    const float multiplier = 10.0f / max_abs;
    for (float& value : init) value *= multiplier;
  }

  std::mt19937 rng(static_cast<std::uint32_t>(seed));
  std::normal_distribution<float> noise(0.0f, 1.0e-4f);
  for (float& value : init) value += noise(rng);
  for (int component = 0; component < components; ++component) {
    float mean = 0.0f;
    for (int row = 0; row < samples; ++row) {
      mean += init[static_cast<std::size_t>(row) * components + component];
    }
    mean /= static_cast<float>(samples);
    for (int row = 0; row < samples; ++row) {
      init[static_cast<std::size_t>(row) * components + component] -= mean;
    }
  }
}

void set_embedding_initialization(
  EmbeddingResult& result,
  const bool has_explicit_init,
  const std::string& requested_source,
  const Backend requested_backend,
  const char* fallback_source
) {
  if (has_explicit_init) {
    result.initialization = requested_source.empty() ? "explicit" : requested_source;
    result.initialization_backend = requested_backend == Backend::Auto ?
      result.backend :
      requested_backend;
  } else {
    result.initialization = fallback_source;
    result.initialization_backend = Backend::CPU;
  }
}

void validate_raw_initialization_rows(
  const NeighborGraph& graph,
  const MatrixView raw_data
) {
  if (graph.neighbors < 1 ||
      graph.indices.size() % static_cast<std::size_t>(graph.neighbors) != 0u) {
    throw std::invalid_argument("Visualization graph has inconsistent dimensions.");
  }
  const std::size_t graph_samples =
    graph.indices.size() / static_cast<std::size_t>(graph.neighbors);
  if (raw_data.rows != graph_samples) {
    throw std::invalid_argument(
      "Raw initialization data must have one row per graph sample."
    );
  }
}
}  // namespace

VisualizationInitResult KODAMAVisualizationPCAInit(
  const MatrixView raw_data,
  const VisualizationInitOptions& options
) {
  if (raw_data.data == nullptr || raw_data.rows < 2 || raw_data.cols < 1) {
    throw std::invalid_argument(
      "Visualization PCA initialization requires at least two rows and one column."
    );
  }
  if (options.n_components < 1) {
    throw std::invalid_argument("Visualization initialization components must be positive.");
  }
  if (options.n_threads < 1) {
    throw std::invalid_argument("Visualization initialization threads must be positive.");
  }

  const auto started = Clock::now();
  PCAOptions pca_options;
  pca_options.n_components = std::min<int>(
    options.n_components,
    std::min<std::size_t>(raw_data.cols, raw_data.rows - 1)
  );
  pca_options.center = true;
  pca_options.scale = false;
  pca_options.n_threads = options.n_threads;
  pca_options.seed = options.seed;
  pca_options.gpu_device = options.gpu_device;
  pca_options.backend = options.backend;
  const PCAResult pca = PCA(raw_data, pca_options);

  VisualizationInitResult result;
  result.samples = pca.samples;
  result.components = options.n_components;
  result.backend = pca.backend;
  result.opentsne = centered_pca_scores(pca, options.n_components);
  result.umap = result.opentsne;
  scale_opentsne_pca_init(result.opentsne, result.samples, result.components);
  scale_umap_pca_init(
    result.umap,
    result.samples,
    result.components,
    options.seed
  );
  const auto finished = Clock::now();
  result.runtime_seconds = std::chrono::duration<double>(finished - started).count();
  return result;
}

EmbeddingResult KODAMAUMAP_CPU(
  const NeighborGraph& graph,
  const UMAPOptions& options
) {
  if (options.n_components != 2) {
    throw std::invalid_argument("CPU UMAP currently supports n_components=2.");
  }
  validate_umap_options(options);
  const auto started = Clock::now();
  PreparedGraph prepared = prepare_graph(graph, options.n_neighbors);
  CsrGraph csr = build_selected_umap_graph(prepared, options);
  EmbeddingResult result;
  result.samples = prepared.samples;
  result.components = 2;
  result.backend = Backend::CPU;
  result.optimizer = "csr_epoch_schedule";
  result.graph_edges = csr.neighbors.size();
  result.graph_max_weight = csr.max_weight;
  result.embedding = optimize_umap_csr_cpu(csr, options, prepared.samples);
  set_embedding_initialization(
    result,
    options.init.size() == static_cast<std::size_t>(prepared.samples) * 2u,
    options.init_source,
    options.init_backend,
    "graph_spectral"
  );
  const auto finished = Clock::now();
  result.runtime_seconds = std::chrono::duration<double>(finished - started).count();
  return result;
}

EmbeddingResult KODAMAUMAP_CUDA(
  const NeighborGraph& graph,
  const UMAPOptions& options
) {
#if !defined(KODAMA_ENABLE_CUDA)
  (void)graph;
  (void)options;
  throw std::runtime_error(cuda_embedding_error());
#else
  if (options.n_components != 2) {
    throw std::invalid_argument("CUDA UMAP currently supports n_components=2.");
  }
  validate_umap_options(options);
  if (!kodama_cuda_available()) {
    throw std::runtime_error("No CUDA device is available for KODAMA UMAP.");
  }
  const auto started = Clock::now();
  PreparedGraph prepared = prepare_graph(graph, options.n_neighbors);
  CsrGraph csr = build_selected_umap_graph(prepared, options);
  if (csr.neighbors.empty()) {
    throw std::runtime_error("CUDA UMAP requires at least one graph edge.");
  }
  std::vector<int> heads(csr.neighbors.size());
  for (int row = 0; row < prepared.samples; ++row) {
    const int begin = csr.offsets[static_cast<std::size_t>(row)];
    const int end = csr.offsets[static_cast<std::size_t>(row + 1)];
    for (int pos = begin; pos < end; ++pos) {
      heads[static_cast<std::size_t>(pos)] = row;
    }
  }
  std::vector<float> init = options.init;
  if (init.size() != static_cast<std::size_t>(prepared.samples) * 2u) {
    init = initialize_layout_csr_spectral(
      csr,
      prepared.samples,
      2,
      options.spectral_n_iter,
      options.seed
    );
  }
  const auto ab = find_ab_params(1.0, options.min_dist);
  EmbeddingResult result;
  result.samples = prepared.samples;
  result.components = 2;
  result.backend = Backend::CUDA;
  result.optimizer = "cuda_atomic_coo_epoch_schedule";
  result.graph_edges = csr.neighbors.size();
  result.graph_max_weight = csr.max_weight;
  result.embedding.assign(static_cast<std::size_t>(prepared.samples) * 2u, 0.0f);

  const int status = kodama_cuda_umap_optimize_coo(
    heads.data(),
    csr.neighbors.data(),
    csr.weights.data(),
    csr.epochs_per_sample.data(),
    init.data(),
    prepared.samples,
    static_cast<int>(csr.neighbors.size()),
    options.n_epochs,
    options.negative_sample_rate,
    static_cast<float>(options.learning_rate),
    static_cast<float>(ab.first),
    static_cast<float>(ab.second),
    static_cast<float>(options.repulsion_strength),
    static_cast<unsigned int>(options.seed),
    0,
    result.embedding.data()
  );
  if (status != 0) {
    throw std::runtime_error(std::string("CUDA UMAP failed: ") + cuda_embedding_error());
  }
  set_embedding_initialization(
    result,
    options.init.size() == static_cast<std::size_t>(prepared.samples) * 2u,
    options.init_source,
    options.init_backend,
    "graph_spectral"
  );
  const auto finished = Clock::now();
  result.runtime_seconds = std::chrono::duration<double>(finished - started).count();
  return result;
#endif
}

EmbeddingResult KODAMAUMAP_METAL(
  const NeighborGraph& graph,
  const UMAPOptions& options
) {
  if (options.n_components != 2) {
    throw std::invalid_argument("Metal UMAP currently supports n_components=2.");
  }
  validate_umap_options(options);
  if (!detail::metal_backend_available()) {
    throw std::runtime_error("No Metal device is available for KODAMA UMAP.");
  }
  const auto started = Clock::now();
  PreparedGraph prepared = prepare_graph(graph, options.n_neighbors);
  CsrGraph csr = build_selected_umap_graph(prepared, options);
  if (csr.neighbors.empty() || !std::isfinite(csr.max_weight) ||
      csr.max_weight <= 0.0f) {
    throw std::runtime_error("Metal UMAP requires at least one usable graph edge.");
  }

  int width = 0;
  for (int row = 0; row < prepared.samples; ++row) {
    width = std::max(
      width,
      csr.offsets[static_cast<std::size_t>(row + 1)] -
      csr.offsets[static_cast<std::size_t>(row)]
    );
  }
  if (width < 1) {
    throw std::runtime_error("Metal UMAP graph has no usable rows.");
  }
  const std::size_t rectangular_size =
    static_cast<std::size_t>(prepared.samples) * static_cast<std::size_t>(width);
  std::vector<int> neighbors(rectangular_size, -1);
  std::vector<float> weights(rectangular_size, 0.0f);
  for (int row = 0; row < prepared.samples; ++row) {
    const int begin = csr.offsets[static_cast<std::size_t>(row)];
    const int end = csr.offsets[static_cast<std::size_t>(row + 1)];
    int column = 0;
    for (int position = begin; position < end; ++position) {
      const int neighbor = csr.neighbors[static_cast<std::size_t>(position)];
      const float weight = csr.weights[static_cast<std::size_t>(position)];
      if (neighbor < 0 || neighbor >= prepared.samples || neighbor == row ||
          !std::isfinite(weight) || weight <= 0.0f) {
        continue;
      }
      const std::size_t output =
        static_cast<std::size_t>(row) * static_cast<std::size_t>(width) +
        static_cast<std::size_t>(column++);
      neighbors[output] = neighbor;
      weights[output] = weight;
    }
  }

  std::vector<float> initialization = options.init;
  if (initialization.size() != static_cast<std::size_t>(prepared.samples) * 2u) {
    initialization = initialize_layout_csr_spectral(
      csr,
      prepared.samples,
      2,
      options.spectral_n_iter,
      options.seed
    );
  }
  const auto ab = find_ab_params(1.0, options.min_dist);
  EmbeddingResult result;
  result.samples = prepared.samples;
  result.components = 2;
  result.backend = Backend::Metal;
  result.optimizer = "metal_clean_atomic_edge_sampler";
  result.graph_edges = csr.neighbors.size();
  result.graph_max_weight = csr.max_weight;
  result.embedding = detail::metal_umap_optimize(
    neighbors,
    weights,
    initialization,
    prepared.samples,
    width,
    options.n_epochs,
    options.negative_sample_rate,
    static_cast<float>(options.learning_rate),
    static_cast<float>(ab.first),
    static_cast<float>(ab.second),
    csr.max_weight,
    static_cast<float>(options.repulsion_strength),
    static_cast<std::uint32_t>(options.seed)
  );
  set_embedding_initialization(
    result,
    options.init.size() == static_cast<std::size_t>(prepared.samples) * 2u,
    options.init_source,
    options.init_backend,
    "graph_spectral"
  );
  const auto finished = Clock::now();
  result.runtime_seconds = std::chrono::duration<double>(finished - started).count();
  return result;
}

EmbeddingResult KODAMAOpenTSNE_CPU(
  const NeighborGraph& graph,
  const OpenTSNEOptions& options
) {
  const auto started = Clock::now();
  const int requested_neighbors = options.n_neighbors > 0 ?
    options.n_neighbors :
    static_cast<int>(std::ceil(options.perplexity));
  PreparedGraph prepared = prepare_graph(graph, requested_neighbors);
  EmbeddingResult result;
  result.samples = prepared.samples;
  result.components = options.n_components;
  result.backend = Backend::CPU;
  result.embedding = optimize_opentsne_cpu(prepared, options);
  result.optimizer = prepared.samples > 3000 &&
      options.theta > 0.0 && options.n_components == 2 ?
    "opentsne_fitsne_fft_grid_sparse_knn_float32" :
    "opentsne_exact_sparse_knn_float32";
  set_embedding_initialization(
    result,
    options.init.size() ==
      static_cast<std::size_t>(prepared.samples) * options.n_components,
    options.init_source,
    options.init_backend,
    "random"
  );
  const auto finished = Clock::now();
  result.runtime_seconds = std::chrono::duration<double>(finished - started).count();
  return result;
}

EmbeddingResult KODAMAOpenTSNE_CUDA(
  const NeighborGraph& graph,
  const OpenTSNEOptions& options
) {
#if !defined(KODAMA_ENABLE_CUDA)
  (void)graph;
  (void)options;
  throw std::runtime_error(cuda_embedding_error());
#else
  if (options.n_components != 2) {
    throw std::invalid_argument("CUDA openTSNE currently supports n_components=2.");
  }
  if (!kodama_cuda_available()) {
    throw std::runtime_error("No CUDA device is available for KODAMA openTSNE.");
  }
  const auto started = Clock::now();
  const int requested_neighbors = options.n_neighbors > 0 ?
    options.n_neighbors :
    static_cast<int>(std::ceil(options.perplexity));
  PreparedGraph prepared = prepare_graph(graph, requested_neighbors);
  validate_opentsne_options(options, prepared.samples);
  std::vector<float> init = options.init;
  int has_init = static_cast<int>(init.size() == static_cast<std::size_t>(prepared.samples) * 2u);
  if (!has_init) {
    init = make_opentsne_random_init(prepared.samples, options.n_components, options.seed);
    has_init = 1;
  }

  EmbeddingResult result;
  result.samples = prepared.samples;
  result.components = 2;
  result.backend = Backend::CUDA;
  result.optimizer = "cuda_opentsne_fft_grid_sparse_knn_float32";
  result.embedding.assign(static_cast<std::size_t>(prepared.samples) * 2u, 0.0f);
  const int status = kodama_cuda_opentsne_fft_from_knn_float(
    prepared.indices.data(),
    prepared.distances.data(),
    init.data(),
    has_init,
    prepared.samples,
    prepared.neighbors,
    options.n_components,
    static_cast<float>(options.perplexity),
    options.early_exaggeration_iter,
    options.n_iter,
    static_cast<float>(options.early_exaggeration),
    static_cast<float>(options.exaggeration),
    static_cast<float>(options.learning_rate),
    options.learning_rate_auto ? 1 : 0,
    static_cast<float>(options.initial_momentum),
    static_cast<float>(options.final_momentum),
    static_cast<float>(options.min_gain),
    static_cast<float>(options.max_step_norm),
    static_cast<unsigned int>(options.seed),
    prepared.index_offset,
    result.embedding.data()
  );
  if (status != 0) {
    throw std::runtime_error(std::string("CUDA openTSNE failed: ") + cuda_embedding_error());
  }
  set_embedding_initialization(
    result,
    options.init.size() == static_cast<std::size_t>(prepared.samples) * 2u,
    options.init_source,
    options.init_backend,
    "random"
  );
  const auto finished = Clock::now();
  result.runtime_seconds = std::chrono::duration<double>(finished - started).count();
  return result;
#endif
}

EmbeddingResult KODAMAOpenTSNE_METAL(
  const NeighborGraph& graph,
  const OpenTSNEOptions& options
) {
#if !defined(KODAMA_ENABLE_METAL)
  (void)graph;
  (void)options;
  throw std::runtime_error("The Metal backend is not available in this build.");
#else
  if (options.n_components != 2) {
    throw std::invalid_argument("Metal openTSNE currently supports n_components=2.");
  }
  if (!detail::metal_backend_available()) {
    throw std::runtime_error("No Metal device is available for KODAMA openTSNE.");
  }
  const auto started = Clock::now();
  const int requested_neighbors = options.n_neighbors > 0 ?
    options.n_neighbors :
    static_cast<int>(std::ceil(options.perplexity));
  PreparedGraph prepared = prepare_graph(graph, requested_neighbors);
  validate_opentsne_options(options, prepared.samples);
  const int threads = effective_cpu_threads(options.n_threads, prepared.samples);
  SparseProbabilitiesF probabilities = build_tsne_probabilities_float(
    prepared, options.perplexity, threads);
  std::vector<float> init = options.init;
  if (init.size() != static_cast<std::size_t>(prepared.samples) * 2u) {
    init = make_opentsne_random_init(prepared.samples, 2, options.seed);
  }

  EmbeddingResult result;
  result.samples = prepared.samples;
  result.components = 2;
  result.backend = Backend::Metal;
  result.optimizer = "metal_opentsne_fft_grid_sparse_knn_float32";
  result.embedding = detail::metal_opentsne_optimize(
    probabilities.row_ptr,
    probabilities.col,
    probabilities.val,
    init,
    prepared.samples,
    options.early_exaggeration_iter,
    options.n_iter,
    static_cast<float>(options.early_exaggeration),
    static_cast<float>(options.exaggeration),
    static_cast<float>(options.learning_rate),
    options.learning_rate_auto,
    static_cast<float>(options.initial_momentum),
    static_cast<float>(options.final_momentum),
    static_cast<float>(options.min_gain),
    static_cast<float>(options.max_step_norm),
    static_cast<std::uint32_t>(options.seed)
  );
  set_embedding_initialization(
    result,
    options.init.size() == static_cast<std::size_t>(prepared.samples) * 2u,
    options.init_source,
    options.init_backend,
    "random"
  );
  result.runtime_seconds = std::chrono::duration<double>(
    Clock::now() - started).count();
  return result;
#endif
}

EmbeddingResult KODAMAUMAP_CPU(
  const NeighborGraph& graph,
  const MatrixView raw_data,
  const UMAPOptions& options
) {
  validate_raw_initialization_rows(graph, raw_data);
  UMAPOptions resolved = options;
  if (resolved.init.size() != raw_data.rows * 2u) {
    VisualizationInitOptions init_options;
    init_options.n_components = 2;
    init_options.n_threads = options.n_threads;
    init_options.seed = static_cast<std::uint64_t>(options.seed);
    init_options.gpu_device = options.gpu_device;
    init_options.backend = Backend::CPU;
    const VisualizationInitResult init =
      KODAMAVisualizationPCAInit(raw_data, init_options);
    resolved.init = init.umap;
    resolved.init_source = "raw_pca";
    resolved.init_backend = init.backend;
  }
  return KODAMAUMAP_CPU(graph, resolved);
}

EmbeddingResult KODAMAUMAP_CUDA(
  const NeighborGraph& graph,
  const MatrixView raw_data,
  const UMAPOptions& options
) {
  validate_raw_initialization_rows(graph, raw_data);
  UMAPOptions resolved = options;
  if (resolved.init.size() != raw_data.rows * 2u) {
    VisualizationInitOptions init_options;
    init_options.n_components = 2;
    init_options.n_threads = options.n_threads;
    init_options.seed = static_cast<std::uint64_t>(options.seed);
    init_options.gpu_device = options.gpu_device;
    init_options.backend = Backend::CUDA;
    const VisualizationInitResult init =
      KODAMAVisualizationPCAInit(raw_data, init_options);
    resolved.init = init.umap;
    resolved.init_source = "raw_pca";
    resolved.init_backend = init.backend;
  }
  return KODAMAUMAP_CUDA(graph, resolved);
}

EmbeddingResult KODAMAUMAP_METAL(
  const NeighborGraph& graph,
  const MatrixView raw_data,
  const UMAPOptions& options
) {
  validate_raw_initialization_rows(graph, raw_data);
  UMAPOptions resolved = options;
  if (resolved.init.size() != raw_data.rows * 2u) {
    VisualizationInitOptions init_options;
    init_options.n_components = 2;
    init_options.n_threads = options.n_threads;
    init_options.seed = static_cast<std::uint64_t>(options.seed);
    init_options.gpu_device = options.gpu_device;
    init_options.backend = Backend::Metal;
    const VisualizationInitResult init =
      KODAMAVisualizationPCAInit(raw_data, init_options);
    resolved.init = init.umap;
    resolved.init_source = "raw_pca";
    resolved.init_backend = init.backend;
  }
  return KODAMAUMAP_METAL(graph, resolved);
}

EmbeddingResult KODAMAOpenTSNE_CPU(
  const NeighborGraph& graph,
  const MatrixView raw_data,
  const OpenTSNEOptions& options
) {
  validate_raw_initialization_rows(graph, raw_data);
  OpenTSNEOptions resolved = options;
  if (resolved.init.size() !=
      raw_data.rows * static_cast<std::size_t>(options.n_components)) {
    VisualizationInitOptions init_options;
    init_options.n_components = options.n_components;
    init_options.n_threads = options.n_threads;
    init_options.seed = static_cast<std::uint64_t>(options.seed);
    init_options.gpu_device = options.gpu_device;
    init_options.backend = Backend::CPU;
    const VisualizationInitResult init =
      KODAMAVisualizationPCAInit(raw_data, init_options);
    resolved.init = init.opentsne;
    resolved.init_source = "raw_pca";
    resolved.init_backend = init.backend;
  }
  return KODAMAOpenTSNE_CPU(graph, resolved);
}

EmbeddingResult KODAMAOpenTSNE_CUDA(
  const NeighborGraph& graph,
  const MatrixView raw_data,
  const OpenTSNEOptions& options
) {
  validate_raw_initialization_rows(graph, raw_data);
  OpenTSNEOptions resolved = options;
  if (resolved.init.size() !=
      raw_data.rows * static_cast<std::size_t>(options.n_components)) {
    VisualizationInitOptions init_options;
    init_options.n_components = options.n_components;
    init_options.n_threads = options.n_threads;
    init_options.seed = static_cast<std::uint64_t>(options.seed);
    init_options.gpu_device = options.gpu_device;
    init_options.backend = Backend::CUDA;
    const VisualizationInitResult init =
      KODAMAVisualizationPCAInit(raw_data, init_options);
    resolved.init = init.opentsne;
    resolved.init_source = "raw_pca";
    resolved.init_backend = init.backend;
  }
  return KODAMAOpenTSNE_CUDA(graph, resolved);
}

EmbeddingResult KODAMAOpenTSNE_METAL(
  const NeighborGraph& graph,
  const MatrixView raw_data,
  const OpenTSNEOptions& options
) {
  validate_raw_initialization_rows(graph, raw_data);
  OpenTSNEOptions resolved = options;
  if (resolved.init.size() != raw_data.rows * 2u) {
    VisualizationInitOptions init_options;
    init_options.n_components = 2;
    init_options.n_threads = options.n_threads;
    init_options.seed = static_cast<std::uint64_t>(options.seed);
    init_options.gpu_device = options.gpu_device;
    init_options.backend = Backend::Metal;
    const VisualizationInitResult init =
      KODAMAVisualizationPCAInit(raw_data, init_options);
    resolved.init = init.opentsne;
    resolved.init_source = "raw_pca";
    resolved.init_backend = init.backend;
  }
  return KODAMAOpenTSNE_METAL(graph, resolved);
}

}  // namespace kodama
