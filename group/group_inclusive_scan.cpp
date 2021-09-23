#include "common.h"
#include "group_common.hpp"

namespace s = cl::sycl;

template <typename DataT, int Iterations>
class MicroBenchGroupInclusiveScanKernel;

/**
 * Microbenchmark benchmarking group_reduce
 */
template <typename DataT, int Iterations = 512>
class MicroBenchGroupInclusiveScan {
protected:
  BenchmarkArgs args;

  PrefetchedBuffer<DataT, 1> output_buf;

public:
  MicroBenchGroupInclusiveScan(const BenchmarkArgs& _args) : args(_args) {}

  void setup() {
    size_t num_groups = (args.problem_size + args.local_size - 1) / args.local_size;

    output_buf.initialize(args.device_queue, s::range<1>(num_groups * args.local_size));
  }

  void run(std::vector<cl::sycl::event>& events) {
    size_t num_groups = (args.problem_size + args.local_size - 1) / args.local_size;
    events.push_back(args.device_queue.submit([&](cl::sycl::handler& cgh) {
      auto out = output_buf.template get_access<s::access::mode::discard_write>(cgh);

      cgh.parallel_for<MicroBenchGroupInclusiveScanKernel<DataT, Iterations>>(
          s::nd_range<1>{num_groups * args.local_size, args.local_size}, [=](cl::sycl::nd_item<1> item) {
            auto g = item.get_group();
            size_t gid = item.get_global_linear_id();
            DataT d = initialize_type<DataT>(0);

            for(int i = 1; i <= Iterations; ++i) {
              DataT j = initialize_type<DataT>(i);
              d = s::group_inclusive_scan(g, j, [](DataT a, DataT b) { return a + b; });
            }

            out[gid] = d;
          });
    }));
  }

  bool verify(VerificationSetting& ver) {
    auto result = output_buf.template get_access<s::access::mode::read>();
    DataT expected = initialize_type<DataT>(0);

    for(size_t i = 0; i < args.problem_size; ++i) {
      if(i % args.local_size == 0)
        expected = 0;
      expected += initialize_type<DataT>(Iterations);
      if(!compare_type(result[i], expected)) {
        std::cout << i << ":" << type_to_string(expected) << ":" << type_to_string(result[i]) << std::endl;
        return false;
      }
    }

    return true;
  }

  static std::string getBenchmarkName() {
    std::stringstream name;
    name << "GroupFunctionBench_InclusiveScan";
    name << ReadableTypename<DataT>::name << "_";
    name << Iterations;
    return name.str();
  }
};

int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);

  app.run<MicroBenchGroupInclusiveScan<int>>();
  app.run<MicroBenchGroupInclusiveScan<long long>>();
  app.run<MicroBenchGroupInclusiveScan<float>>();
  app.run<MicroBenchGroupInclusiveScan<double>>();
  app.run<MicroBenchGroupInclusiveScan<cl::sycl::vec<int, 1>>>();
  app.run<MicroBenchGroupInclusiveScan<cl::sycl::vec<int, 4>>>();
  app.run<MicroBenchGroupInclusiveScan<cl::sycl::vec<int, 8>>>();
  app.run<MicroBenchGroupInclusiveScan<cl::sycl::vec<float, 1>>>();
  app.run<MicroBenchGroupInclusiveScan<cl::sycl::vec<double, 2>>>();
  app.run<MicroBenchGroupInclusiveScan<cl::sycl::vec<float, 4>>>();
  app.run<MicroBenchGroupInclusiveScan<cl::sycl::vec<float, 8>>>();
  return 0;
}
