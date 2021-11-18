#include "common.h"
#include "group_common.hpp"
#include <iostream>

namespace s = cl::sycl;

template <typename DataT, int Iterations>
class MicroBenchGroupReduceKernel;

/**
 * Microbenchmark benchmarking group_reduce
 */
template <typename DataT, bool IsVector = false, int Iterations = 512>
class MicroBenchGroupReduce {
protected:
  BenchmarkArgs args;

  PrefetchedBuffer<DataT, 1> output_buf;

public:
  MicroBenchGroupReduce(const BenchmarkArgs& _args) : args(_args) {}

  void setup() { output_buf.initialize(args.device_queue, s::range<1>(1)); }

  void run(std::vector<cl::sycl::event>& events) {
    size_t num_groups = (args.problem_size + args.local_size - 1) / args.local_size;
    events.push_back(args.device_queue.submit([&](cl::sycl::handler& cgh) {
      auto out = output_buf.template get_access<s::access::mode::discard_write>(cgh);

      cgh.parallel_for<MicroBenchGroupReduceKernel<DataT, Iterations>>(
          s::nd_range<1>{num_groups * args.local_size, args.local_size}, [=](cl::sycl::nd_item<1> item) {
            auto g = item.get_group();
            size_t gid = item.get_global_linear_id();
            DataT d = initialize_type<DataT>(0);

            for(int i = 1; i <= Iterations; ++i) {
              DataT j = initialize_type<DataT>(1);
              d = s::group_reduce(g, j, s::plus<DataT>());
            }

            if(gid == 0)
              out[0] = d;
          });
    }));
  }

  bool verify(VerificationSetting& ver) {
    return true;
    auto result = output_buf.template get_access<s::access::mode::read>();
    DataT expected = initialize_type<DataT>(args.local_size);

    if(!compare_type(result[0], expected))
      std::cout << type_to_string(expected) << ":" << type_to_string(result[0]) << std::endl;
    return compare_type(result[0], expected);
  }

  static std::string getBenchmarkName() {
    std::stringstream name;
    name << "GroupFunctionBench_Reduce_";
    name << ReadableTypename<DataT>::name << "_";
    name << Iterations;
    return name.str();
  }
};

int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);

  app.run<MicroBenchGroupReduce<int>>();
  app.run<MicroBenchGroupReduce<long long>>();
  app.run<MicroBenchGroupReduce<float>>();
  app.run<MicroBenchGroupReduce<double>>();
  // app.run<MicroBenchGroupReduce<cl::sycl::vec<int, 1>, true>>();
  // app.run<MicroBenchGroupReduce<cl::sycl::vec<unsigned char, 4>, true>>();
  // app.run<MicroBenchGroupReduce<cl::sycl::vec<int, 4>, true>>();
  // app.run<MicroBenchGroupReduce<cl::sycl::vec<int, 8>, true>>();
  // app.run<MicroBenchGroupReduce<cl::sycl::vec<float, 1>, true>>();
  // app.run<MicroBenchGroupReduce<cl::sycl::vec<double, 2>, true>>();
  // app.run<MicroBenchGroupReduce<cl::sycl::vec<float, 4>, true>>();
  // app.run<MicroBenchGroupReduce<cl::sycl::vec<float, 8>, true>>();
  return 0;
}
