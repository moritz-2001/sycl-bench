#include "common.h"
#include "group_common.hpp"
#include <iostream>

namespace s = cl::sycl;

template <typename DataT, int Iterations>
class MicroBenchGroupReducePtrKernel;

/**
 * Microbenchmark benchmarking group_reduceA_ptr
 */
template <typename DataT, bool IsVector = false, int Iterations = 512>
class MicroBenchGroupReducePtr {
protected:
  BenchmarkArgs args;

  std::vector<DataT> input;
  PrefetchedBuffer<DataT, 1> input_buf;
  PrefetchedBuffer<DataT, 1> output_buf;

public:
  MicroBenchGroupReducePtr(const BenchmarkArgs& _args) : args(_args) {}

  void setup() {
    input.resize(args.problem_size, initialize_type<DataT>(1));

    output_buf.initialize(args.device_queue, s::range<1>(1));
    input_buf.initialize(args.device_queue, input.data(), s::range<1>(input.size()));
  }

  void run(std::vector<cl::sycl::event>& events) {
    size_t num_groups = args.local_size;
    size_t input_size = args.problem_size;
    events.push_back(args.device_queue.submit([&](cl::sycl::handler& cgh) {
      auto in = input_buf.template get_access<s::access::mode::read>(cgh);
      auto out = output_buf.template get_access<s::access::mode::discard_write>(cgh);

      cgh.parallel_for<MicroBenchGroupReducePtrKernel<DataT, Iterations>>(
          s::nd_range<1>{num_groups * args.local_size, args.local_size}, [=](cl::sycl::nd_item<1> item) {
            auto g = item.get_group();
            size_t gid = item.get_global_linear_id();
            DataT d = initialize_type<DataT>(0);

            auto start = in.get_pointer();
            auto end = start + static_cast<size_t>(input_size);

            for(int i = 1; i <= Iterations; ++i) {
              d = s::detail::reduce(g, start.get(), end.get(), [](DataT a, DataT b) { return a + b; });
              out[0] = d;
            }

            if(gid == 0)
              out[0] = d;
          });
    }));
  }

  bool verify(VerificationSetting& ver) {
    auto result = output_buf.template get_access<s::access::mode::read>();
    DataT expected = initialize_type<DataT>(args.problem_size);

    if(!compare_type(result[0], expected))
      std::cout << type_to_string(expected) << ":" << type_to_string(result[0]) << std::endl;
    return compare_type(result[0], expected);
  }

  static std::string getBenchmarkName() {
    std::stringstream name;
    name << "GroupFunctionBench_Reduce_Ptr_";
    name << ReadableTypename<DataT>::name << "_";
    name << Iterations;
    return name.str();
  }
};

int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);

  app.run<MicroBenchGroupReducePtr<int>>();
  app.run<MicroBenchGroupReducePtr<long long>>();
  app.run<MicroBenchGroupReducePtr<float>>();
  app.run<MicroBenchGroupReducePtr<double>>();
  app.run<MicroBenchGroupReducePtr<cl::sycl::vec<int, 1>, true>>();
  app.run<MicroBenchGroupReducePtr<cl::sycl::vec<unsigned char, 4>, true>>();
  app.run<MicroBenchGroupReducePtr<cl::sycl::vec<int, 4>, true>>();
  app.run<MicroBenchGroupReducePtr<cl::sycl::vec<int, 8>, true>>();
  app.run<MicroBenchGroupReducePtr<cl::sycl::vec<float, 1>, true>>();
  app.run<MicroBenchGroupReducePtr<cl::sycl::vec<double, 2>, true>>();
  app.run<MicroBenchGroupReducePtr<cl::sycl::vec<float, 4>, true>>();
  app.run<MicroBenchGroupReducePtr<cl::sycl::vec<float, 8>, true>>();
  return 0;
}
