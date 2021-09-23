#include "common.h"
#include "group_common.hpp"
#include <iostream>

namespace s = cl::sycl;

template <typename DataT, int Iterations>
class MicroBenchGroupInclusiveScanPtrKernel;

/**
 * Microbenchmark benchmarking group_inclusive_scan_ptr
 */
template <typename DataT, bool IsVector = false, int Iterations = 512>
class MicroBenchGroupInclusiveScanPtr {
protected:
  BenchmarkArgs args;

  std::vector<DataT> input;
  std::vector<DataT> output;
  PrefetchedBuffer<DataT, 1> input_buf;
  PrefetchedBuffer<DataT, 1> output_buf;

public:
  MicroBenchGroupInclusiveScanPtr(const BenchmarkArgs& _args) : args(_args) {}

  void setup() {
    input.resize(args.problem_size, initialize_type<DataT>(1));
    output.resize(args.problem_size, initialize_type<DataT>(1));

    output_buf.initialize(args.device_queue, output.data(), s::range<1>(input.size()));
    input_buf.initialize(args.device_queue, input.data(), s::range<1>(input.size()));
  }

  void run(std::vector<cl::sycl::event>& events) {
    size_t num_groups = args.local_size;
    size_t problem_size = args.problem_size;
    events.push_back(args.device_queue.submit([&](cl::sycl::handler& cgh) {
      auto in = input_buf.template get_access<s::access::mode::read>(cgh);
      auto out = output_buf.template get_access<s::access::mode::discard_write>(cgh);

      cgh.parallel_for<MicroBenchGroupInclusiveScanPtrKernel<DataT, Iterations>>(
          s::nd_range<1>{num_groups * args.local_size, args.local_size}, [=](cl::sycl::nd_item<1> item) {
            auto g = item.get_group();
            size_t gid = item.get_global_linear_id();

            auto start = in.get_pointer();
            auto end = start + static_cast<size_t>(problem_size);
            auto result = out.get_pointer();

            for(int i = 1; i <= Iterations; ++i) {
              s::detail::inclusive_scan(
                  g, start.get(), end.get(), result.get(), [](DataT a, DataT b) { return a + b; });
            }
          });
    }));
  }

  bool verify(VerificationSetting& ver) {
    auto result = output_buf.template get_access<s::access::mode::read>();
    DataT expected = initialize_type<DataT>(0);

    for(size_t i = 0; i < args.problem_size; ++i) {
      expected += initialize_type<DataT>(1);
      if(!compare_type(result[i], expected)) {
        std::cout << i << ":" << type_to_string(expected) << ":" << type_to_string(result[i]) << std::endl;
        return false;
      }
    }

    return true;
  }

  static std::string getBenchmarkName() {
    std::stringstream name;
    name << "GroupFunctionBench_Inclusive_Scan_Ptr_";
    name << ReadableTypename<DataT>::name << "_";
    name << Iterations;
    return name.str();
  }
};

int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);

  app.run<MicroBenchGroupInclusiveScanPtr<int>>();
  app.run<MicroBenchGroupInclusiveScanPtr<long long>>();
  app.run<MicroBenchGroupInclusiveScanPtr<float>>();
  app.run<MicroBenchGroupInclusiveScanPtr<double>>();
  app.run<MicroBenchGroupInclusiveScanPtr<cl::sycl::vec<int, 1>>>();
  app.run<MicroBenchGroupInclusiveScanPtr<cl::sycl::vec<unsigned char, 4>>>();
  app.run<MicroBenchGroupInclusiveScanPtr<cl::sycl::vec<int, 4>>>();
  app.run<MicroBenchGroupInclusiveScanPtr<cl::sycl::vec<int, 8>>>();
  app.run<MicroBenchGroupInclusiveScanPtr<cl::sycl::vec<float, 1>>>();
  app.run<MicroBenchGroupInclusiveScanPtr<cl::sycl::vec<double, 2>>>();
  app.run<MicroBenchGroupInclusiveScanPtr<cl::sycl::vec<float, 4>>>();
  app.run<MicroBenchGroupInclusiveScanPtr<cl::sycl::vec<float, 8>>>();
  return 0;
}
