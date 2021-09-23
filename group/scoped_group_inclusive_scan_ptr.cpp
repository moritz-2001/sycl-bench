#include "common.h"
#include "group_common.hpp"
#include <iostream>

namespace s = cl::sycl;

template <typename DataT, int Iterations>
class ScopedMicroBenchGroupInclusiveScanPtrKernel;

/**
 * ScopedMicrobenchmark benchmarking group_inclusive_scan_ptr
 */
template <typename DataT, int Iterations = 512>
class ScopedMicroBenchGroupInclusiveScanPtr {
protected:
  BenchmarkArgs args;

  std::vector<DataT> input;
  std::vector<DataT> output;
  std::vector<DataT> devNull;
  PrefetchedBuffer<DataT, 1> input_buf;
  PrefetchedBuffer<DataT, 1> output_buf;
  PrefetchedBuffer<DataT, 1> devNull_buf;

public:
  ScopedMicroBenchGroupInclusiveScanPtr(const BenchmarkArgs& _args) : args(_args) {}

  void setup() {
    input.resize(args.problem_size, initialize_type<DataT>(1));
    output.resize(args.problem_size, initialize_type<DataT>(1));
    devNull.resize(args.problem_size, initialize_type<DataT>(1));

    output_buf.initialize(args.device_queue, output.data(), s::range<1>(input.size()));
    devNull_buf.initialize(args.device_queue, devNull.data(), s::range<1>(input.size()));
    input_buf.initialize(args.device_queue, input.data(), s::range<1>(input.size()));
  }

  void run(std::vector<cl::sycl::event>& events) {
    size_t num_groups = args.local_size;
    size_t problem_size = args.problem_size;
    events.push_back(args.device_queue.submit([&](cl::sycl::handler& cgh) {
      auto in = input_buf.template get_access<s::access::mode::read>(cgh);
      auto out = output_buf.template get_access<s::access::mode::discard_write>(cgh);
      auto devNull = output_buf.template get_access<s::access::mode::discard_write>(cgh);

      cgh.parallel<ScopedMicroBenchGroupInclusiveScanPtr<DataT, Iterations>>(
          s::range<1>{num_groups}, s::range<1>{args.local_size}, [=](s::group<1> g, s::physical_item<1> pitem) {
            DataT d = initialize_type<DataT>(-1);
            size_t gid = pitem.get_global_id(0);
            auto start = in.get_pointer();
            auto end = start + static_cast<size_t>(problem_size);

            auto result = devNull.get_pointer();
            if(g.get_linear() == 0)
              result = out.get_pointer();

            for(int i = 1; i <= Iterations; ++i) {
              s::detail::leader_inclusive_scan(g, start.get(), end.get(), result.get(), s::plus<DataT>());
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

  app.run<ScopedMicroBenchGroupInclusiveScanPtr<int>>();
  app.run<ScopedMicroBenchGroupInclusiveScanPtr<long long>>();
  app.run<ScopedMicroBenchGroupInclusiveScanPtr<float>>();
  app.run<ScopedMicroBenchGroupInclusiveScanPtr<double>>();
  app.run<ScopedMicroBenchGroupInclusiveScanPtr<cl::sycl::vec<int, 1>>>();
  app.run<ScopedMicroBenchGroupInclusiveScanPtr<cl::sycl::vec<unsigned char, 4>>>();
  app.run<ScopedMicroBenchGroupInclusiveScanPtr<cl::sycl::vec<int, 4>>>();
  app.run<ScopedMicroBenchGroupInclusiveScanPtr<cl::sycl::vec<int, 8>>>();
  app.run<ScopedMicroBenchGroupInclusiveScanPtr<cl::sycl::vec<float, 1>>>();
  app.run<ScopedMicroBenchGroupInclusiveScanPtr<cl::sycl::vec<double, 2>>>();
  app.run<ScopedMicroBenchGroupInclusiveScanPtr<cl::sycl::vec<float, 4>>>();
  app.run<ScopedMicroBenchGroupInclusiveScanPtr<cl::sycl::vec<float, 8>>>();
  return 0;
}
