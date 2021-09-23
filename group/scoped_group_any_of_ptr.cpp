#include "common.h"
#include <iostream>

namespace s = cl::sycl;

template <int Iterations>
class ScopedMicroBenchGroupAnyOfPtrKernel;

/**
 * ScopedMicrobenchmark benchmarking group_reduceA_ptr
 */
template <int Iterations = 512>
class ScopedMicroBenchGroupAnyOfPtr {
protected:
  BenchmarkArgs args;

  std::vector<char> input;
  PrefetchedBuffer<char, 1> input_buf;
  PrefetchedBuffer<char, 1> output_buf;

public:
  ScopedMicroBenchGroupAnyOfPtr(const BenchmarkArgs& _args) : args(_args) {}

  void setup() {
    input.resize(args.problem_size, 0);

    output_buf.initialize(args.device_queue, s::range<1>(1));
    input_buf.initialize(args.device_queue, input.data(), s::range<1>(input.size()));
  }

  void run(std::vector<cl::sycl::event>& events) {
    size_t num_groups = args.local_size;
    size_t input_size = args.problem_size;
    events.push_back(args.device_queue.submit([&](cl::sycl::handler& cgh) {
      auto in = input_buf.template get_access<s::access::mode::read>(cgh);
      auto out = output_buf.template get_access<s::access::mode::discard_write>(cgh);

      cgh.parallel<ScopedMicroBenchGroupAnyOfPtrKernel<Iterations>>(
          s::range<1>{num_groups}, s::range<1>{args.local_size}, [=](s::group<1> g, s::physical_item<1> pitem) {
            size_t gid = pitem.get_global_id(0);
            bool d;

            auto start = in.get_pointer();
            auto end = start + static_cast<size_t>(input_size);

            for(int i = 1; i <= Iterations; ++i) {
              d = s::detail::leader_any_of(g, start.get(), end.get());
            }

            if(gid == 0)
              out[0] = d;
          });
    }));
  }

  bool verify(VerificationSetting& ver) {
    auto result = output_buf.template get_access<s::access::mode::read>();

    if(result[0])
      std::cout << "true"
                << ":" << +result[0] << std::endl;
    return !result[0];
  }

  static std::string getBenchmarkName() {
    std::stringstream name;
    name << "ScopedGroupFunctionBench_AnyOf_Ptr_";
    name << Iterations;
    return name.str();
  }
};

int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);

  app.run<ScopedMicroBenchGroupAnyOfPtr<>>();
  return 0;
}
