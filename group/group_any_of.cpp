#include "common.h"

namespace s = cl::sycl;

template <int Iterations>
class MicroBenchGroupAnyOfKernel;

/**
 * Microbenchmark benchmarking group_reduce
 */
template <int Iterations = 512>
class MicroBenchGroupAnyOf {
protected:
  BenchmarkArgs args;

  PrefetchedBuffer<char, 1> output_buf;

public:
  MicroBenchGroupAnyOf(const BenchmarkArgs& _args) : args(_args) {}

  void setup() { output_buf.initialize(args.device_queue, s::range<1>(1)); }

  void run(std::vector<cl::sycl::event>& events) {
    size_t num_groups = (args.problem_size + args.local_size - 1) / args.local_size;
    events.push_back(args.device_queue.submit([&](cl::sycl::handler& cgh) {
      auto out = output_buf.get_access<s::access::mode::discard_write>(cgh);

      cgh.parallel_for<MicroBenchGroupAnyOfKernel<Iterations>>(
          s::nd_range<1>{num_groups * args.local_size, args.local_size}, [=](cl::sycl::nd_item<1> item) {
            auto g = item.get_group();
            size_t gid = item.get_global_linear_id();
            bool d = true;

            for(int i = 1; i <= Iterations; ++i) {
              d &= s::group_any_of(g, true);
            }

            if(gid == 0)
              out[0] = d;
          });
    }));
  }

  bool verify(VerificationSetting& ver) {
    auto result = output_buf.get_access<s::access::mode::read>();
    return result[0] == true;
  }

  static std::string getBenchmarkName() {
    std::stringstream name;
    name << "GroupFunctionBench_AnyOf_";
    name << Iterations;
    return name.str();
  }
};

int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);

  app.run<MicroBenchGroupAnyOf<>>();
  return 0;
}
