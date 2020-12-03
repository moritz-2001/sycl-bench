#include "common.h"

namespace s = cl::sycl;

template <typename DataT, int Iterations>
class MicroBenchGroupReduceKernel;

/**
 * Microbenchmark benchmarking group_reduce
 */
template <typename DataT, int Iterations = 512>
class MicroBenchGroupReduce {
protected:
  BenchmarkArgs args;

  PrefetchedBuffer<DataT, 1> output_buf;

public:
  MicroBenchGroupReduce(const BenchmarkArgs& _args) : args(_args) {}

  void setup() {
    output_buf.initialize(args.device_queue, s::range<1>(1));
  }

  void run(std::vector<cl::sycl::event>& events) {
    size_t num_groups = (args.problem_size + args.local_size - 1) / args.local_size;
    events.push_back(args.device_queue.submit([&](cl::sycl::handler& cgh) {
      auto out = output_buf.template get_access<s::access::mode::discard_write>(cgh);

      cgh.parallel_for<MicroBenchGroupReduceKernel<DataT, Iterations>>(
          s::nd_range<1>{num_groups * args.local_size, args.local_size}, [=](cl::sycl::nd_item<1> item) {
            auto g  = item.get_group();
            size_t gid = item.get_global_linear_id();
            DataT d = 0;

            for(int i = 1; i <= Iterations; ++i) {
              DataT j(i);
              d += s::group_reduce(g, j, [](DataT a, DataT b) { return a+b; });
            }

            if (gid == 0)
              out[0] = d;
          });
    }));
  }

  bool verify(VerificationSetting& ver) {
    auto result = output_buf.template get_access<s::access::mode::read>();
    DataT expected = DataT{(Iterations*(Iterations+1))/2} * DataT(static_cast<int>(args.local_size));

    return (result[0] == expected);
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
  app.run<MicroBenchGroupReduce<float>>();
  app.run<MicroBenchGroupReduce<double>>();
  return 0;
}
