#include "common.h"
#include "group_common.hpp"
#include <iostream>

namespace s = cl::sycl;

template <typename DataT, int Iterations>
class MicroBenchGroupReduceKernel;

/**
 * Microbenchmark benchmarking group_reduce/aaaax
 */
template <typename DataT, bool IsVector = false, int Iterations = 10000>
class MicroBenchGroupReduce {
protected:
  BenchmarkArgs args;

  PrefetchedBuffer<DataT, 1> a_buf;
  PrefetchedBuffer<DataT, 1> output_buf;

public:
  MicroBenchGroupReduce(const BenchmarkArgs& _args) : args(_args) {}

  void setup() {
    output_buf.initialize(args.device_queue, s::range<1>(1));
    a_buf.initialize(args.device_queue, s::range<1>(1024));
      using namespace cl::sycl::access;
        for (auto i = 0; i < 1024; ++i) {
          a_buf.template get_access<mode::write>()[i] = initialize_type<DataT>(i);
        }
  }

  void run(std::vector<cl::sycl::event>& events) {
    size_t num_groups = (args.problem_size + args.local_size - 1) / args.local_size;
    events.push_back(args.device_queue.submit([&](cl::sycl::handler& cgh) {
      auto out = output_buf.template get_access<s::access::mode::discard_write>(cgh);
      auto a_ = a_buf.template get_access<s::access::mode::read>(cgh);

      cgh.parallel_for<MicroBenchGroupReduceKernel<DataT, Iterations>>(
          s::nd_range<1>{num_groups * args.local_size, args.local_size}, [=](cl::sycl::nd_item<1> item) {
            auto g = item.get_group();
            auto sg = item.get_sub_group();
            size_t gid = item.get_global_linear_id();
            DataT d = initialize_type<DataT>(0);

            for(int i = 1; i <= Iterations; ++i) {
              d = s::group_reduce(g, a_[item.get_local_linear_id()], s::plus<DataT>());
            }

            if(gid == 0)
              out[0] = d;
          });
    }));
  }

  bool verify(VerificationSetting& ver) {
    auto result = output_buf.template get_access<s::access::mode::read>();
    auto a_ = a_buf.template get_access<s::access::mode::read>();
    DataT res = initialize_type<DataT>(0);
    for (auto i = 0ul; i < args.local_size; ++i) {
      res = s::plus<DataT>{}(res, initialize_type<DataT>(i));
    }

    if(!compare_type(result[0], res))
      std::cout << type_to_string(res) << ":" << type_to_string(result[0]) << std::endl;
    return compare_type(result[0], res);
  }

  static std::string getBenchmarkName() {
    std::stringstream name;
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
  return 0;
}
