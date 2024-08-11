#include "../include/common.h"
#include "group_common.hpp"
#include <iostream>

namespace s = cl::sycl;

template <typename DataT, size_t Iterations>
class MicroBenchVoteAllKernel;

template <typename DataT, size_t Iterations = 200000000>
class MicroBenchVoteAll {
protected:
  BenchmarkArgs args;

  PrefetchedBuffer<DataT, 1> output_buf;

public:
  explicit MicroBenchVoteAll(const BenchmarkArgs& _args) : args(_args) {}

  void setup() { output_buf.initialize(args.device_queue, s::range<1>(1)); }

  void run(std::vector<cl::sycl::event>& events) {
    size_t num_groups = (args.problem_size + args.local_size - 1) / args.local_size;
    events.push_back(args.device_queue.submit([&](cl::sycl::handler& cgh) {
      auto out = output_buf.template get_access<s::access::mode::discard_write>(cgh);

      cgh.parallel_for<MicroBenchVoteAllKernel<DataT, Iterations>>(
          s::nd_range<1>{num_groups * args.local_size, args.local_size}, [=](cl::sycl::nd_item<1> item) {
            auto g = item.get_group();
            auto sg = item.get_sub_group();
            DataT d;
              for(size_t i = 0; i < Iterations; ++i) {
                DataT x = initialize_type<DataT>(sg.get_local_linear_id() < sg.get_local_linear_range());
                d = s::all_of_group(sg, x);
              }

            if (g.leader())
              out[0] = d;
          });
    }));
  }

  bool verify(VerificationSetting& ver) {
    auto result = output_buf.template get_access<s::access::mode::read>();
    DataT expected = initialize_type<DataT>(true);

    if(!compare_type(result[0], expected))
      std::cout << type_to_string(expected) << ":" << type_to_string(result[0]) << std::endl;
    return compare_type(result[0], expected);
  }

  static std::string getBenchmarkName() {
    std::stringstream name;
    name << "VoteAll";
    //name << ReadableTypename<DataT>::name << "_";
    name << Iterations;
    return name.str();
  }
};

int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);

  app.run<MicroBenchVoteAll<bool>>();
  //app.run<MicroBenchVoteAll<Big<0>>>();
  return 0;
}
