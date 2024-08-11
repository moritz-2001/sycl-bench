#include "../include/common.h"
#include "group_common.hpp"
#include <iostream>

namespace s = cl::sycl;

template <typename DataT, size_t Iterations>
class MicroBenchShuffleKernel;

template <typename DataT, size_t Iterations = 200000>
class MicroBenchShuffle {
protected:
  BenchmarkArgs args;

  PrefetchedBuffer<DataT, 1> a_buf;
  PrefetchedBuffer<DataT, 1> output_buf;

public:
  explicit MicroBenchShuffle(const BenchmarkArgs& _args) : args(_args) {}

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

      cgh.parallel_for<MicroBenchShuffleKernel<DataT, Iterations>>(
          s::nd_range<1>{num_groups * args.local_size, args.local_size}, [=](cl::sycl::nd_item<1> item) {
            auto g = item.get_group();
            auto sg = item.get_sub_group();
            DataT d;
             for(size_t i = 0; i < Iterations; ++i) {
                d = s::select_from_group(sg, a_[item.get_local_linear_id()], sg.get_local_linear_id()+2);
             }
            if (g.leader())
              out[0] = d;
          });
    }));
  }

  bool verify(VerificationSetting& ver) {
    auto result = output_buf.template get_access<s::access::mode::read>();
    DataT expected = initialize_type<DataT>(2);

    if(!compare_type(result[0], expected))
      std::cout << type_to_string(expected) << ":" << type_to_string(result[0]) << std::endl;
    return compare_type(result[0], expected);
  }

  static std::string getBenchmarkName() {
    std::stringstream name;
    name << "Shuffle";
    //name << ReadableTypename<DataT>::name << "_";
    name << Iterations;
    return name.str();
  }
};

int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);

  //app.run<MicroBenchShuffle<double>>();
  app.run<MicroBenchShuffle<int>>();///////
  //app.run<MicroBenchShuffle<Big<8>>>();//
  return 0;
}
