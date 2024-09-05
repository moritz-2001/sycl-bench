#include "../include/common.h"
#include "group_common.hpp"
#include <iostream>

namespace s = cl::sycl;

template <typename DataT, size_t Iterations>
class MicroBenchShuffleKernel;

template <typename DataT, size_t Iterations = 1000000>
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
    auto SgSize = args.device_queue.get_device().get_info<s::info::device::sub_group_sizes>().at(0);
    events.push_back(args.device_queue.submit([&](cl::sycl::handler& cgh) {
      auto out = output_buf.template get_access<s::access::mode::discard_write>(cgh);
      auto a_ = a_buf.template get_access<s::access::mode::read>(cgh);

      cgh.parallel_for<MicroBenchShuffleKernel<DataT, Iterations>>(
          s::nd_range<1>{num_groups * args.local_size, args.local_size}, [=](cl::sycl::nd_item<1> item) {
            auto sg = item.get_sub_group();
            volatile DataT d;
             for(size_t i = 0; i < Iterations; ++i) {
                d = s::inclusive_scan_over_group(sg, a_[sg.get_local_linear_id()], s::plus<DataT>());
             }
            if (item.get_group().get_local_linear_id() == SgSize-1) {
              out[0] = d;
            }
          });
    }));
  }

  bool verify(VerificationSetting& ver) {
    auto SgSize = args.device_queue.get_device().get_info<s::info::device::sub_group_sizes>().at(0);
    auto result = output_buf.template get_access<s::access::mode::read>();
    auto res = (SgSize-1)*SgSize / 2;
    DataT expected = initialize_type<DataT>(res);

    if(!compare_type(result[0], expected))
      std::cout << type_to_string(expected) << ":" << type_to_string(result[0]) << std::endl;
    return compare_type(result[0], expected);
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

  app.run<MicroBenchShuffle<uint8_t>>();
  app.run<MicroBenchShuffle<int>>();
  app.run<MicroBenchShuffle<long long>>();
  app.run<MicroBenchShuffle<float>>();
  app.run<MicroBenchShuffle<double>>();
  return 0;
}
