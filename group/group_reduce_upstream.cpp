#include "common.h"
#include "group_common.h"
#include <iostream>

namespace s = cl::sycl;
template <typename DataT, int Iterations>
class MicroBenchGroupReduceKernel;

/**
 * Microbenchmark benchmarking group_reduce/aaaax
 */
template <typename DataT, bool IsVector = false, int Iterations = 1000>
class MicroBenchGroupReduce {
protected:
  BenchmarkArgs args;

  PrefetchedBuffer<DataT, 1> a_buf;
  PrefetchedBuffer<DataT, 1> b_buf;
  PrefetchedBuffer<DataT, 1> output_buf;

public:
  MicroBenchGroupReduce(const BenchmarkArgs& _args) : args(_args) {}

  void setup() {
    size_t num_groups = (args.problem_size + args.local_size - 1) / args.local_size;
    output_buf.initialize(args.device_queue, s::range<1>(1));
    a_buf.initialize(args.device_queue, s::range<1>(1024));
    b_buf.initialize(args.device_queue, s::range<1>(num_groups * args.local_size));
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
      auto b_ = b_buf.template get_access<s::access::mode::read_write>(cgh);

      cgh.parallel_for<MicroBenchGroupReduceKernel<DataT, Iterations>>(
          s::nd_range<1>{num_groups * args.local_size, args.local_size}, [=](cl::sycl::nd_item<1> item) {
            auto g = item.get_group();

            DataT d = initialize_type<DataT>(0);

           for(int i = 1; i <= Iterations; ++i) {
             const auto lid = g.get_local_linear_id();
             b_[lid] = a_[item.get_local_linear_id()];
             item.barrier();
              if (g.leader()) {
                auto result = b_[0];
                  for (auto i = 1ul; i < g.get_local_range().size(); ++i) {
                    result = s::plus<DataT>{}(result, b_[i]);
                  }
                b_[0] = result;
              }
              item.barrier();
              auto result = b_[0];
              item.barrier();

              d = result;

             if(item.get_local_linear_id() == 0)
               out[0] = d;
           }
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

  app.run<MicroBenchGroupReduce<double>>();
  return 0;
}