#include "common.h"
#include "group_common.hpp"
#include <iostream>

namespace s = cl::sycl;

template <typename DataT, int Iterations>
class MicroBenchMatrixMultiplyKernel;

/**
 * Microbenchmark benchmarking group_reduce
 */
template <typename DataT = int, int Iterations = 50000>
class MicroBenchMatrixMultiply {
protected:
  BenchmarkArgs args;


  PrefetchedBuffer<DataT, 2> output_buf;
  PrefetchedBuffer<DataT, 2> a_buf;
  PrefetchedBuffer<DataT, 2> b_buf;

public:
  MicroBenchMatrixMultiply(const BenchmarkArgs& _args) : args(_args) {}

  void setup() {
    output_buf.initialize(args.device_queue, s::range<2>(32, 32));
    a_buf.initialize(args.device_queue, s::range<2>(32, 32));
    b_buf.initialize(args.device_queue, s::range<2>(32, 32));
    args.device_queue.submit([&](cl::sycl::handler& cgh) {
      using namespace cl::sycl::access;
      auto a_ = a_buf.template get_access<mode::write>(cgh);
      auto b_ = b_buf.template get_access<mode::write>(cgh);
      cgh.single_task([=] {
        for (auto i = 0; i < 32; ++i) {
          for (auto j = 0; j < 32; ++j) {
            a_[i][j] = 1;
            b_[i][j] = 1;
          }
        }
      });
    });
  }

  void run(std::vector<cl::sycl::event>& events) {
    size_t num_groups = (args.problem_size + args.local_size - 1) / args.local_size;
    events.push_back(args.device_queue.submit([&](cl::sycl::handler& cgh) {
      using namespace cl::sycl::access;
      auto a_ = a_buf.template get_access<mode::read>(cgh);
      auto b_ = b_buf.template get_access<mode::read>(cgh);
      auto out_ = output_buf.template get_access<mode::discard_write>(cgh);

      constexpr size_t LOCAL_SIZE = 32;

      cgh.parallel_for<MicroBenchMatrixMultiplyKernel<DataT, Iterations>>(
          s::nd_range<2>{{LOCAL_SIZE, LOCAL_SIZE}, {LOCAL_SIZE, LOCAL_SIZE}}, [=](cl::sycl::nd_item<2> item) {
                const auto sg = item.get_sub_group();
                auto m = item.get_local_id()[0];
                auto n = item.get_local_id()[1];

               const size_t SG_SIZE = sg.get_local_linear_range();

              volatile int res;
               for (auto i = 0; i < Iterations; ++i) {
                  DataT sum{};
                  for (auto kk = 0ul; kk < LOCAL_SIZE; kk += SG_SIZE) {
                      const auto tile = a_[m][kk + sg.get_local_linear_id()];
                      for (auto k = 0ul; k < SG_SIZE; ++k) {
                          sum += s::group_broadcast(sg, tile, k)  * b_[kk + k][n];
                      }
                  }
                  out_[m][n] = sum;
                  res = sum;
               }
          });
    }));
  }

  bool verify(VerificationSetting& ver) {
    auto result = output_buf.template get_access<s::access::mode::read>();
    DataT expected = initialize_type<DataT>(32);

    for (auto i = 0; i < 32; ++i) {
      for (auto j = 0; j < 32; ++j) {
        if (result[i][j] != expected) {
          return false;
        }
      }
    }
    return true;
  }

  static std::string getBenchmarkName() {
    std::stringstream name;
    //name << "GroupFunctionBench_Reduce_";
    name << ReadableTypename<DataT>::name << "_";
    name << Iterations;
    return name.str();
  }
};

int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);

  app.run<MicroBenchMatrixMultiply<int>>();
  app.run<MicroBenchMatrixMultiply<long long>>();
  app.run<MicroBenchMatrixMultiply<float>>();
  app.run<MicroBenchMatrixMultiply<double>>();
  return 0;
}
