#include "common.h"
#include<iostream>
#include<rocprim/rocprim.hpp>

namespace s = cl::sycl;

/**
 * Microbenchmark benchmarking group_reduce
 */
template <typename DataT, int Blocksize, rocprim::block_reduce_algorithm Algorithm, int Iterations = 512>
class MicroBenchGroupReduce {
protected:
  BenchmarkArgs args;

  PrefetchedBuffer<DataT, 1> output_buf;

  using BlockReduce = rocprim::block_reduce<DataT, Blocksize, Algorithm>;

public:
  MicroBenchGroupReduce(const BenchmarkArgs& _args) : args(_args) {}

  void setup() {
    output_buf.initialize(args.device_queue, s::range<1>(1));
  }

  __global__
  static void kernel() {
    DataT d = 0;

    __shared__ typename BlockReduce::storage_type temp_storage;
    DataT* scratch = reinterpret_cast<DataT*>(&temp_storage);

    for(int i = 1; i <= Iterations; ++i) {
      DataT j = i;
      BlockReduce().reduce(j, d, temp_storage, rocprim::plus<DataT>());
    }
  }

  void run(std::vector<cl::sycl::event>& events) {
    size_t num_groups = (args.problem_size + args.local_size - 1) / args.local_size;

#ifdef HIPSYCL_PLATFORM_HIP
    hipLaunchKernelGGL(kernel, dim3(num_groups), dim3(args.local_size), 0, 0);
    hipDeviceSynchronize();
#endif
  }

  bool verify(VerificationSetting& ver) {
    return true;
  }

  static std::string getBenchmarkName() {
    std::stringstream name;
    name << "GroupFunctionBench_Reduce_";
    name << ReadableTypename<DataT>::name << "_";
    switch(Algorithm){
      case rocprim::block_reduce_algorithm::raking_reduce:
        name << "raking_reduce" << "_";
        break;
      case rocprim::block_reduce_algorithm::using_warp_reduce:
        name << "using_warp_reduce" << "_";
        break;
      default:
        name << "unknown" << "_";
      break;
    }
    name << Iterations;
    return name.str();
  }
};

int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);

  size_t local_size = app.getArgs().local_size;

  switch(local_size)
  {
    case 32:
      app.run<MicroBenchGroupReduce<int, 32, rocprim::block_reduce_algorithm::raking_reduce>>();
      app.run<MicroBenchGroupReduce<int, 32, rocprim::block_reduce_algorithm::using_warp_reduce>>();
      app.run<MicroBenchGroupReduce<long long, 32, rocprim::block_reduce_algorithm::raking_reduce>>();
      app.run<MicroBenchGroupReduce<long long, 32, rocprim::block_reduce_algorithm::using_warp_reduce>>();
      app.run<MicroBenchGroupReduce<float, 32, rocprim::block_reduce_algorithm::raking_reduce>>();
      app.run<MicroBenchGroupReduce<float, 32, rocprim::block_reduce_algorithm::using_warp_reduce>>();
      app.run<MicroBenchGroupReduce<double, 32, rocprim::block_reduce_algorithm::raking_reduce>>();
      app.run<MicroBenchGroupReduce<double, 32, rocprim::block_reduce_algorithm::using_warp_reduce>>();
      break;
    case 64:
      app.run<MicroBenchGroupReduce<int, 64, rocprim::block_reduce_algorithm::raking_reduce>>();
      app.run<MicroBenchGroupReduce<int, 64, rocprim::block_reduce_algorithm::using_warp_reduce>>();
      app.run<MicroBenchGroupReduce<long long, 64, rocprim::block_reduce_algorithm::raking_reduce>>();
      app.run<MicroBenchGroupReduce<long long, 64, rocprim::block_reduce_algorithm::using_warp_reduce>>();
      app.run<MicroBenchGroupReduce<float, 64, rocprim::block_reduce_algorithm::raking_reduce>>();
      app.run<MicroBenchGroupReduce<float, 64, rocprim::block_reduce_algorithm::using_warp_reduce>>();
      app.run<MicroBenchGroupReduce<double, 64, rocprim::block_reduce_algorithm::raking_reduce>>();
      app.run<MicroBenchGroupReduce<double, 64, rocprim::block_reduce_algorithm::using_warp_reduce>>();
      break;
    case 128:
      app.run<MicroBenchGroupReduce<int, 128, rocprim::block_reduce_algorithm::raking_reduce>>();
      app.run<MicroBenchGroupReduce<int, 128, rocprim::block_reduce_algorithm::using_warp_reduce>>();
      app.run<MicroBenchGroupReduce<long long, 128, rocprim::block_reduce_algorithm::raking_reduce>>();
      app.run<MicroBenchGroupReduce<long long, 128, rocprim::block_reduce_algorithm::using_warp_reduce>>();
      app.run<MicroBenchGroupReduce<float, 128, rocprim::block_reduce_algorithm::raking_reduce>>();
      app.run<MicroBenchGroupReduce<float, 128, rocprim::block_reduce_algorithm::using_warp_reduce>>();
      app.run<MicroBenchGroupReduce<double, 128, rocprim::block_reduce_algorithm::raking_reduce>>();
      app.run<MicroBenchGroupReduce<double, 128, rocprim::block_reduce_algorithm::using_warp_reduce>>();
      break;
    case 256:
      app.run<MicroBenchGroupReduce<int, 256, rocprim::block_reduce_algorithm::raking_reduce>>();
      app.run<MicroBenchGroupReduce<int, 256, rocprim::block_reduce_algorithm::using_warp_reduce>>();
      app.run<MicroBenchGroupReduce<long long, 256, rocprim::block_reduce_algorithm::raking_reduce>>();
      app.run<MicroBenchGroupReduce<long long, 256, rocprim::block_reduce_algorithm::using_warp_reduce>>();
      app.run<MicroBenchGroupReduce<float, 256, rocprim::block_reduce_algorithm::raking_reduce>>();
      app.run<MicroBenchGroupReduce<float, 256, rocprim::block_reduce_algorithm::using_warp_reduce>>();
      app.run<MicroBenchGroupReduce<double, 256, rocprim::block_reduce_algorithm::raking_reduce>>();
      app.run<MicroBenchGroupReduce<double, 256, rocprim::block_reduce_algorithm::using_warp_reduce>>();
      break;
    case 512:
      app.run<MicroBenchGroupReduce<int, 512, rocprim::block_reduce_algorithm::raking_reduce>>();
      app.run<MicroBenchGroupReduce<int, 512, rocprim::block_reduce_algorithm::using_warp_reduce>>();
      app.run<MicroBenchGroupReduce<long long, 512, rocprim::block_reduce_algorithm::raking_reduce>>();
      app.run<MicroBenchGroupReduce<long long, 512, rocprim::block_reduce_algorithm::using_warp_reduce>>();
      app.run<MicroBenchGroupReduce<float, 512, rocprim::block_reduce_algorithm::raking_reduce>>();
      app.run<MicroBenchGroupReduce<float, 512, rocprim::block_reduce_algorithm::using_warp_reduce>>();
      app.run<MicroBenchGroupReduce<double, 512, rocprim::block_reduce_algorithm::raking_reduce>>();
      app.run<MicroBenchGroupReduce<double, 512, rocprim::block_reduce_algorithm::using_warp_reduce>>();
      break;
    case 1024:
      app.run<MicroBenchGroupReduce<int, 1024, rocprim::block_reduce_algorithm::raking_reduce>>();
      app.run<MicroBenchGroupReduce<int, 1024, rocprim::block_reduce_algorithm::using_warp_reduce>>();
      app.run<MicroBenchGroupReduce<long long, 1024, rocprim::block_reduce_algorithm::raking_reduce>>();
      app.run<MicroBenchGroupReduce<long long, 1024, rocprim::block_reduce_algorithm::using_warp_reduce>>();
      app.run<MicroBenchGroupReduce<float, 1024, rocprim::block_reduce_algorithm::raking_reduce>>();
      app.run<MicroBenchGroupReduce<float, 1024, rocprim::block_reduce_algorithm::using_warp_reduce>>();
      app.run<MicroBenchGroupReduce<double, 1024, rocprim::block_reduce_algorithm::raking_reduce>>();
      app.run<MicroBenchGroupReduce<double, 1024, rocprim::block_reduce_algorithm::using_warp_reduce>>();
      break;
    default:
      std::cout << "No valide group size given!" << std::endl;
      return -1;
  }
  return 0;
}
