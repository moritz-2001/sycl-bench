#include "common.h"
#include <cub/cub.cuh>
#include <iostream>

namespace s = cl::sycl;
/**
 * Microbenchmark benchmarking group_reduce
 */
template <typename DataT, int Blocksize, cub::BlockReduceAlgorithm Algorithm, int Iterations = 512>
class MicroBenchGroupReduce {
protected:
  BenchmarkArgs args;

  DataT result;

  using BlockReduce = cub::BlockReduce<DataT, Blocksize, Algorithm>;

public:
  MicroBenchGroupReduce(const BenchmarkArgs& _args) : args(_args) {}

  void setup() {}

  __global__ static void kernel(DataT* out) {
    DataT d = 0;

    __shared__ typename BlockReduce::TempStorage temp_storage;
    DataT* scratch = reinterpret_cast<DataT*>(&temp_storage);

    for(int i = 1; i <= Iterations; ++i) {
      DataT j = 1;
      d = BlockReduce(temp_storage).Sum(j);
    }

    if(threadIdx.x == 0)
      scratch[0] = d;
    __syncthreads();

    if(blockIdx.x * blockDim.x + threadIdx.x == 0)
      *out = scratch[0];
  }

  void run(std::vector<cl::sycl::event>& events) {
    size_t num_groups = (args.problem_size + args.local_size - 1) / args.local_size;

#ifdef HIPSYCL_PLATFORM_CUDA
    DataT* d_out_ptr = nullptr;
    cudaMalloc(&d_out_ptr, sizeof(DataT));

    kernel<<<num_groups, args.local_size>>>(d_out_ptr);

    cudaDeviceSynchronize();
    cudaMemcpy(&result, d_out_ptr, sizeof(DataT), cudaMemcpyDeviceToHost);
#endif
  }

  bool verify(VerificationSetting& ver) {
    DataT expected = args.local_size;

    if(result != expected)
      std::cout << expected << ":" << result << std::endl;
    return result == expected;
  }

  static std::string getBenchmarkName() {
    std::stringstream name;
    name << "GroupFunctionBench_Reduce_";
    name << ReadableTypename<DataT>::name << "_";
    name << Algorithm << "_";
    name << Iterations;
    return name.str();
  }
};

int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);

  size_t local_size = app.getArgs().local_size;

  switch(local_size) {
  case 32:
    app.run<MicroBenchGroupReduce<int, 32, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>>();
    app.run<MicroBenchGroupReduce<int, 32, cub::BLOCK_REDUCE_WARP_REDUCTIONS>>();
    app.run<MicroBenchGroupReduce<long long, 32, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>>();
    app.run<MicroBenchGroupReduce<long long, 32, cub::BLOCK_REDUCE_WARP_REDUCTIONS>>();
    app.run<MicroBenchGroupReduce<float, 32, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>>();
    app.run<MicroBenchGroupReduce<float, 32, cub::BLOCK_REDUCE_WARP_REDUCTIONS>>();
    app.run<MicroBenchGroupReduce<double, 32, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>>();
    app.run<MicroBenchGroupReduce<double, 32, cub::BLOCK_REDUCE_WARP_REDUCTIONS>>();
    break;
  case 64:
    app.run<MicroBenchGroupReduce<int, 64, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>>();
    app.run<MicroBenchGroupReduce<int, 64, cub::BLOCK_REDUCE_WARP_REDUCTIONS>>();
    app.run<MicroBenchGroupReduce<long long, 64, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>>();
    app.run<MicroBenchGroupReduce<long long, 64, cub::BLOCK_REDUCE_WARP_REDUCTIONS>>();
    app.run<MicroBenchGroupReduce<float, 64, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>>();
    app.run<MicroBenchGroupReduce<float, 64, cub::BLOCK_REDUCE_WARP_REDUCTIONS>>();
    app.run<MicroBenchGroupReduce<double, 64, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>>();
    app.run<MicroBenchGroupReduce<double, 64, cub::BLOCK_REDUCE_WARP_REDUCTIONS>>();
    break;
  case 128:
    app.run<MicroBenchGroupReduce<int, 128, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>>();
    app.run<MicroBenchGroupReduce<int, 128, cub::BLOCK_REDUCE_WARP_REDUCTIONS>>();
    app.run<MicroBenchGroupReduce<long long, 128, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>>();
    app.run<MicroBenchGroupReduce<long long, 128, cub::BLOCK_REDUCE_WARP_REDUCTIONS>>();
    app.run<MicroBenchGroupReduce<float, 128, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>>();
    app.run<MicroBenchGroupReduce<float, 128, cub::BLOCK_REDUCE_WARP_REDUCTIONS>>();
    app.run<MicroBenchGroupReduce<double, 128, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>>();
    app.run<MicroBenchGroupReduce<double, 128, cub::BLOCK_REDUCE_WARP_REDUCTIONS>>();
    break;
  case 256:
    app.run<MicroBenchGroupReduce<int, 256, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>>();
    app.run<MicroBenchGroupReduce<int, 256, cub::BLOCK_REDUCE_WARP_REDUCTIONS>>();
    app.run<MicroBenchGroupReduce<long long, 256, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>>();
    app.run<MicroBenchGroupReduce<long long, 256, cub::BLOCK_REDUCE_WARP_REDUCTIONS>>();
    app.run<MicroBenchGroupReduce<float, 256, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>>();
    app.run<MicroBenchGroupReduce<float, 256, cub::BLOCK_REDUCE_WARP_REDUCTIONS>>();
    app.run<MicroBenchGroupReduce<double, 256, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>>();
    app.run<MicroBenchGroupReduce<double, 256, cub::BLOCK_REDUCE_WARP_REDUCTIONS>>();
    break;
  case 512:
    app.run<MicroBenchGroupReduce<int, 512, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>>();
    app.run<MicroBenchGroupReduce<int, 512, cub::BLOCK_REDUCE_WARP_REDUCTIONS>>();
    app.run<MicroBenchGroupReduce<long long, 512, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>>();
    app.run<MicroBenchGroupReduce<long long, 512, cub::BLOCK_REDUCE_WARP_REDUCTIONS>>();
    app.run<MicroBenchGroupReduce<float, 512, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>>();
    app.run<MicroBenchGroupReduce<float, 512, cub::BLOCK_REDUCE_WARP_REDUCTIONS>>();
    app.run<MicroBenchGroupReduce<double, 512, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>>();
    app.run<MicroBenchGroupReduce<double, 512, cub::BLOCK_REDUCE_WARP_REDUCTIONS>>();
    break;
  case 1024:
    app.run<MicroBenchGroupReduce<int, 1024, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>>();
    app.run<MicroBenchGroupReduce<int, 1024, cub::BLOCK_REDUCE_WARP_REDUCTIONS>>();
    app.run<MicroBenchGroupReduce<long long, 1024, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>>();
    app.run<MicroBenchGroupReduce<long long, 1024, cub::BLOCK_REDUCE_WARP_REDUCTIONS>>();
    app.run<MicroBenchGroupReduce<float, 1024, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>>();
    app.run<MicroBenchGroupReduce<float, 1024, cub::BLOCK_REDUCE_WARP_REDUCTIONS>>();
    app.run<MicroBenchGroupReduce<double, 1024, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>>();
    app.run<MicroBenchGroupReduce<double, 1024, cub::BLOCK_REDUCE_WARP_REDUCTIONS>>();
    break;
  default:
    std::cout << "No valide group size given!" << std::endl;
    return -1;
  }
  return 0;
}
