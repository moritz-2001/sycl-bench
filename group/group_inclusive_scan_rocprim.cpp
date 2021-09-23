#include "common.h"
#include<rocprim/rocprim.hpp>

namespace s = cl::sycl;

/**
 * Microbenchmark benchmarking group_reduce
 */
template <typename DataT, int Blocksize, rocprim::block_scan_algorithm Algorithm, int Iterations = 512>
class MicroBenchGroupInclusiveScan {
protected:
  BenchmarkArgs args;

  std::vector<DataT> results;

  using BlockScan = rocprim::block_scan<DataT, Blocksize, Algorithm>;

public:
  MicroBenchGroupInclusiveScan(const BenchmarkArgs& _args) : args(_args) {}

  void setup() {
    size_t num_groups = (args.problem_size + args.local_size - 1) / args.local_size;

    results.resize(num_groups * args.local_size);
  }

  __global__ 
  static void kernel(DataT* out) {
    DataT d = 0;
    size_t gid = hipThreadIdx_x + hipBlockIdx_x*hipBlockDim_x;

    __shared__ typename BlockScan::storage_type temp_storage;

    for(int i = 1; i <= Iterations; ++i) {
      DataT j = i;
      BlockScan().inclusive_scan(j, *(out+gid), temp_storage, rocprim::plus<DataT>());
      __syncthreads();
    }
  }

  void run(std::vector<cl::sycl::event>& events) {
    size_t num_groups = (args.problem_size + args.local_size - 1) / args.local_size;

#ifdef HIPSYCL_PLATFORM_HIP
    DataT* d_out_ptr = nullptr;
    hipMalloc(&d_out_ptr, sizeof(DataT) * num_groups * args.local_size);

    hipLaunchKernelGGL(kernel, dim3(num_groups), dim3(args.local_size), 0, 0, d_out_ptr);

    hipDeviceSynchronize();
    hipMemcpy(results.data(), d_out_ptr, sizeof(DataT) * num_groups * args.local_size, hipMemcpyDeviceToHost);
#endif
  }

  bool verify(VerificationSetting& ver) {
    DataT expected = 0;

    for(size_t i = 0; i < args.problem_size; ++i) {
      if(i%args.local_size == 0)
        expected = 0;
      expected += Iterations;
      if(results[i] != expected) {
        std::cout << i << ":" << expected << ":" << results[i] << std::endl;
        return false;
      }
    }

    return true;
  }

  static std::string getBenchmarkName() {
    std::stringstream name;
    name << "GroupFunctionBench_InclusiveScan";
    name << ReadableTypename<DataT>::name << "_";
    switch(Algorithm){
      case rocprim::block_scan_algorithm::reduce_then_scan:
        name << "reduce_then_scan" << "_";
        break;
      case rocprim::block_scan_algorithm::using_warp_scan:
        name << "using_warp_scan" << "_";
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
      app.run<MicroBenchGroupInclusiveScan<int, 32, rocprim::block_scan_algorithm::reduce_then_scan>>();
      app.run<MicroBenchGroupInclusiveScan<int, 32, rocprim::block_scan_algorithm::using_warp_scan>>();
      app.run<MicroBenchGroupInclusiveScan<long long, 32, rocprim::block_scan_algorithm::reduce_then_scan>>();
      app.run<MicroBenchGroupInclusiveScan<long long, 32, rocprim::block_scan_algorithm::using_warp_scan>>();
      app.run<MicroBenchGroupInclusiveScan<float, 32, rocprim::block_scan_algorithm::reduce_then_scan>>();
      app.run<MicroBenchGroupInclusiveScan<float, 32, rocprim::block_scan_algorithm::using_warp_scan>>();
      app.run<MicroBenchGroupInclusiveScan<double, 32, rocprim::block_scan_algorithm::reduce_then_scan>>();
      app.run<MicroBenchGroupInclusiveScan<double, 32, rocprim::block_scan_algorithm::using_warp_scan>>();
      break;
    case 64:
      app.run<MicroBenchGroupInclusiveScan<int, 64, rocprim::block_scan_algorithm::reduce_then_scan>>();
      app.run<MicroBenchGroupInclusiveScan<int, 64, rocprim::block_scan_algorithm::using_warp_scan>>();
      app.run<MicroBenchGroupInclusiveScan<long long, 64, rocprim::block_scan_algorithm::reduce_then_scan>>();
      app.run<MicroBenchGroupInclusiveScan<long long, 64, rocprim::block_scan_algorithm::using_warp_scan>>();
      app.run<MicroBenchGroupInclusiveScan<float, 64, rocprim::block_scan_algorithm::reduce_then_scan>>();
      app.run<MicroBenchGroupInclusiveScan<float, 64, rocprim::block_scan_algorithm::using_warp_scan>>();
      app.run<MicroBenchGroupInclusiveScan<double, 64, rocprim::block_scan_algorithm::reduce_then_scan>>();
      app.run<MicroBenchGroupInclusiveScan<double, 64, rocprim::block_scan_algorithm::using_warp_scan>>();
      break;
    case 128:
      app.run<MicroBenchGroupInclusiveScan<int, 128, rocprim::block_scan_algorithm::reduce_then_scan>>();
      app.run<MicroBenchGroupInclusiveScan<int, 128, rocprim::block_scan_algorithm::using_warp_scan>>();
      app.run<MicroBenchGroupInclusiveScan<long long, 128, rocprim::block_scan_algorithm::reduce_then_scan>>();
      app.run<MicroBenchGroupInclusiveScan<long long, 128, rocprim::block_scan_algorithm::using_warp_scan>>();
      app.run<MicroBenchGroupInclusiveScan<float, 128, rocprim::block_scan_algorithm::reduce_then_scan>>();
      app.run<MicroBenchGroupInclusiveScan<float, 128, rocprim::block_scan_algorithm::using_warp_scan>>();
      app.run<MicroBenchGroupInclusiveScan<double, 128, rocprim::block_scan_algorithm::reduce_then_scan>>();
      app.run<MicroBenchGroupInclusiveScan<double, 128, rocprim::block_scan_algorithm::using_warp_scan>>();
      break;
    case 256:
      app.run<MicroBenchGroupInclusiveScan<int, 256, rocprim::block_scan_algorithm::reduce_then_scan>>();
      app.run<MicroBenchGroupInclusiveScan<int, 256, rocprim::block_scan_algorithm::using_warp_scan>>();
      app.run<MicroBenchGroupInclusiveScan<long long, 256, rocprim::block_scan_algorithm::reduce_then_scan>>();
      app.run<MicroBenchGroupInclusiveScan<long long, 256, rocprim::block_scan_algorithm::using_warp_scan>>();
      app.run<MicroBenchGroupInclusiveScan<float, 256, rocprim::block_scan_algorithm::reduce_then_scan>>();
      app.run<MicroBenchGroupInclusiveScan<float, 256, rocprim::block_scan_algorithm::using_warp_scan>>();
      app.run<MicroBenchGroupInclusiveScan<double, 256, rocprim::block_scan_algorithm::reduce_then_scan>>();
      app.run<MicroBenchGroupInclusiveScan<double, 256, rocprim::block_scan_algorithm::using_warp_scan>>();
      break;
    case 512:
      app.run<MicroBenchGroupInclusiveScan<int, 512, rocprim::block_scan_algorithm::reduce_then_scan>>();
      app.run<MicroBenchGroupInclusiveScan<int, 512, rocprim::block_scan_algorithm::using_warp_scan>>();
      app.run<MicroBenchGroupInclusiveScan<long long, 512, rocprim::block_scan_algorithm::reduce_then_scan>>();
      app.run<MicroBenchGroupInclusiveScan<long long, 512, rocprim::block_scan_algorithm::using_warp_scan>>();
      app.run<MicroBenchGroupInclusiveScan<float, 512, rocprim::block_scan_algorithm::reduce_then_scan>>();
      app.run<MicroBenchGroupInclusiveScan<float, 512, rocprim::block_scan_algorithm::using_warp_scan>>();
      app.run<MicroBenchGroupInclusiveScan<double, 512, rocprim::block_scan_algorithm::reduce_then_scan>>();
      app.run<MicroBenchGroupInclusiveScan<double, 512, rocprim::block_scan_algorithm::using_warp_scan>>();
      break;
    case 1024:
      app.run<MicroBenchGroupInclusiveScan<int, 1024, rocprim::block_scan_algorithm::reduce_then_scan>>();
      app.run<MicroBenchGroupInclusiveScan<int, 1024, rocprim::block_scan_algorithm::using_warp_scan>>();
      app.run<MicroBenchGroupInclusiveScan<long long, 1024, rocprim::block_scan_algorithm::reduce_then_scan>>();
      app.run<MicroBenchGroupInclusiveScan<long long, 1024, rocprim::block_scan_algorithm::using_warp_scan>>();
      app.run<MicroBenchGroupInclusiveScan<float, 1024, rocprim::block_scan_algorithm::reduce_then_scan>>();
      app.run<MicroBenchGroupInclusiveScan<float, 1024, rocprim::block_scan_algorithm::using_warp_scan>>();
      app.run<MicroBenchGroupInclusiveScan<double, 1024, rocprim::block_scan_algorithm::reduce_then_scan>>();
      app.run<MicroBenchGroupInclusiveScan<double, 1024, rocprim::block_scan_algorithm::using_warp_scan>>();
      break;
    default:
      std::cout << "No valide group size given!" << std::endl;
      return -1;
  }
  return 0;
}
