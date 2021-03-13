#include "common.h"
#include<cub/cub.cuh>

namespace s = cl::sycl;

template <typename DataT, int Iterations, int Blocksize>
class MicroBenchGroupInclusiveScanKernel;

/**
 * Microbenchmark benchmarking group_reduce
 */
template <typename DataT, int Blocksize, cub::BlockScanAlgorithm Algorithm, int Iterations = 512>
class MicroBenchGroupInclusiveScan {
protected:
  BenchmarkArgs args;

  PrefetchedBuffer<DataT, 1> output_buf;

  using BlockScan = cub::BlockScan<DataT, Blocksize, Algorithm>;

public:
  MicroBenchGroupInclusiveScan(const BenchmarkArgs& _args) : args(_args) {}

  void setup() {
    size_t num_groups = (args.problem_size + args.local_size - 1) / args.local_size;

    output_buf.initialize(args.device_queue, s::range<1>(num_groups * args.local_size));
  }

  __global__ 
  static void kernel() {
    DataT d = 0;

    __shared__ typename BlockScan::TempStorage temp_storage;
    DataT* scratch = reinterpret_cast<DataT*>(&temp_storage);

    for(int i = 1; i <= Iterations; ++i) {
      DataT j = 1;
      BlockScan(temp_storage).InclusiveSum(j, d);
    }
    __syncthreads();

    //printf("%d: %d\n", threadIdx.x, d);
  }

  void run(std::vector<cl::sycl::event>& events) {
    size_t num_groups = (args.problem_size + args.local_size - 1) / args.local_size;
#ifdef HIPSYCL_PLATFORM_CUDA
    kernel<<<num_groups, args.local_size>>>();
    cudaDeviceSynchronize();
#endif
  }

  bool verify(VerificationSetting& ver) {
    return true;
  }

  static std::string getBenchmarkName() {
    std::stringstream name;
    name << "GroupFunctionBench_InclusiveScan";
    name << ReadableTypename<DataT>::name << "_";
    name << Algorithm <<"_";
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
      app.run<MicroBenchGroupInclusiveScan<int, 32, cub::BLOCK_SCAN_RAKING>>();
      app.run<MicroBenchGroupInclusiveScan<int, 32, cub::BLOCK_SCAN_WARP_SCANS>>();
      app.run<MicroBenchGroupInclusiveScan<long long, 32, cub::BLOCK_SCAN_RAKING>>();
      app.run<MicroBenchGroupInclusiveScan<long long, 32, cub::BLOCK_SCAN_WARP_SCANS>>();
      app.run<MicroBenchGroupInclusiveScan<float, 32, cub::BLOCK_SCAN_RAKING>>();
      app.run<MicroBenchGroupInclusiveScan<float, 32, cub::BLOCK_SCAN_WARP_SCANS>>();
      app.run<MicroBenchGroupInclusiveScan<double, 32, cub::BLOCK_SCAN_RAKING>>();
      app.run<MicroBenchGroupInclusiveScan<double, 32, cub::BLOCK_SCAN_WARP_SCANS>>();
      break;
    case 64:
      app.run<MicroBenchGroupInclusiveScan<int, 64, cub::BLOCK_SCAN_RAKING>>();
      app.run<MicroBenchGroupInclusiveScan<int, 64, cub::BLOCK_SCAN_WARP_SCANS>>();
      app.run<MicroBenchGroupInclusiveScan<long long, 64, cub::BLOCK_SCAN_RAKING>>();
      app.run<MicroBenchGroupInclusiveScan<long long, 64, cub::BLOCK_SCAN_WARP_SCANS>>();
      app.run<MicroBenchGroupInclusiveScan<float, 64, cub::BLOCK_SCAN_RAKING>>();
      app.run<MicroBenchGroupInclusiveScan<float, 64, cub::BLOCK_SCAN_WARP_SCANS>>();
      app.run<MicroBenchGroupInclusiveScan<double, 64, cub::BLOCK_SCAN_RAKING>>();
      app.run<MicroBenchGroupInclusiveScan<double, 64, cub::BLOCK_SCAN_WARP_SCANS>>();
      break;
    case 128:
      app.run<MicroBenchGroupInclusiveScan<int, 128, cub::BLOCK_SCAN_RAKING>>();
      app.run<MicroBenchGroupInclusiveScan<int, 128, cub::BLOCK_SCAN_WARP_SCANS>>();
      app.run<MicroBenchGroupInclusiveScan<long long, 128, cub::BLOCK_SCAN_RAKING>>();
      app.run<MicroBenchGroupInclusiveScan<long long, 128, cub::BLOCK_SCAN_WARP_SCANS>>();
      app.run<MicroBenchGroupInclusiveScan<float, 128, cub::BLOCK_SCAN_RAKING>>();
      app.run<MicroBenchGroupInclusiveScan<float, 128, cub::BLOCK_SCAN_WARP_SCANS>>();
      app.run<MicroBenchGroupInclusiveScan<double, 128, cub::BLOCK_SCAN_RAKING>>();
      app.run<MicroBenchGroupInclusiveScan<double, 128, cub::BLOCK_SCAN_WARP_SCANS>>();
      break;
    case 256:
      app.run<MicroBenchGroupInclusiveScan<int, 256, cub::BLOCK_SCAN_RAKING>>();
      app.run<MicroBenchGroupInclusiveScan<int, 256, cub::BLOCK_SCAN_WARP_SCANS>>();
      app.run<MicroBenchGroupInclusiveScan<long long, 256, cub::BLOCK_SCAN_RAKING>>();
      app.run<MicroBenchGroupInclusiveScan<long long, 256, cub::BLOCK_SCAN_WARP_SCANS>>();
      app.run<MicroBenchGroupInclusiveScan<float, 256, cub::BLOCK_SCAN_RAKING>>();
      app.run<MicroBenchGroupInclusiveScan<float, 256, cub::BLOCK_SCAN_WARP_SCANS>>();
      app.run<MicroBenchGroupInclusiveScan<double, 256, cub::BLOCK_SCAN_RAKING>>();
      app.run<MicroBenchGroupInclusiveScan<double, 256, cub::BLOCK_SCAN_WARP_SCANS>>();
      break;
    case 512:
      app.run<MicroBenchGroupInclusiveScan<int, 512, cub::BLOCK_SCAN_RAKING>>();
      app.run<MicroBenchGroupInclusiveScan<int, 512, cub::BLOCK_SCAN_WARP_SCANS>>();
      app.run<MicroBenchGroupInclusiveScan<long long, 512, cub::BLOCK_SCAN_RAKING>>();
      app.run<MicroBenchGroupInclusiveScan<long long, 512, cub::BLOCK_SCAN_WARP_SCANS>>();
      app.run<MicroBenchGroupInclusiveScan<float, 512, cub::BLOCK_SCAN_RAKING>>();
      app.run<MicroBenchGroupInclusiveScan<float, 512, cub::BLOCK_SCAN_WARP_SCANS>>();
      app.run<MicroBenchGroupInclusiveScan<double, 512, cub::BLOCK_SCAN_RAKING>>();
      app.run<MicroBenchGroupInclusiveScan<double, 512, cub::BLOCK_SCAN_WARP_SCANS>>();
      break;
    case 1024:
      app.run<MicroBenchGroupInclusiveScan<int, 1024, cub::BLOCK_SCAN_RAKING>>();
      app.run<MicroBenchGroupInclusiveScan<int, 1024, cub::BLOCK_SCAN_WARP_SCANS>>();
      app.run<MicroBenchGroupInclusiveScan<long long, 1024, cub::BLOCK_SCAN_RAKING>>();
      app.run<MicroBenchGroupInclusiveScan<long long, 1024, cub::BLOCK_SCAN_WARP_SCANS>>();
      app.run<MicroBenchGroupInclusiveScan<float, 1024, cub::BLOCK_SCAN_RAKING>>();
      app.run<MicroBenchGroupInclusiveScan<float, 1024, cub::BLOCK_SCAN_WARP_SCANS>>();
      app.run<MicroBenchGroupInclusiveScan<double, 1024, cub::BLOCK_SCAN_RAKING>>();
      app.run<MicroBenchGroupInclusiveScan<double, 1024, cub::BLOCK_SCAN_WARP_SCANS>>();
      break;
    default:
      std::cout << "No valide group size given!" << std::endl;
      return -1;
  }
  return 0;
}
