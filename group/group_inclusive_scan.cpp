#include "common.h"

namespace s = cl::sycl;

template <typename DataT, int Iterations>
class MicroBenchGroupInclusiveScanKernel;

template <typename T>
using elementType = std::remove_reference_t<decltype(T{}.s0())>;

template <typename T, int N>
std::string type_to_string(s::vec<T, N> v) {
  std::stringstream ss{};

  ss << "(";
  if constexpr(1 <= N)
    ss << +v.s0();
  if constexpr(2 <= N)
    ss << ", " << +v.s1();
  if constexpr(3 <= N)
    ss << ", " << +v.s2();
  if constexpr(4 <= N)
    ss << ", " << +v.s3();
  if constexpr(8 <= N) {
    ss << ", " << +v.s4();
    ss << ", " << +v.s5();
    ss << ", " << +v.s6();
    ss << ", " << +v.s7();
  }
  if constexpr(16 <= N) {
    ss << ", " << +v.s8();
    ss << ", " << +v.s9();
    ss << ", " << +v.sA();
    ss << ", " << +v.sB();
    ss << ", " << +v.sC();
    ss << ", " << +v.sD();
    ss << ", " << +v.sE();
    ss << ", " << +v.sF();
  }
  ss << ")";

  return ss.str();
}

template <typename T>
std::string type_to_string(T x) {
  std::stringstream ss{};
  ss << +x;

  return ss.str();
}

template <typename T, typename std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
HIPSYCL_KERNEL_TARGET T initialize_type(T init) {
  return init;
}

template <typename T, typename std::enable_if_t<!std::is_arithmetic_v<T>, int> = 0>
HIPSYCL_KERNEL_TARGET T initialize_type(elementType<T> init) {
  constexpr size_t N = T::get_count();

  if constexpr(std::is_same_v<elementType<T>, bool>)
    return T{init};

  if constexpr(N == 1) {
    return T{init};
  } else if constexpr(N == 2) {
    return T{init, init};
  } else if constexpr(N == 3) {
    return T{init, init, init};
  } else if constexpr(N == 4) {
    return T{init, init, init, init};
  } else if constexpr(N == 8) {
    return T{init, init, init, init, init, init, init, init};
  } else if constexpr(N == 16) {
    return T{init, init, init, init, init, init, init, init, init, init, init, init, init, init, init, init};
  }

  static_assert(true, "invalide vector type!");
}

template <typename T, int N>
bool compare_type(s::vec<T, N> v1, s::vec<T, N> v2) {
  bool ret = true;
  if constexpr(1 <= N)
    ret &= v1.s0() == v2.s0();
  if constexpr(2 <= N)
    ret &= v1.s1() == v2.s1();
  if constexpr(3 <= N)
    ret &= v1.s2() == v2.s2();
  if constexpr(4 <= N)
    ret &= v1.s3() == v2.s3();
  if constexpr(8 <= N) {
    ret &= v1.s4() == v2.s4();
    ret &= v1.s5() == v2.s5();
    ret &= v1.s6() == v2.s6();
    ret &= v1.s7() == v2.s7();
  }
  if constexpr(16 <= N) {
    ret &= v1.s8() == v2.s8();
    ret &= v1.s9() == v2.s9();
    ret &= v1.sA() == v2.sA();
    ret &= v1.sB() == v2.sB();
    ret &= v1.sC() == v2.sC();
    ret &= v1.sD() == v2.sD();
    ret &= v1.sE() == v2.sE();
    ret &= v1.sF() == v2.sF();
  }

  return ret;
}

template <typename T>
bool compare_type(T x1, T x2) {
  return (x1 - x2 < 1.0) && (x2 - x1) < 1.0;
}

/**
 * Microbenchmark benchmarking group_reduce
 */
template <typename DataT, int Iterations = 512>
class MicroBenchGroupInclusiveScan {
protected:
  BenchmarkArgs args;

  PrefetchedBuffer<DataT, 1> output_buf;

public:
  MicroBenchGroupInclusiveScan(const BenchmarkArgs& _args) : args(_args) {}

  void setup() {
    size_t num_groups = (args.problem_size + args.local_size - 1) / args.local_size;

    output_buf.initialize(args.device_queue, s::range<1>(num_groups * args.local_size));
  }

  void run(std::vector<cl::sycl::event>& events) {
    size_t num_groups = (args.problem_size + args.local_size - 1) / args.local_size;
    events.push_back(args.device_queue.submit([&](cl::sycl::handler& cgh) {
      auto out = output_buf.template get_access<s::access::mode::discard_write>(cgh);

      cgh.parallel_for<MicroBenchGroupInclusiveScanKernel<DataT, Iterations>>(
          s::nd_range<1>{num_groups * args.local_size, args.local_size}, [=](cl::sycl::nd_item<1> item) {
            auto g = item.get_group();
            size_t gid = item.get_global_linear_id();
            DataT d = initialize_type<DataT>(0);

            for(int i = 1; i <= Iterations; ++i) {
              DataT j = initialize_type<DataT>(i);
              d = s::group_inclusive_scan(g, j, [](DataT a, DataT b) { return a + b; });
            }

            out[gid] = d;
          });
    }));
  }

  bool verify(VerificationSetting& ver) {
    auto result = output_buf.template get_access<s::access::mode::read>();
    DataT expected = initialize_type<DataT>(0);

    for(size_t i = 0; i < args.problem_size; ++i) {
      if(i % args.local_size == 0)
        expected = 0;
      expected += initialize_type<DataT>(Iterations);
      if(!compare_type(result[i], expected)) {
        std::cout << i << ":" << type_to_string(expected) << ":" << type_to_string(result[i]) << std::endl;
        return false;
      }
    }

    return true;
  }

  static std::string getBenchmarkName() {
    std::stringstream name;
    name << "GroupFunctionBench_InclusiveScan";
    name << ReadableTypename<DataT>::name << "_";
    name << Iterations;
    return name.str();
  }
};

int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);

  app.run<MicroBenchGroupInclusiveScan<int>>();
  app.run<MicroBenchGroupInclusiveScan<long long>>();
  app.run<MicroBenchGroupInclusiveScan<float>>();
  app.run<MicroBenchGroupInclusiveScan<double>>();
  app.run<MicroBenchGroupInclusiveScan<cl::sycl::vec<int, 1>>>();
  app.run<MicroBenchGroupInclusiveScan<cl::sycl::vec<int, 4>>>();
  app.run<MicroBenchGroupInclusiveScan<cl::sycl::vec<int, 8>>>();
  app.run<MicroBenchGroupInclusiveScan<cl::sycl::vec<float, 1>>>();
  app.run<MicroBenchGroupInclusiveScan<cl::sycl::vec<double, 2>>>();
  app.run<MicroBenchGroupInclusiveScan<cl::sycl::vec<float, 4>>>();
  app.run<MicroBenchGroupInclusiveScan<cl::sycl::vec<float, 8>>>();
  return 0;
}
