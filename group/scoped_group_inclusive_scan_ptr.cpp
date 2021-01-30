#include "common.h"
#include<iostream>

namespace s = cl::sycl;

template <typename DataT, int Iterations>
class ScopedMicroBenchGroupInclusiveScanPtrKernel;

template<typename T>
using elementType = std::remove_reference_t<decltype(T{}.s0())>;

template<typename T, int N>
std::string type_to_string(s::vec<T, N> v) {
  std::stringstream ss{};

  ss << "(";
  if constexpr (1 <= N)
    ss << +v.s0();
  if constexpr (2 <= N)
    ss << ", " << +v.s1();
  if constexpr (3 <= N)
    ss << ", " << +v.s2();
  if constexpr (4 <= N)
    ss << ", " << +v.s3();
  if constexpr (8 <= N) {
    ss << ", " << +v.s4();
    ss << ", " << +v.s5();
    ss << ", " << +v.s6();
    ss << ", " << +v.s7();
  }
  if constexpr (16 <= N) {
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

template<typename T>
std::string type_to_string(T x) {
  std::stringstream ss{};
  ss << +x;

  return ss.str();
}

template<typename T, typename std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
HIPSYCL_KERNEL_TARGET
T initialize_type(T init) {
  return init;
}

template<typename T, typename std::enable_if_t<!std::is_arithmetic_v<T>, int> = 0>
HIPSYCL_KERNEL_TARGET
T initialize_type(elementType<T> init) {
  constexpr size_t N = T::get_count();

  if constexpr (std::is_same_v<elementType<T>, bool>)
    return T{init};

  if constexpr (N == 1) {
    return T{init};
  } else if constexpr (N == 2) {
    return T{init, init};
  } else if constexpr (N == 3) {
    return T{init, init, init};
  } else if constexpr (N == 4) {
    return T{init, init, init, init};
  } else if constexpr (N == 8) {
    return T{init, init, init, init, init, init, init, init};
  } else if constexpr (N == 16) {
    return T{init, init, init, init, init, init,
             init, init, init, init, init, init,
             init, init, init, init};
  }

  static_assert(true, "invalide vector type!");
}

template<typename T, int N>
bool compare_type(s::vec<T, N> v1, s::vec<T, N> v2) {
  bool ret = true;
  if constexpr (1 <= N)
    ret &= v1.s0() == v2.s0();
  if constexpr (2 <= N)
    ret &= v1.s1() == v2.s1();
  if constexpr (3 <= N)
    ret &= v1.s2() == v2.s2();
  if constexpr (4 <= N)
    ret &= v1.s3() == v2.s3();
  if constexpr (8 <= N) {
    ret &= v1.s4() == v2.s4();
    ret &= v1.s5() == v2.s5();
    ret &= v1.s6() == v2.s6();
    ret &= v1.s7() == v2.s7();
  }
  if constexpr (16 <= N) {
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

template<typename T>
bool compare_type(T x1, T x2) {
  return x1 == x2;
}

/**
 * ScopedMicrobenchmark benchmarking group_inclusive_scan_ptr
 */
template <typename DataT, int Iterations = 512>
class ScopedMicroBenchGroupInclusiveScanPtr {
protected:
  BenchmarkArgs args;

  std::vector<DataT> input;
  std::vector<DataT> output;
  std::vector<DataT> devNull;
  PrefetchedBuffer<DataT, 1> input_buf;
  PrefetchedBuffer<DataT, 1> output_buf;
  PrefetchedBuffer<DataT, 1> devNull_buf;

public:
  ScopedMicroBenchGroupInclusiveScanPtr(const BenchmarkArgs& _args) : args(_args) {}

  void setup() {
    input.resize(args.problem_size, initialize_type<DataT>(1));
    output.resize(args.problem_size, initialize_type<DataT>(1));
    devNull.resize(args.problem_size, initialize_type<DataT>(1));

    output_buf.initialize(args.device_queue, output.data(), s::range<1>(input.size()));
    devNull_buf.initialize(args.device_queue, devNull.data(), s::range<1>(input.size()));
    input_buf.initialize(args.device_queue, input.data(), s::range<1>(input.size()));
  }

  void run(std::vector<cl::sycl::event>& events) {
    size_t num_groups = 1;
    size_t problem_size = args.problem_size;
    events.push_back(args.device_queue.submit([&](cl::sycl::handler& cgh) {
      auto in = input_buf.template get_access<s::access::mode::read>(cgh);
      auto out = output_buf.template get_access<s::access::mode::discard_write>(cgh);
      auto devNull = output_buf.template get_access<s::access::mode::discard_write>(cgh);

      cgh.parallel<ScopedMicroBenchGroupInclusiveScanPtr<DataT, Iterations>>(
          s::range<1>{num_groups}, s::range<1>{args.local_size}, [=](s::group<1> g, s::physical_item<1> pitem) {
              DataT d = initialize_type<DataT>(-1);
              size_t gid = pitem.get_global_id(0);
              auto start = in.get_pointer();
              auto end = start + static_cast<size_t>(problem_size);

              auto result = devNull.get_pointer();
              if (g.get_linear() == 0)
                result = out.get_pointer();

              for(int i = 1; i <= Iterations; ++i) {
                s::detail::leader_inclusive_scan(g, start.get(), end.get(), result.get(), s::plus<DataT>());
              }
            });
    }));
  }

  bool verify(VerificationSetting& ver) {
    auto result = output_buf.template get_access<s::access::mode::read>();
    DataT expected = initialize_type<DataT>(0);

    for(size_t i = 0; i < args.problem_size; ++i) {
      expected += initialize_type<DataT>(1);
      if(!compare_type(result[i], expected)) {
        std::cout << i << ":" << type_to_string(expected) << ":" << type_to_string(result[i]) << std::endl;
        return false;
      }
    }

    return true;
  }

  static std::string getBenchmarkName() {
    std::stringstream name;
    name << "GroupFunctionBench_Inclusive_Scan_Ptr_";
    name << ReadableTypename<DataT>::name << "_";
    name << Iterations;
    return name.str();
  }
};

int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);

  //app.run<ScopedMicroBenchGroupInclusiveScanPtr<int>>();
  app.run<ScopedMicroBenchGroupInclusiveScanPtr<long long>>();
  app.run<ScopedMicroBenchGroupInclusiveScanPtr<float>>();
  app.run<ScopedMicroBenchGroupInclusiveScanPtr<double>>();
  //app.run<ScopedMicroBenchGroupInclusiveScanPtr<cl::sycl::vec<int, 1>>>();
  //app.run<ScopedMicroBenchGroupInclusiveScanPtr<cl::sycl::vec<unsigned char, 4>>>();
  //app.run<ScopedMicroBenchGroupInclusiveScanPtr<cl::sycl::vec<int, 4>>>();
  app.run<ScopedMicroBenchGroupInclusiveScanPtr<cl::sycl::vec<int, 8>>>();
  app.run<ScopedMicroBenchGroupInclusiveScanPtr<cl::sycl::vec<float, 1>>>();
  app.run<ScopedMicroBenchGroupInclusiveScanPtr<cl::sycl::vec<double, 2>>>();
  app.run<ScopedMicroBenchGroupInclusiveScanPtr<cl::sycl::vec<float, 4>>>();
  app.run<ScopedMicroBenchGroupInclusiveScanPtr<cl::sycl::vec<float, 8>>>();
  return 0;
}
