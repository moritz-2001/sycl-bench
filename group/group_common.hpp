#pragma once
#include "common.h"
#include "group_common.hpp"
#include <iostream>

namespace s = cl::sycl;

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
  return T{init};
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
  return x1 == x2;
}
